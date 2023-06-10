import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
from torchvision import transforms

from lib.pymafx.core import constants


def transform_to_tensor(res, mean=None, std=None, is_tensor=False):
    all_ops = []
    if res is not None:
        all_ops.append(transforms.Resize(size=res))
    if not is_tensor:
        all_ops.append(transforms.ToTensor())
    if mean is not None and std is not None:
        all_ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(all_ops)


def get_affine_matrix_wh(w1, h1, w2, h2):

    transl = torch.tensor([(w2 - w1) / 2.0, (h2 - h1) / 2.0]).unsqueeze(0)
    center = torch.tensor([w1 / 2.0, h1 / 2.0]).unsqueeze(0)
    scale = torch.min(torch.tensor([w2 / w1, h2 / h1])).repeat(2).unsqueeze(0)
    M = get_affine_matrix2d(transl, center, scale, angle=torch.tensor([0.]))

    return M


def get_affine_matrix_box(boxes, w2, h2):

    # boxes [left, top, right, bottom]
    width = boxes[:, 2] - boxes[:, 0]    #(N,)
    height = boxes[:, 3] - boxes[:, 1]    #(N,)
    center = torch.tensor([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                           (boxes[:, 1] + boxes[:, 3]) / 2.0]).T    #(N,2)
    scale = torch.min(torch.tensor([w2 / width, h2 / height]),
                      dim=0)[0].unsqueeze(1).repeat(1, 2) * 0.9    #(N,2)
    transl = torch.cat([w2 / 2.0 - center[:, 0:1], h2 / 2.0 - center[:, 1:2]], dim=1)    #(N,2)
    M = get_affine_matrix2d(transl, center, scale, angle=torch.tensor([
        0.,
    ] * transl.shape[0]))

    return M


def load_img(img_file):

    if img_file.endswith("exr"):
        img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # considering non 8-bit image
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float(), img.shape[:2]


def get_keypoints(image):
    def collect_xyv(x, body=True):
        lmk = x.landmark
        all_lmks = []
        for i in range(len(lmk)):
            visibility = lmk[i].visibility if body else 1.0
            all_lmks.append(torch.Tensor([lmk[i].x, lmk[i].y, lmk[i].z, visibility]))
        return torch.stack(all_lmks).view(-1, 4)

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
    ) as holistic:
        results = holistic.process(image)

    fake_kps = torch.zeros(33, 4)

    result = {}
    result["body"] = collect_xyv(results.pose_landmarks) if results.pose_landmarks else fake_kps
    result["lhand"] = collect_xyv(
        results.left_hand_landmarks, False
    ) if results.left_hand_landmarks else fake_kps
    result["rhand"] = collect_xyv(
        results.right_hand_landmarks, False
    ) if results.right_hand_landmarks else fake_kps
    result["face"] = collect_xyv(
        results.face_landmarks, False
    ) if results.face_landmarks else fake_kps

    return result


def get_pymafx(image, landmarks):

    # image [3,512,512]

    item = {
        'img_body': F.interpolate(image.unsqueeze(0), size=224, mode='bicubic',
                                  align_corners=True)[0]
    }

    for part in ['lhand', 'rhand', 'face']:
        kp2d = landmarks[part]
        kp2d_valid = kp2d[kp2d[:, 3] > 0.]
        if len(kp2d_valid) > 0:
            bbox = [
                min(kp2d_valid[:, 0]),
                min(kp2d_valid[:, 1]),
                max(kp2d_valid[:, 0]),
                max(kp2d_valid[:, 1])
            ]
            center_part = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
            scale_part = 2. * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

        # handle invalid part keypoints
        if len(kp2d_valid) < 1 or scale_part < 0.01:
            center_part = [0, 0]
            scale_part = 0.5
            kp2d[:, 3] = 0

        center_part = torch.tensor(center_part).float()

        theta_part = torch.zeros(1, 2, 3)
        theta_part[:, 0, 0] = scale_part
        theta_part[:, 1, 1] = scale_part
        theta_part[:, :, -1] = center_part

        grid = F.affine_grid(theta_part, torch.Size([1, 3, 224, 224]), align_corners=False)
        img_part = F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0).float()

        item[f'img_{part}'] = img_part

        theta_i_inv = torch.zeros_like(theta_part)
        theta_i_inv[:, 0, 0] = 1. / theta_part[:, 0, 0]
        theta_i_inv[:, 1, 1] = 1. / theta_part[:, 1, 1]
        theta_i_inv[:, :, -1] = -theta_part[:, :, -1] / theta_part[:, 0, 0].unsqueeze(-1)
        item[f'{part}_theta_inv'] = theta_i_inv[0]

    return item


def remove_floats(mask):

    # 1. find all the contours
    # 2. fillPoly "True" for the largest one
    # 3. fillPoly "False" for its childrens

    new_mask = np.zeros(mask.shape)
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_index = sorted(range(len(cnts)), key=lambda k: cv2.contourArea(cnts[k]), reverse=True)
    body_cnt = cnts[cnt_index[0]]
    childs_cnt_idx = np.where(np.array(hier)[0, :, -1] == cnt_index[0])[0]
    childs_cnt = [cnts[idx] for idx in childs_cnt_idx]
    cv2.fillPoly(new_mask, [body_cnt], 1)
    cv2.fillPoly(new_mask, childs_cnt, 0)

    return new_mask


def process_image(img_file, hps_type, single, input_res, detector):

    img_raw, (in_height, in_width) = load_img(img_file)
    tgt_res = input_res * 2
    M_square = get_affine_matrix_wh(in_width, in_height, tgt_res, tgt_res)
    img_square = warp_affine(
        img_raw,
        M_square[:, :2], (tgt_res, ) * 2,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # detection for bbox
    predictions = detector(img_square / 255.)[0]

    if single:
        top_score = predictions["scores"][predictions["labels"] == 1].max()
        human_ids = torch.where(predictions["scores"] == top_score)[0]
    else:
        human_ids = torch.logical_and(predictions["labels"] == 1,
                                      predictions["scores"] > 0.9).nonzero().squeeze(1)

    boxes = predictions["boxes"][human_ids, :].detach().cpu().numpy()
    masks = predictions["masks"][human_ids, :, :].permute(0, 2, 3, 1).detach().cpu().numpy()

    M_crop = get_affine_matrix_box(boxes, input_res, input_res)

    img_icon_lst = []
    img_crop_lst = []
    img_hps_lst = []
    img_mask_lst = []
    landmark_lst = []
    hands_visibility_lst = []
    img_pymafx_lst = []

    uncrop_param = {
        "ori_shape": [in_height, in_width], "box_shape": [input_res, input_res], "square_shape":
        [tgt_res, tgt_res], "M_square": M_square, "M_crop": M_crop
    }

    for idx in range(len(boxes)):

        # mask out the pixels of others
        if len(masks) > 1:
            mask_detection = (masks[np.arange(len(masks)) != idx]).max(axis=0)
        else:
            mask_detection = masks[0] * 0.

        img_square_rgba = torch.cat([
            img_square.squeeze(0).permute(1, 2, 0),
            torch.tensor(mask_detection < 0.4) * 255
        ],
                                    dim=2)

        img_crop = warp_affine(
            img_square_rgba.unsqueeze(0).permute(0, 3, 1, 2),
            M_crop[idx:idx + 1, :2], (input_res, ) * 2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

        # get accurate person segmentation mask
        img_rembg = remove(img_crop, post_process_mask=True, session=new_session("u2net"))
        img_mask = remove_floats(img_rembg[:, :, [3]])

        mean_icon = std_icon = (0.5, 0.5, 0.5)
        img_np = (img_rembg[..., :3] * img_mask).astype(np.uint8)
        img_icon = transform_to_tensor(512, mean_icon, std_icon)(
            Image.fromarray(img_np)
        ) * torch.tensor(img_mask).permute(2, 0, 1)
        img_hps = transform_to_tensor(224, constants.IMG_NORM_MEAN,
                                      constants.IMG_NORM_STD)(Image.fromarray(img_np))

        landmarks = get_keypoints(img_np)

        # get hands visibility
        hands_visibility = [True, True]
        if landmarks['lhand'][:, -1].mean() == 0.:
            hands_visibility[0] = False
        if landmarks['rhand'][:, -1].mean() == 0.:
            hands_visibility[1] = False
        hands_visibility_lst.append(hands_visibility)

        if hps_type == 'pymafx':
            img_pymafx_lst.append(
                get_pymafx(
                    transform_to_tensor(512, constants.IMG_NORM_MEAN,
                                        constants.IMG_NORM_STD)(Image.fromarray(img_np)), landmarks
                )
            )

        img_crop_lst.append(torch.tensor(img_crop).permute(2, 0, 1) / 255.0)
        img_icon_lst.append(img_icon)
        img_hps_lst.append(img_hps)
        img_mask_lst.append(torch.tensor(img_mask[..., 0]))
        landmark_lst.append(landmarks['body'])

    # required image tensors / arrays

    # img_icon  (tensor): (-1, 1),          [3,512,512]
    # img_hps   (tensor): (-2.11, 2.44),    [3,224,224]

    # img_np    (array): (0, 255),          [512,512,3]
    # img_rembg (array): (0, 255),          [512,512,4]
    # img_mask  (array): (0, 1),            [512,512,1]
    # img_crop  (array): (0, 255),          [512,512,4]

    return_dict = {
        "img_icon": torch.stack(img_icon_lst).float(),    #[N, 3, res, res]
        "img_crop": torch.stack(img_crop_lst).float(),    #[N, 4, res, res]               
        "img_hps": torch.stack(img_hps_lst).float(),    #[N, 3, res, res]
        "img_raw": img_raw,    #[1, 3, H, W]
        "img_mask": torch.stack(img_mask_lst).float(),    #[N, res, res]
        "uncrop_param": uncrop_param,
        "landmark": torch.stack(landmark_lst),    #[N, 33, 4]
        "hands_visibility": hands_visibility_lst,
    }

    img_pymafx = {}

    if len(img_pymafx_lst) > 0:
        for idx in range(len(img_pymafx_lst)):
            for key in img_pymafx_lst[idx].keys():
                if key not in img_pymafx.keys():
                    img_pymafx[key] = [img_pymafx_lst[idx][key]]
                else:
                    img_pymafx[key] += [img_pymafx_lst[idx][key]]

        for key in img_pymafx.keys():
            img_pymafx[key] = torch.stack(img_pymafx[key]).float()

        return_dict.update({"img_pymafx": img_pymafx})

    return return_dict


def blend_rgb_norm(norms, data):

    # norms [N, 3, res, res]
    masks = (norms.sum(dim=1) != norms[0, :, 0, 0].sum()).float().unsqueeze(1)
    norm_mask = F.interpolate(
        torch.cat([norms, masks], dim=1).detach(),
        size=data["uncrop_param"]["box_shape"],
        mode="bilinear",
        align_corners=False
    )
    final = data["img_raw"].type_as(norm_mask)

    for idx in range(len(norms)):

        norm_pred = (norm_mask[idx:idx + 1, :3, :, :] + 1.0) * 255.0 / 2.0
        mask_pred = norm_mask[idx:idx + 1, 3:4, :, :].repeat(1, 3, 1, 1)

        norm_ori = unwrap(norm_pred, data["uncrop_param"], idx)
        mask_ori = unwrap(mask_pred, data["uncrop_param"], idx)

        final = final * (1.0 - mask_ori) + norm_ori * mask_ori

    return final.detach().cpu()


def unwrap(image, uncrop_param, idx):

    device = image.device

    img_square = warp_affine(
        image,
        torch.inverse(uncrop_param["M_crop"])[idx:idx + 1, :2].to(device),
        uncrop_param["square_shape"],
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    img_ori = warp_affine(
        img_square,
        torch.inverse(uncrop_param["M_square"])[:, :2].to(device),
        uncrop_param["ori_shape"],
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return img_ori
