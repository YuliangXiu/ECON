import cv2
import mediapipe as mp
import torch
import numpy as np
import torch.nn.functional as F
from rembg import remove
from rembg.session_factory import new_session
from PIL import Image
from torchvision.models import detection

from lib.pymafx.core import constants
from lib.common.cloth_extraction import load_segmentation
from torchvision import transforms


def transform_to_tensor(res, mean=None, std=None, is_tensor=False):
    all_ops = []
    if res is not None:
        all_ops.append(transforms.Resize(size=res))
    if not is_tensor:
        all_ops.append(transforms.ToTensor())
    if mean is not None and std is not None:
        all_ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(all_ops)


def aug_matrix(w1, h1, w2, h2):
    dx = (w2 - w1) / 2.0
    dy = (h2 - h1) / 2.0

    matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

    scale = np.min([float(w2) / w1, float(h2) / h1])

    M = get_affine_matrix(center=(w2 / 2.0, h2 / 2.0), translate=(0, 0), scale=scale)

    M = np.array(M + [0.0, 0.0, 1.0]).reshape(3, 3)
    M = M.dot(matrix_trans)

    return M


def get_affine_matrix(center, translate, scale):
    cx, cy = center
    tx, ty = translate

    M = [1, 0, 0, 0, 1, 0]
    M = [x * scale for x in M]

    # Apply translation and of center translation: RSS * C^-1
    M[2] += M[0] * (-cx) + M[1] * (-cy)
    M[5] += M[3] * (-cx) + M[4] * (-cy)

    # Apply center translation: T * C * RSS * C^-1
    M[2] += cx + tx
    M[5] += cy + ty
    return M


def load_img(img_file):

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def get_keypoints(image):

    def collect_xyv(x):
        lmk = x.landmark
        all_lmks = []
        for i in range(len(lmk)):
            all_lmks.append(torch.Tensor([lmk[i].x, lmk[i].y, lmk[i].z, lmk[i].visibility]))
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
        results.left_hand_landmarks) if results.left_hand_landmarks else fake_kps
    result["rhand"] = collect_xyv(
        results.right_hand_landmarks) if results.right_hand_landmarks else fake_kps
    result["face"] = collect_xyv(results.face_landmarks) if results.face_landmarks else fake_kps
    
    return result


def get_pymafx(image, landmarks):

    # image [3,512,512]

    item = {
        'img_body':
            F.interpolate(image.unsqueeze(0), size=224, mode='bicubic', align_corners=True)[0]
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


def expand_bbox(bbox, width, height, ratio=0.1):

    bbox = np.around(bbox).astype(np.int16)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    bbox[1] = max(bbox[1] - bbox_height * ratio, 0)
    bbox[3] = min(bbox[3] + bbox_height * ratio, height)
    bbox[0] = max(bbox[0] - bbox_width * ratio, 0)
    bbox[2] = min(bbox[2] + bbox_width * ratio, width)

    return bbox


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


def process_image(img_file, hps_type, single, input_res=512):

    img_raw = load_img(img_file)

    in_height, in_width = img_raw.shape[:2]
    M = aug_matrix(in_width, in_height, input_res * 2, input_res * 2)

    # from rectangle to square by padding (input_res*2, input_res*2)
    img_square = cv2.warpAffine(img_raw,
                                M[0:2, :], (input_res * 2, input_res * 2),
                                flags=cv2.INTER_CUBIC)

    # detection for bbox
    detector = detection.maskrcnn_resnet50_fpn(weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights)
    detector.eval()
    predictions = detector([torch.from_numpy(img_square).permute(2, 0, 1) / 255.])[0]
    
    if single:
        top_score = predictions["scores"][predictions["labels"]==1].max() 
        human_ids = torch.where(predictions["scores"]==top_score)[0]
    else:
        human_ids = torch.logical_and(predictions["labels"] == 1,
                                    predictions["scores"] > 0.9).nonzero().squeeze(1)
    boxes = predictions["boxes"][human_ids, :].detach().cpu().numpy()

    masks = predictions["masks"][human_ids, :, :].permute(0, 2, 3, 1).detach().cpu().numpy()

    width = boxes[:, 2] - boxes[:, 0]  #(N,)
    height = boxes[:, 3] - boxes[:, 1]  #(N,)
    center = np.array([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                       (boxes[:, 1] + boxes[:, 3]) / 2.0]).T  #(N,2)
    scale = np.array([width, height]).max(axis=0) / 90.

    img_icon_lst = []
    img_crop_lst = []
    img_hps_lst = []
    img_mask_lst = []
    uncrop_param_lst = []
    landmark_lst = []
    img_pymafx_lst = []

    uncrop_param = {
        "center": center,
        "scale": scale,
        "ori_shape": [in_height, in_width],
        "box_shape": [input_res, input_res],
        "crop_shape": [input_res * 2, input_res * 2, 3],
        "M": M,
    }

    for idx in range(len(boxes)):

        # mask out the pixels of others
        if len(masks) > 1:
            mask_detection = (masks[np.arange(len(masks)) != idx]).max(axis=0)
        else:
            mask_detection = masks[0] * 0.

        img_crop, _ = crop(np.concatenate([img_square, (mask_detection < 0.4) * 255], axis=2),
                           center[idx], scale[idx], [input_res, input_res])

        # get accurate segmentation mask of focus person
        img_rembg = remove(img_crop, post_process_mask=True, session=new_session("u2net"))
        img_mask = remove_floats(img_rembg[:, :, [3]])

        # required image tensors / arrays

        # img_icon  (tensor): (-1, 1),          [3,512,512]
        # img_hps   (tensor): (-2.11, 2.44),    [3,224,224]

        # img_np    (array): (0, 255),          [512,512,3]
        # img_rembg (array): (0, 255),          [512,512,4]
        # img_mask  (array): (0, 1),            [512,512,1]
        # img_crop  (array): (0, 255),          [512,512,4]

        mean_icon = std_icon = (0.5, 0.5, 0.5)
        img_np = (img_rembg[..., :3] * img_mask).astype(np.uint8)
        img_icon = transform_to_tensor(512, mean_icon, std_icon)(
            Image.fromarray(img_np)) * torch.tensor(img_mask).permute(2, 0, 1)
        img_hps = transform_to_tensor(224, constants.IMG_NORM_MEAN,
                                      constants.IMG_NORM_STD)(Image.fromarray(img_np))

        landmarks = get_keypoints(img_np)

        if hps_type == 'pymafx':
            img_pymafx_lst.append(
                get_pymafx(
                    transform_to_tensor(512, constants.IMG_NORM_MEAN,
                                        constants.IMG_NORM_STD)(Image.fromarray(img_np)),
                    landmarks))

        img_crop_lst.append(torch.tensor(img_crop).permute(2, 0, 1) / 255.0)
        img_icon_lst.append(img_icon)
        img_hps_lst.append(img_hps)
        img_mask_lst.append(torch.tensor(img_mask[..., 0]))
        uncrop_param_lst.append(uncrop_param)
        landmark_lst.append(landmarks['body'])

    return_dict = {
        "img_icon": torch.stack(img_icon_lst).float(),  #[N, 3, res, res]
        "img_crop": torch.stack(img_crop_lst).float(),  #[N, 4, res, res]               
        "img_hps": torch.stack(img_hps_lst).float(),  #[N, 3, res, res]
        "img_raw": img_raw,  #[H, W, 3]
        "img_mask": torch.stack(img_mask_lst).float(),  #[N, res, res]
        "uncrop_param": uncrop_param,
        "landmark": torch.stack(landmark_lst),  #[N, 33, 4]
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


def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h = 100 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1

    return t


def transform(pt, center, scale, res, invert=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return np.around(new_pt[:2]).astype(np.int16)


def crop(img, center, scale, res):
    """Crop image according to the supplied bounding box."""

    img_height, img_width = img.shape[:2]

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))

    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img_width) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img_height) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(img_width, br[0])
    old_y = max(0, ul[1]), min(img_height, br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    new_img = F.interpolate(torch.tensor(new_img).permute(2, 0, 1).unsqueeze(0),
                            res,
                            mode='bilinear').permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)

    return new_img, (old_x, new_x, old_y, new_y, new_shape)


def crop_segmentation(org_coord, res, cropping_parameters):
    old_x, new_x, old_y, new_y, new_shape = cropping_parameters

    new_coord = np.zeros((org_coord.shape))
    new_coord[:, 0] = new_x[0] + (org_coord[:, 0] - old_x[0])
    new_coord[:, 1] = new_y[0] + (org_coord[:, 1] - old_y[0])

    new_coord[:, 0] = res[0] * (new_coord[:, 0] / new_shape[1])
    new_coord[:, 1] = res[1] * (new_coord[:, 1] / new_shape[0])

    return new_coord


def corner_align(ul, br):

    if ul[1] - ul[0] != br[1] - br[0]:
        ul[1] = ul[0] + br[1] - br[0]

    return ul, br


def uncrop(img, center, scale, orig_shape):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """

    res = img.shape[:2]

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # quick fix
    ul, br = corner_align(ul, br)

    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])

    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(crop_shape))

    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]

    return new_img


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([
        [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
        [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
        [0, 0, 1],
    ])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def flip_kp(kp, is_smpl=False):
    """Flip keypoints."""
    if len(kp) == 24:
        if is_smpl:
            flipped_parts = constants.SMPL_JOINTS_FLIP_PERM
        else:
            flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        if is_smpl:
            flipped_parts = constants.SMPL_J49_FLIP_PERM
        else:
            flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:, 0] = -kp[:, 0]
    return kp


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0) / (2 * ratio)

    return kp_2d


def visualize_landmarks(image, joints, color):

    img_w, img_h = image.shape[:2]

    for joint in joints:
        image = cv2.circle(image, (int(joint[0] * img_w), int(joint[1] * img_h)), 5, color)

    return image


def generate_heatmap(joints, heatmap_size, sigma=1, joints_vis=None):
    """
    param joints:  [num_joints, 3]
    param joints_vis: [num_joints, 3]
    return: target, target_weight(1: visible, 0: invisible)
    """
    num_joints = joints.shape[0]
    device = joints.device
    cur_device = torch.device(device.type, device.index)
    if not hasattr(heatmap_size, "__len__"):
        # width  height
        heatmap_size = [heatmap_size, heatmap_size]
    assert len(heatmap_size) == 2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    if joints_vis is not None:
        target_weight[:, 0] = joints_vis[:, 0]
    target = torch.zeros(
        (num_joints, heatmap_size[1], heatmap_size[0]),
        dtype=torch.float32,
        device=cur_device,
    )

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] * heatmap_size[0] + 0.5)
        mu_y = int(joints[joint_id][1] * heatmap_size[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if (ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        # x = np.arange(0, size, 1, np.float32)
        # y = x[:, np.newaxis]
        # x0 = y0 = size // 2
        # # The gaussian is not normalized, we want the center value to equal 1
        # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # g = torch.from_numpy(g.astype(np.float32))

        x = torch.arange(0, size, dtype=torch.float32, device=cur_device)
        y = x.unsqueeze(-1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight
