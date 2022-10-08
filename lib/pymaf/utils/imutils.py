import cv2
import torch
import numpy as np
import torch.nn.functional as F
from rembg import remove
from PIL import Image
from torchvision.models import detection

from lib.pymaf.core import constants
from lib.pymaf.utils.streamer import aug_matrix
from lib.common.cloth_extraction import load_segmentation
from torchvision import transforms

image_to_icon_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_to_pymaf_tensor = transforms.Compose([
    transforms.Resize(size=224),
    transforms.Normalize(mean=constants.IMG_NORM_MEAN,
                         std=constants.IMG_NORM_STD),
])

image_to_hybrik_tensor = transforms.Compose([
    transforms.Resize(256),
    transforms.Normalize(mean=(0.406, 0.457, 0.480), std=(0.225, 0.224, 0.229))
])


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

    import mediapipe as mp

    def collect_xyv(x):
        lmk = x.landmark
        all_lmks = []
        for i in range(len(lmk)):
            all_lmks.append(
                torch.Tensor([lmk[i].x, lmk[i].y, lmk[i].z,
                              lmk[i].visibility]))
        return torch.stack(all_lmks).view(-1, 4)

    with mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
    ) as holistic:
        results = holistic.process(image)

    return collect_xyv(results.pose_landmarks)


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
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)
    cnt_index = sorted(range(len(cnts)),
                       key=lambda k: cv2.contourArea(cnts[k]),
                       reverse=True)
    body_cnt = cnts[cnt_index[0]]
    childs_cnt_idx = np.where(np.array(hier)[0, :, -1] == cnt_index[0])[0]
    childs_cnt = [cnts[idx] for idx in childs_cnt_idx]
    cv2.fillPoly(new_mask, [body_cnt], 1)
    cv2.fillPoly(new_mask, childs_cnt, 0)

    return new_mask


def process_image(img_file,
                  use_seg,
                  hps_type,
                  input_res=512,
                  device=None,
                  seg_path=None):

    img_raw = load_img(img_file)

    in_height, in_width = img_raw.shape[:2]
    M = aug_matrix(in_width, in_height, input_res * 2, input_res * 2)

    # from rectangle to square by padding (input_res*2, input_res*2)
    img_square = cv2.warpAffine(img_raw,
                                M[0:2, :], (input_res * 2, input_res * 2),
                                flags=cv2.INTER_CUBIC)

    # detection for bbox
    detector = detection.maskrcnn_resnet50_fpn(
        weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights)
    detector.eval()
    predictions = detector(
        [torch.from_numpy(img_square).permute(2, 0, 1) / 255.])[0]
    human_ids = torch.logical_and(
        predictions["labels"] == 1,
        predictions["scores"] > 0.9).nonzero().squeeze(1)
    boxes = predictions["boxes"][human_ids, :].detach().cpu().numpy()

    masks = predictions["masks"][human_ids, :, :].permute(
        0, 2, 3, 1).detach().cpu().numpy()

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

    uncrop_param = {
        "center": center,
        "scale": scale,
        "ori_shape": [in_height, in_width],
        "box_shape": [input_res, input_res],
        "crop_shape": [input_res * 2, input_res * 2, 3],
        "M": M,
    }

    for idx in range(len(boxes)):

        mask_detection = (masks[np.arange(len(masks)) != idx]).max(axis=0)
        mask_erosion = (mask_detection < 0.6).astype(np.uint8) * 255

        img_crop, cropping_parameters = crop(
            np.concatenate([img_square, mask_erosion], axis=2), center[idx],
            scale[idx], [input_res, input_res])

        img_rembg = remove(img_crop[:, :, :4])
        img_mask = torch.tensor(remove_floats(img_rembg[:, :, 3] > 200))

        # required image tensors / arrays
        img_icon = image_to_icon_tensor(
            img_rembg[:, :, :3]) * img_mask  # [-1,1]
        img_hps = (img_icon + 1.0) * 0.5  # [0,1]
        img_np = (img_hps * 255).permute(1, 2, 0).numpy().astype(
            np.uint8)  # [0, 255]

        if hps_type == "bev":
            img_hps = img_np[:, :, ::-1]
        elif hps_type == "hybrik":
            img_hps = image_to_hybrik_tensor(img_hps).unsqueeze(0).to(device)
        elif hps_type in ["pymaf", "pixie"]:
            img_hps = image_to_pymaf_tensor(img_hps).unsqueeze(0).to(device)
        else:
            print(f"No {hps_type} HPS")

        # Image.fromarray(img_np).show()

        img_crop_lst.append(img_crop[:, :, :3].transpose(2, 0, 1) / 255.)
        img_icon_lst.append(img_icon)
        img_hps_lst.append(img_hps)
        img_mask_lst.append(img_mask)
        uncrop_param_lst.append(uncrop_param)
        landmark_lst.append(get_keypoints(img_np))

    return_dict = {
        "img_icon": torch.stack(img_icon_lst).float(),  #[N, 3, res, res]
        "img_crop":
        torch.tensor(np.stack(img_crop_lst)),  #[N, res, res, 3]               
        "img_hps": torch.cat(img_hps_lst).float(),  #[N, 3, res, res]
        "img_raw": img_raw,  #[H, W, 3]
        "img_mask": torch.cat(img_mask_lst).float(),  #[N, res, res]
        "uncrop_param": uncrop_param,
        "landmark": torch.stack(landmark_lst),  #[N, 33, 4]
    }

    if seg_path is not None:
        segmentations = load_segmentation(seg_path, (in_height, in_width))
        seg_coord_normalized = []
        for seg in segmentations:
            coord_normalized = []
            for xy in seg["coordinates"]:
                xy_h = np.vstack((xy[:, 0], xy[:, 1], np.ones(len(xy)))).T
                warped_indeces = M[0:2, :] @ xy_h[:, :, None]
                warped_indeces = np.array(warped_indeces).astype(int)
                warped_indeces.resize((warped_indeces.shape[:2]))

                # cropped_indeces = crop_segmentation(warped_indeces, center, scale, (input_res, input_res), img_np.shape)
                cropped_indeces = crop_segmentation(warped_indeces,
                                                    (input_res, input_res),
                                                    cropping_parameters)

                indices = np.vstack(
                    (cropped_indeces[:, 0], cropped_indeces[:, 1])).T

                # Convert to NDC coordinates
                seg_cropped_normalized = 2 * (indices / input_res) - 1
                # Don't know why we need to divide by 50 but it works ¯\_(ツ)_/¯ (probably some scaling factor somewhere)
                # Divide only by 45 on the horizontal axis to take the curve of the human body into account
                seg_cropped_normalized[:,
                                       0] = (1 /
                                             40) * seg_cropped_normalized[:, 0]
                seg_cropped_normalized[:,
                                       1] = (1 /
                                             50) * seg_cropped_normalized[:, 1]
                coord_normalized.append(seg_cropped_normalized)

            seg["coord_normalized"] = coord_normalized
            seg_coord_normalized.append(seg)

        return_dict.update({"segmentations": seg_coord_normalized})

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

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]
    new_img = F.interpolate(
        torch.tensor(new_img).permute(2, 0, 1).unsqueeze(0),
        res,
        mode='bicubic').permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)

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

    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1],
                                                        new_x[0]:new_x[1]]

    return new_img


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([
        [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
        [np.sin(np.deg2rad(-rot)),
         np.cos(np.deg2rad(-rot)), 0],
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
        image = cv2.circle(image,
                           (int(joint[0] * img_w), int(joint[1] * img_h)), 5,
                           color)

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
        if (ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0
                or br[1] < 0):
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
            target[joint_id][img_y[0]:img_y[1],
                             img_x[0]:img_x[1]] = g[g_y[0]:g_y[1],
                                                    g_x[0]:g_x[1]]

    return target, target_weight


if __name__ == "__main__":

    mask = remove(cv2.imread("log/ECON/2.png"))[:, :, 3] > 200
    mask = remove_floats(mask)
    Image.fromarray(mask.astype(np.uint8) * 255).show()
