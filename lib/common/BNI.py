from lib.common.BNI_utils import (verts_inverse_transform, depth_inverse_transform,
                                  bilateral_normal_integration, bilateral_normal_integration_new,
                                  mean_value_cordinates, find_contour, depth2png, dispCorres,
                                  repeat_pts, get_dst_mat)

import torch
import os, cv2
import trimesh
import numpy as np
import os.path as osp
from scipy.optimize import linear_sum_assignment
from pytorch3d.structures import Meshes
from pytorch3d.io import IO


class BNI:

    def __init__(self, dir_path, name, BNI_dict, device, mvc=False):

        self.scale = 256.0

        self.normal_front = BNI_dict["normal_F"]
        self.normal_back = BNI_dict["normal_B"]
        self.mask = BNI_dict["mask"]

        self.depth_front = BNI_dict["depth_F"]
        self.depth_back = BNI_dict["depth_B"]
        self.depth_mask = BNI_dict["depth_mask"]

        # hparam:
        # k --> smaller, keep continuity
        # lambda --> larger, more depth-awareness
        self.k = 1e-3
        self.lambda1 = 1e-2
        self.name = name
        self.thickness = 0.0

        self.F_B_surface = None
        self.F_B_trimesh = None
        self.F_depth = None
        self.B_depth = None

        self.device = device
        self.export_dir = dir_path

        if mvc:
            self.mvp_expand(self.mask, self.depth_mask, self.depth_front, self.depth_back)

    def mvp_expand(self, cloth_mask, body_mask, depth_front, depth_back):

        # contour [num_contour, 2]
        # extract contour points from body and cloth masks
        contour_cloth = find_contour(cloth_mask, method='all')
        contour_body = find_contour(body_mask, method='simple')

        # correspondence body_contour --> cloth_contour
        # construct distance_matrix and solve bipartite matching
        dst_mat = get_dst_mat(contour_body, contour_cloth)
        _, cloth_ind = linear_sum_assignment(dst_mat)
        dispCorres(512, contour_body, contour_cloth, cloth_ind, self.export_dir)

        # weights [num_innners, num_body_contour]
        # compute barycentric weights from all the inner points from body mask
        # Law of cosines: https://en.wikipedia.org/wiki/Law_of_cosines
        # cos_beta = (a^2+c^2-b^2)/2ac
        body_inner_pts = np.array(np.where(body_mask)).transpose()[:, [1, 0]]
        body_inner_pts = body_inner_pts[~repeat_pts(body_inner_pts, contour_body)]
        weights_body = mean_value_cordinates(body_inner_pts, contour_body)

        # fill depth values on the cloth_depth using correspondence
        cloth_inner_pts = np.around(weights_body @ contour_cloth[cloth_ind]).clip(
            0, cloth_mask.shape[0] - 1).astype(np.int32)

        # fill depth holes
        cloth_hole_pts = np.array(np.where(cloth_mask)).transpose()[:, [1, 0]]
        cloth_hole_pts = cloth_hole_pts[~repeat_pts(cloth_hole_pts,
                                                    cloth_inner_pts)]  # remove already filled pts
        cloth_hole_pts = cloth_hole_pts[~repeat_pts(cloth_hole_pts,
                                                    contour_cloth)]  # remove contour pts
        weights_cloth = mean_value_cordinates(cloth_hole_pts, contour_cloth[cloth_ind])

        # backward from hole_cloth_pts into body_depth_map
        body_backward_pts = np.around(weights_cloth @ contour_body).clip(0, body_mask.shape[0] -
                                                                         1).astype(np.int32)

        # complete back and front depth maps
        left_idx_x = cloth_inner_pts[:, 1].tolist() + cloth_hole_pts[:, 1].tolist()
        left_idx_y = cloth_inner_pts[:, 0].tolist() + cloth_hole_pts[:, 0].tolist()
        right_idx_x = body_inner_pts[:, 1].tolist() + body_backward_pts[:, 1].tolist()
        right_idx_y = body_inner_pts[:, 0].tolist() + body_backward_pts[:, 0].tolist()

        depth_front = np.zeros_like(self.depth_front)
        depth_back = np.zeros_like(self.depth_back)

        depth_front[left_idx_x, left_idx_y] = self.depth_front[right_idx_x, right_idx_y]
        depth_back[left_idx_x, left_idx_y] = self.depth_back[right_idx_x, right_idx_y]

        self.depth_back = depth_back
        self.depth_front = depth_front

        cv2.imwrite(osp.join(self.export_dir, "depth_front.png"),
                    depth2png(self.depth_front / self.scale + 100.))
        cv2.imwrite(osp.join(self.export_dir, "depth_back.png"),
                    depth2png(-(self.depth_back / self.scale - 100.)))

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface(self, idx, verbose=True):

        F_verts, F_faces, F_depth = bilateral_normal_integration_new(
            normal_map=self.normal_front,
            normal_mask=self.mask,
            k=self.k,
            lambda1=self.lambda1,
            depth_map=self.depth_front,
            depth_mask=self.depth_mask,
            label="Front",
            verbose=verbose
        )

        B_verts, B_faces, B_depth = bilateral_normal_integration_new(
            normal_map=self.normal_back,
            normal_mask=self.mask,
            k=self.k,
            lambda1=self.lambda1,
            depth_map=self.depth_back,
            depth_mask=self.depth_mask,
            label="Back",
            verbose=verbose
        )

        F_verts = verts_inverse_transform(F_verts, self.scale)
        B_verts = verts_inverse_transform(B_verts, self.scale)

        self.F_depth = depth_inverse_transform(F_depth, self.scale)
        self.B_depth = depth_inverse_transform(B_depth, self.scale)

        # thickness shift from BiNI surfaces
        depth_offset = self.F_depth - self.B_depth
        depth_mask = cv2.GaussianBlur((~torch.isnan(depth_offset)).numpy().astype(np.uint8) * 255,
                                      (3, 3), 0)
        contour_uv = cv2.findContours((depth_mask == 255).astype(np.uint8), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
        self.thickness = depth_offset[contour_uv[:, 1], contour_uv[:, 0]].topk(100,
                                                                          largest=False)[0].mean()
        
        F_verts[:, 2] -= self.thickness / 2
        B_verts[:, 2] += self.thickness / 2
        self.F_depth[~torch.isnan(self.F_depth)] -= self.thickness / 2
        self.B_depth[~torch.isnan(self.B_depth)] += self.thickness / 2

        F_B_verts = torch.cat((F_verts, B_verts), dim=0)
        F_B_faces = torch.cat((F_faces, B_faces + F_faces.max() + 1), dim=0)

        self.F_B_surfaces = Meshes(F_B_verts.float().unsqueeze(0),
                                   F_B_faces.long().unsqueeze(0)).to(self.device)

        self.F_B_trimesh = trimesh.Trimesh(F_B_verts.float(),
                                           F_B_faces.long(),
                                           process=False,
                                           maintain_order=True)

        # IO().save_mesh(
        #     self.F_B_surfaces,
        #     os.path.join(self.export_dir, f"{self.name}_{idx}_F_B_surface.obj"),
        # )


if __name__ == "__main__":

    dir_path = "./results/icon-mvp/BNI"
    name = "e1e7622af7074a022f5d96dc16672517"
    BNI_obj = BNI(dir_path,
                  name,
                  in_tensor=torch.load(osp.join(dir_path, f"{name}_in_tensor.pt")),
                  device=torch.device("cuda:0"),
                  mvc=False)
    BNI_obj.extract_surface()
