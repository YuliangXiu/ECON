from gettext import lngettext
from lib.common.BNI_utils import (depth2arr, tensor2arr,
                                  verts_inverse_transform,
                                  bilateral_normal_integration,
                                  mean_value_cordinates, find_contour,
                                  depth2png, dispCorres, repeat_pts,
                                  get_dst_mat)

import torch
import os, cv2
import trimesh
import numpy as np
import os.path as osp
from scipy.optimize import linear_sum_assignment
from pytorch3d.structures import Meshes
from pytorch3d.io import IO


class BNI:

    def __init__(self, dir_path, name, in_tensor, device, mvc=False):

        self.scale = 256.0

        self.normal_front = tensor2arr(in_tensor["normal_F"])
        self.normal_back = tensor2arr(in_tensor["normal_B"])
        self.mask = tensor2arr(in_tensor["image"], True) > 0.
        self.depth_mask = tensor2arr(in_tensor["T_normal_F"], True) > 0.

        self.depth_front = (depth2arr(in_tensor["depth_F"]) -
                            100.0) * self.scale
        self.depth_back = (100.0 -
                           depth2arr(in_tensor["depth_B"])) * self.scale

        # hparam
        self.k = 2
        self.lambda1 = 1e-4
        self._DEFAULT_MIN_TRIANGLE_AREA = 5e-3
        self.name = name

        self.F_B_surface = None
        self.F_B_trimesh = None
        self.device = device
        self.export_dir = dir_path

        if mvc:
            self.mvp_expand(self.mask, self.depth_mask, self.depth_front,
                            self.depth_back)

    def mvp_expand(self, cloth_mask, body_mask, depth_front, depth_back):

        # contour [num_contour, 2]
        # extract contour points from body and cloth masks
        contour_cloth = find_contour(cloth_mask, method='all')
        contour_body = find_contour(body_mask, method='simple')

        # correspondence body_contour --> cloth_contour
        # construct distance_matrix and solve bipartite matching
        dst_mat = get_dst_mat(contour_body, contour_cloth)
        _, cloth_ind = linear_sum_assignment(dst_mat)
        dispCorres(512, contour_body, contour_cloth, cloth_ind,
                   self.export_dir)

        # weights [num_innners, num_body_contour]
        # compute barycentric weights from all the inner points from body mask
        # Law of cosines: https://en.wikipedia.org/wiki/Law_of_cosines
        # cos_beta = (a^2+c^2-b^2)/2ac
        body_inner_pts = np.array(np.where(body_mask)).transpose()[:, [1, 0]]
        body_inner_pts = body_inner_pts[
            ~repeat_pts(body_inner_pts, contour_body)]
        weights_body = mean_value_cordinates(body_inner_pts, contour_body)

        # fill depth values on the cloth_depth using correspondence
        cloth_inner_pts = np.around(
            weights_body @ contour_cloth[cloth_ind]).clip(
                0, cloth_mask.shape[0] - 1).astype(np.int32)

        # fill depth holes
        cloth_hole_pts = np.array(np.where(cloth_mask)).transpose()[:, [1, 0]]
        cloth_hole_pts = cloth_hole_pts[~repeat_pts(
            cloth_hole_pts, cloth_inner_pts)]  # remove already filled pts
        cloth_hole_pts = cloth_hole_pts[~repeat_pts(
            cloth_hole_pts, contour_cloth)]  # remove contour pts
        weights_cloth = mean_value_cordinates(cloth_hole_pts,
                                              contour_cloth[cloth_ind])

        # backward from hole_cloth_pts into body_depth_map
        body_backward_pts = np.around(weights_cloth @ contour_body).clip(
            0, body_mask.shape[0] - 1).astype(np.int32)

        # complete back and front depth maps
        left_idx_x = cloth_inner_pts[:,
                                     1].tolist() + cloth_hole_pts[:,
                                                                  1].tolist()
        left_idx_y = cloth_inner_pts[:,
                                     0].tolist() + cloth_hole_pts[:,
                                                                  0].tolist()
        right_idx_x = body_inner_pts[:, 1].tolist(
        ) + body_backward_pts[:, 1].tolist()
        right_idx_y = body_inner_pts[:, 0].tolist(
        ) + body_backward_pts[:, 0].tolist()

        depth_front = np.zeros_like(self.depth_front)
        depth_back = np.zeros_like(self.depth_back)

        depth_front[left_idx_x, left_idx_y] = self.depth_front[right_idx_x,
                                                               right_idx_y]
        depth_back[left_idx_x, left_idx_y] = self.depth_back[right_idx_x,
                                                             right_idx_y]

        self.depth_mask = self.mask.copy()

        self.depth_back = depth_back
        self.depth_front = depth_front

        cv2.imwrite(osp.join(self.export_dir, "depth_front.png"),
                    depth2png(self.depth_front / self.scale + 100.))
        cv2.imwrite(osp.join(self.export_dir, "depth_back.png"),
                    depth2png(-(self.depth_back / self.scale - 100.)))


    @staticmethod
    def load_all(export_dir, name):

        F_verts = torch.load(os.path.join(export_dir, f"{name}_F_verts.pt"))
        F_faces = torch.load(os.path.join(export_dir, f"{name}_F_faces.pt"))
        B_verts = torch.load(os.path.join(export_dir, f"{name}_B_verts.pt"))
        B_faces = torch.load(os.path.join(export_dir, f"{name}_B_faces.pt"))

        return F_verts, F_faces, B_verts, B_faces

    @staticmethod
    def export_all(F_verts, F_faces, B_verts, B_faces, export_dir, name):

        torch.save(F_verts, os.path.join(export_dir, f"{name}_F_verts.pt"))
        torch.save(F_faces, os.path.join(export_dir, f"{name}_F_faces.pt"))
        torch.save(B_verts, os.path.join(export_dir, f"{name}_B_verts.pt"))
        torch.save(B_faces, os.path.join(export_dir, f"{name}_B_faces.pt"))

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface(self):

        # if not os.path.exists(
        #         os.path.join(self.export_dir, f"{self.name}_F_verts.pt")):
        if True:

            F_verts, F_faces = bilateral_normal_integration(
                normal_map=self.normal_front,
                normal_mask=self.mask,
                k=self.k,
                lambda1=self.lambda1,
                depth_map=self.depth_front,
                depth_mask=self.depth_mask,
                label="Front",
            )

            B_verts, B_faces = bilateral_normal_integration(
                normal_map=self.normal_back,
                normal_mask=self.mask,
                k=self.k,
                lambda1=self.lambda1,
                depth_map=self.depth_back,
                depth_mask=self.depth_mask,
                label="Back",
            )
            # self.export_all(F_verts, F_faces, B_verts, B_faces,
            #                 self.export_dir, self.name)
        else:
            F_verts, F_faces, B_verts, B_faces = self.load_all(
                self.export_dir, self.name)
            pass

        F_verts = verts_inverse_transform(F_verts, self.scale)
        B_verts = verts_inverse_transform(B_verts, self.scale)

        F_B_verts = torch.cat((F_verts, B_verts), dim=0)
        F_B_faces = torch.cat((F_faces, B_faces + F_faces.max() + 1), dim=0)

        self.F_B_surfaces = Meshes(F_B_verts.float().unsqueeze(0),
                                   F_B_faces.long().unsqueeze(0)).to(
                                       self.device)

        self.F_B_trimesh = trimesh.Trimesh(F_B_verts.float(),
                                           F_B_faces.long(),
                                           process=False,
                                           maintain_order=True)

        IO().save_mesh(
            self.F_B_surfaces,
            os.path.join(self.export_dir, f"{self.name}_F_B_surface.obj"),
        )


if __name__ == "__main__":

    dir_path = "./results/icon-mvp/BNI"
    name = "e1e7622af7074a022f5d96dc16672517"
    BNI_obj = BNI(dir_path,
                  name,
                  in_tensor=torch.load(
                      osp.join(dir_path, f"{name}_in_tensor.pt")),
                  device=torch.device("cuda:0"),
                  mvc=True)
    BNI_obj.extract_surface()
