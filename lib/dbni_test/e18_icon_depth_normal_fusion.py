import numpy as np
# from u00_utils import construct_vertices_from_depth_map_and_mask, construct_facets_from_depth_map_mask, save_rgba_img
import pyvista as pv
import os
import open3d as o3d
import matplotlib.pyplot as plt

data_dir = "data/icon"
data = np.load("data/icon/climb_trial1-000100.npy", allow_pickle=True)

normal_front = data.item().get("normal_map_F")
normal_back = data.item().get("normal_map_B")

mask = data.item().get("mask").astype(bool)[..., 0]
# plt.imshow((normal_front+1)/2)
# plt.show()

# save_rgba_img(os.path.join(data_dir, "normal_front.png"), (128*(normal_front+1)).astype(np.uint8), mask)
# save_rgba_img(os.path.join(data_dir, "normal_back.png"), (128*(normal_back+1)).astype(np.uint8), mask)


depth_front = data.item().get("depth_F")
depth_back = -data.item().get("depth_B")

mask_depth = data.item().get("depth_mask").astype(bool)

# save_rgba_img(os.path.join(data_dir, "depth_front.png"), (100+ depth_front).astype(np.uint8), mask_depth)
# save_rgba_img(os.path.join(data_dir, "depth_back.png"), (100+ depth_back).astype(np.uint8), mask_depth)


normal_T_back = data.item().get("T_normal_B")
mask_T = data.item().get("T_mask").astype(bool)

from double_side_bilateral_normal_integration_cupy import double_side_bilateral_normal_integration

surface_front, surface_back, depth_map_front_est, _ = \
    double_side_bilateral_normal_integration(normal_front=normal_front,
                                              normal_back=normal_back,
                                              normal_mask=mask,
                                              depth_front=depth_front,
                                              depth_back=depth_back,
                                              depth_mask=mask_depth,
                                              k=2,
                                              lambda_normal_back=1,
                                              lambda_depth_front = 1e-4,
                                              lambda_depth_back = 1e-4,
                                              lambda_boundary_consistency=10,
                                              )
surface_front.save(os.path.join(data_dir, f"shape_combined_front_refined".replace(".", "_")+".ply"))
surface_back.save(os.path.join(data_dir, f"shape_combined_back_refined".replace(".", "_")+".ply"))
# vertices_front = construct_vertices_from_depth_map_and_mask(mask=mask_depth, depth_map=depth_front)
# vertices_back = construct_vertices_from_depth_map_and_mask(mask=mask_depth, depth_map=depth_back)
# facets = construct_facets_from_depth_map_mask(mask_depth)

# pv.PolyData(vertices_front, facets).save(os.path.join(data_dir, "shape_coarse_front.ply"))
# pv.PolyData(vertices_back, facets).save(os.path.join(data_dir, "shape_coarse_back.ply"))
# pv.PolyData(np.concatenate((vertices_front, vertices_back), 0)).save(os.path.join(data_dir, "shape_coarse_merged.ply"))
#
# plt.imshow(normal_T_back)
# plt.show()
# from bilateral_normal_integration_numpy import bilateral_normal_integration

# for k in [2]:
#     for lambda1 in [1e-4]:
#         _, surface_F, _, _, _ = bilateral_normal_integration(normal_map=normal_front,
#                                                         normal_mask=mask,
#                                                         k=k,
#                                                         lambda1=lambda1,
#                                                         depth_map=depth_front,
#                                                         depth_mask=mask_depth)
#         _, surface_B, _, _, _ = bilateral_normal_integration(normal_map=normal_back,
#                                                                 normal_mask=mask,
#                                                                 k=k,
#                                                                 lambda1=lambda1,
#                                                                 depth_map=depth_back,
#                                                                 depth_mask=mask_depth)
#         surface_F.save(os.path.join(data_dir, f"shape_front_refined_k_{k}_lambda_{lambda1}".replace(".", "_")+".ply"))
#         # o3d.io.write_point_cloud(os.path.join(data_dir, f"pcd_front_refined_k_{k}_lambda_{lambda1}".replace(".", "_")+".ply"), pcd_F)
#
#         surface_B.save(os.path.join(data_dir, f"shape_back_refined_k_{k}_lambda_{lambda1}".replace(".", "_")+".ply"))
#         # o3d.io.write_point_cloud(os.path.join(data_dir, f"pcd_back_refined_k_{k}_lambda_{lambda1}".replace(".", "_")+".ply"), pcd_B)