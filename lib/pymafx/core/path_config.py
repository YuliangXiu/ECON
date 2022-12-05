import os.path as osp

pymafx_data_dir = osp.join(osp.dirname(__file__), "../../../data/HPS/pymafx_data")

JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(pymafx_data_dir, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = osp.join(pymafx_data_dir, 'J_regressor_h36m.npy')
SMPL_MEAN_PARAMS = osp.join(pymafx_data_dir, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = osp.join(pymafx_data_dir, 'smpl')
CHECKPOINT_FILE = osp.join(pymafx_data_dir, 'PyMAF-X_model_checkpoint.pt')
PARTIAL_MESH_DIR = osp.join(pymafx_data_dir, "partial_mesh")

MANO_DOWNSAMPLING = osp.join(pymafx_data_dir, 'mano_downsampling.npz')
SMPL_DOWNSAMPLING = osp.join(pymafx_data_dir, 'smpl_downsampling.npz')
