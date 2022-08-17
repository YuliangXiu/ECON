## Prerequirement

Make sure you have already generated all the required synthetic data (refer to [Dataset Instruction](dataset.md)) under `./data/thuman2_{num_views}views`, which includes the rendered RGB (`render/`), normal images(`normal_B/`, `normal_F/`, `T_normal_B/`, `T_normal_F/`), corresponding calibration matrix (`calib/`) and pre-computed visibility arrays (`vis/`).

:eyes: Test your dataloader with [vedo](https://vedo.embl.es/)

```bash

# visualization for SMPL-X mesh
python -m lib.dataloader_demo -v -c ./configs/train/icon-filter.yaml

# visualization for voxelized SMPL
python -m lib.dataloader_demo -v -c ./configs/train/pamir.yaml
```

<p align="center">
    <img src="../assets/vedo.gif" width=50%>
</p>

:warning: Don't support headless mode currently, `unset PYOPENGL_PLATFORM` before training.
## Command

```bash
conda activate icon

# model_type: 
#   "pifu"            reimplemented PIFu
#   "pamir"           reimplemented PaMIR
#   "icon-filter"     ICON w/ global encoder (continous local wrinkles)
#   "icon-nofilter"   ICON w/o global encoder (correct global pose)
#   "icon-mvp"        minimal viable product, simple yet efficient

# Training for implicit MLP
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/icon-filter.yaml

# Training for normal network
CUDA_VISIBLE_DEVICES=0 python -m apps.train-normal -cfg ./configs/train/normal.yaml
```

## Tensorboard

```bash
cd ICON/results/{name}
tensorboard --logdir .
```

## Checkpoint

All the checkpoints are saved at `./data/ckpt/{name}`