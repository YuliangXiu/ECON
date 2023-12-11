## Getting started

Start by cloning the repo:

```bash
git clone git@github.com:YuliangXiu/ECON.git
cd ECON
```

## Environment

- Ubuntu 20 / 18, (Windows as well, see [issue#7](https://github.com/YuliangXiu/ECON/issues/7))
- **CUDA=11.6, GPU Memory > 12GB**
- Python = 3.8
- PyTorch >= 1.13.0 (official [Get Started](https://pytorch.org/get-started/locally/))
- Cupy >= 11.3.0 (offcial [Installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi))
- PyTorch3D = 0.7.2 (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))

```bash

sudo apt-get install libeigen3-dev ffmpeg

# install required packages
cd ECON
conda env create -f environment.yaml
conda activate econ
pip install -r requirements.txt

# the installation(incl. compilation) of PyTorch3D will take ~20min
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2

# install libmesh & libvoxelize
cd lib/common/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize
python setup.py build_ext --inplace
```

## Register at [ICON's website](https://icon.is.tue.mpg.de/)

![Register](../assets/register.png)
Required:

- [SMPL](http://smpl.is.tue.mpg.de/): SMPL Model (Male, Female)
- [SMPL-X](http://smpl-x.is.tue.mpg.de/): SMPL-X Model, used for training
- [SMPLIFY](http://smplify.is.tue.mpg.de/): SMPL Model (Neutral)
- [PIXIE](https://icon.is.tue.mpg.de/user.php): PIXIE SMPL-X estimator

:warning: Click **Register now** on all dependencies, then you can download them all with **ONE** account.

## Downloading required models and extra data

```bash
cd ECON
bash fetch_data.sh # requires username and password
```

## Citation

:+1: Please consider citing these awesome HPS approaches: PyMAF-X, PIXIE


```
@article{pymafx2022,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  journal={arXiv preprint arXiv:2207.06400},
  year={2022}
}


@inproceedings{PIXIE:2021,
  title={Collaborative Regression of Expressive Bodies using Moderation},
  author={Yao Feng and Vasileios Choutas and Timo Bolkart and Dimitrios Tzionas and Michael J. Black},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}


```
