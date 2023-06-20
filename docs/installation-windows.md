# Windows installation tutorial

Another [issue#16](https://github.com/YuliangXiu/ECON/issues/16) shows the whole process to deploy ECON on _Windows_

## Dependencies and Installation

- Use [Anaconda](https://www.anaconda.com/products/distribution)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Wget for Windows](https://eternallybored.org/misc/wget/1.21.3/64/wget.exe)
- Create a new folder on your C drive and rename it "wget" and move the downloaded "wget.exe" over there.
- Add the path to your wget folder to your system environment variables at `Environment Variables > System Variables Path > Edit environment variable`

![image](https://user-images.githubusercontent.com/34035011/210986038-39dbb7a1-12ef-4be9-9af4-5f658c6beb65.png)

- Install [Git for Windows 64-bit](https://git-scm.com/download/win)
- [Visual Studio Community 2022](https://visualstudio.microsoft.com/) (Make sure to check all the boxes as shown in the image below)

![image](https://user-images.githubusercontent.com/34035011/210983023-4e5a0024-68f0-4adb-8089-6ff598aec220.PNG)

## Getting started

Start by cloning the repo:

```bash
git clone https://github.com/yuliangxiu/ECON.git
cd ECON
```

## Environment

- Windows 10 / 11
- **CUDA=11.3**
- Python = 3.8
- PyTorch >= 1.12.1 (official [Get Started](https://pytorch.org/get-started/locally/))
- Cupy >= 11.3.0 (offcial [Installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi))
- PyTorch3D = 0.7.1 (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))

```bash
# install required packages
cd ECON
conda env create -f environment-windows.yaml
conda activate econ

# install pytorch and cupy
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install cupy-cuda11x
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1

# install libmesh & libvoxelize
cd lib/common/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize
python setup.py build_ext --inplace
```

[Issue#69: Discussion of additional argument `--compiler=msvc` in `python setup.py build_ext --inplace`](https://github.com/YuliangXiu/ECON/issues/69)

<br>

## Register at [ICON's website](https://icon.is.tue.mpg.de/)

![Register](../assets/register.png)
Required:

- [SMPL](http://smpl.is.tue.mpg.de/): SMPL Model (Male, Female)
- [SMPL-X](http://smpl-x.is.tue.mpg.de/): SMPL-X Model, used for training
- [SMPLIFY](http://smplify.is.tue.mpg.de/): SMPL Model (Neutral)
- [PIXIE](https://icon.is.tue.mpg.de/user.php): PIXIE SMPL-X estimator

:warning: Click **Register now** on all dependencies, then you can download them all with **ONE** account.

## Downloading required models and extra data (make sure to install git and wget for windows for this to work)

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
