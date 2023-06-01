## Getting started

Start by cloning the repo:

```bash
git clone git@github.com:YuliangXiu/ECON.git
cd ECON
```
## Environment
- **GPU Memory > 12GB**

start with [docker compose](https://docs.docker.com/compose/)
```bash
# you can change your container name by passing --name "parameter" 
docker compose run [--name myecon] econ
```

## Docker container's shell
```bash
# activate the pre-build env
cd code
conda activate econ

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
cd ~/code
bash fetch_data.sh # requires username and password
```
## :whale2: **todo**
- **Image Environment Infos**
    - Ubuntu 18
    - CUDA = 11.3
    - Python = 3.8
- [X] pre-built image with docker compose
- [ ] docker run command, Dockerfile
- [ ] verify on WSL (Windows)

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
