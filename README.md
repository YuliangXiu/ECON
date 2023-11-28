<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">ECON: Explicit Clothed humans Optimized via Normal integration</h1>
  <p align="center">
    <a href="http://xiuyuliang.cn/"><strong>Yuliang Xiu</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/jyang"><strong>Jinlong Yang</strong></a>
    ·
    <a href="https://hoshino042.github.io/homepage/"><strong>Xu Cao</strong></a>
    ·
    <a href="https://ps.is.mpg.de/~dtzionas"><strong>Dimitrios Tzionas</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
  </p>
  <h2 align="center">CVPR 2023 (Highlight)</h2>
  <div align="center">
    <img src="./assets/teaser.gif" alt="Logo" width="100%">
  </div>

  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
    <a href="https://cupy.dev/"><img alt="cupy" src="https://img.shields.io/badge/-Cupy-46C02B?logo=numpy&logoColor=white"></a>
    <a href="https://twitter.com/yuliangxiu"><img alt='Twitter' src="https://img.shields.io/twitter/follow/yuliangxiu?label=%40yuliangxiu"></a>
    <a href="https://discord.gg/Vqa7KBGRyk"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/Vqa7KBGRyk?style=flat"></a>
    <br></br>
    <a href="https://arxiv.org/abs/2212.07422">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://xiuyuliang.cn/econ/'>
      <img src='https://img.shields.io/badge/ECON-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://youtu.be/5PEd_p90kS0"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/5PEd_p90kS0?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
  </p>
</p>

<br/>

ECON is designed for "Human digitization from a color image", which combines the best properties of implicit and explicit representations, to infer high-fidelity 3D clothed humans from in-the-wild images, even with **loose clothing** or in **challenging poses**. ECON also supports **multi-person reconstruction** and **SMPL-X based animation**.
<br/>

<div align="center">

|                                                                            **HuggingFace Demo**                                                                            |                                                                                             **Google Colab**                                                                                              |                                                                                                                                                                        **Blender Add-on**                                                                                                                                                                        |                                                             **Windows**                                                             |                                                                                **Docker**                                                                                 |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://huggingface.co/spaces/Yuliang/ECON"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ECON-orange'></a> | <a href='https://colab.research.google.com/drive/1YRgwoRCZIrSB2e7auEWFyG10Xzjbrbno?usp=sharing'><img src='https://img.shields.io/badge/Vanilla Colab-ec740b.svg?logo=googlecolab' alt='Google Colab'></a> | <a href='https://carlosedubarreto.gumroad.com/l/CEB_ECON'><img src='https://img.shields.io/badge/ECON-F6DDCC.svg?logo=Blender' alt='Blender'></a> <a href="https://youtu.be/sbWZbTf6ZYk"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/sbWZbTf6ZYk?logo=youtube&labelColor=ce4630&style=flat"/></a> | <a href='./docs/installation-windows.md'><img src='https://img.shields.io/badge/Windows-0078D6.svg?logo=windows' alt='Windows'></a> | <a href='https://github.com/YuliangXiu/ECON/blob/master/docs/installation-docker.md'><img src='https://img.shields.io/badge/Docker-9cf.svg?logo=Docker' alt='Docker'></a> |
|                                                                                                                                                                            |                        <a href='https://github.com/camenduru/ECON-colab'><img src='https://img.shields.io/badge/Gradio Colab-ec740b.svg?logo=googlecolab' alt='Google Colab'></a>                         |  <a href='https://github.com/kwan3854/CEB_ECON'><img src='https://img.shields.io/badge/ECON+TEXTure-F6DDCC.svg?logo=Blender' alt='Blender'></a> <a href="https://youtu.be/SDVfCeaI4AY"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/SDVfCeaI4AY?logo=youtube&labelColor=ce4630&style=flat"/></a>   |                                                                                                                                     |                                                                                                                                                                           |

</div>

## Applications

|                              ![SHHQ](assets/SHHQ.gif)                              |                                                ![crowd](assets/crowd.gif)                                                 |
| :--------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| "3D guidance" for [SHHQ Dataset](https://github.com/stylegan-human/StyleGAN-Human) |                                         multi-person reconstruction w/ occlusion                                          |
|                        ![Blender](assets/blender-demo.gif)                         |                                            ![Animation](assets/animation.gif)                                             |
|        "All-in-One" [Blender add-on](https://github.com/kwan3854/CEB_ECON)         | SMPL-X based Animation ([Instruction](https://github.com/YuliangXiu/ECON#animation-with-smpl-x-sequences-econ--hybrik-x)) |

<br/>

## News :triangular_flag_on_post:

- [2023/08/19] We released [TeCH](https://huangyangyi.github.io/TeCH/), which extends ECON with full texture support. 
- [2023/06/01] [Lee Kwan Joong](https://github.com/kwan3854) updates a Blender Addon ([Github](https://github.com/kwan3854/CEB_ECON), [Tutorial](https://youtu.be/SDVfCeaI4AY)).
- [2023/04/16] <a href="https://huggingface.co/spaces/Yuliang/ECON"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange'></a> is ready to use!
- [2023/02/27] ECON got accepted by CVPR 2023 as Highlight (top 10%)!
- [2023/01/12] [Carlos Barreto](https://twitter.com/carlosedubarret/status/1613252471035494403) creates a Blender Addon ([Download](https://carlosedubarreto.gumroad.com/l/CEB_ECON), [Tutorial](https://youtu.be/sbWZbTf6ZYk)).
- [2023/01/08] [Teddy Huang](https://github.com/Teddy12155555) creates [install-with-docker](docs/installation-docker.md) for ECON .
- [2023/01/06] [Justin John](https://github.com/justinjohn0306) and [Carlos Barreto](https://github.com/carlosedubarreto) creates [install-on-windows](docs/installation-windows.md) for ECON .
- [2022/12/22] <a href='https://colab.research.google.com/drive/1YRgwoRCZIrSB2e7auEWFyG10Xzjbrbno?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a> is now available, created by [Aron Arzoomand](https://github.com/AroArz).
- [2022/12/15] Both <a href="#demo">demo</a> and <a href="https://arxiv.org/abs/2212.07422">arXiv</a> are available.

## Key idea: d-BiNI

d-BiNI jointly optimizes front-back 2.5D surfaces such that: (1) high-frequency surface details agree with normal maps, (2) low-frequency surface variations, including discontinuities, align with SMPL-X surfaces, and (3) front-back 2.5D surface silhouettes are coherent with each other.

|        Front-view        |        Back-view        |         Side-view         |
| :----------------------: | :---------------------: | :-----------------------: |
| ![](assets/front-45.gif) | ![](assets/back-45.gif) | ![](assets/double-90.gif) |

<details><summary>Please consider cite <strong>BiNI</strong> if it also helps on your project</summary>

```bibtex
@inproceedings{cao2022bilateral,
  title={Bilateral normal integration},
  author={Cao, Xu and Santo, Hiroaki and Shi, Boxin and Okura, Fumio and Matsushita, Yasuyuki},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part I},
  pages={552--567},
  year={2022},
  organization={Springer}
}
```

</details>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#instructions">Instructions</a>
    </li>
    <li>
      <a href="#demos">Demos</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
<br/>

## Instructions

- See [installion doc for Docker](docs/installation-docker.md) to run a docker container with pre-built image for ECON demo
- See [installion doc for Windows](docs/installation-windows.md) to install all the required packages and setup the models on _Windows_
- See [installion doc for Ubuntu](docs/installation-ubuntu.md) to install all the required packages and setup the models on _Ubuntu_
- See [magic tricks](docs/tricks.md) to know a few technical tricks to further improve and accelerate ECON
- See [testing](docs/testing.md) to prepare the testing data and evaluate ECON

## Demos

- ### Quick Start

```bash
# For single-person image-based reconstruction (w/ l visualization steps, 1.8min)
python -m apps.infer -cfg ./configs/econ.yaml -in_dir ./examples -out_dir ./results

# For multi-person image-based reconstruction (see config/econ.yaml)
python -m apps.infer -cfg ./configs/econ.yaml -in_dir ./examples -out_dir ./results -multi

# To generate the demo video of reconstruction results
python -m apps.multi_render -n <file_name>

```

- ### Animation with SMPL-X sequences (ECON + [HybrIK-X](https://github.com/Jeff-sjtu/HybrIK#smpl-x))

```bash
# 1. Use HybrIK-X to estimate SMPL-X pose sequences from input video
# 2. Rig ECON's reconstruction mesh, to be compatible with SMPL-X's parametrization (-dress for dress/skirts).
# 3. Animate with SMPL-X pose sequences obtained from HybrIK-X, getting <file_name>_motion.npz
# 4. Render the frames with Blender (rgb-partial texture, normal-normal colors), and combine them to get final video

python -m apps.avatarizer -n <file_name>
python -m apps.animation -n <file_name> -m <motion_name>

# Note: to install missing python packages into Blender
# blender -b --python-expr "__import__('pip._internal')._internal.main(['install', 'moviepy'])"

wget https://download.is.tue.mpg.de/icon/econ_empty.blend
blender -b --python apps.blender_dance.py -- normal <file_name> 10 > /tmp/NULL
```

<details><summary>Please consider cite <strong>HybrIK-X</strong> if it also helps on your project</summary>

```bibtex
@article{li2023hybrik,
  title={HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-body Mesh Recovery},
  author={Li, Jiefeng and Bian, Siyuan and Xu, Chao and Chen, Zhicun and Yang, Lixin and Lu, Cewu},
  journal={arXiv preprint arXiv:2304.05690},
  year={2023}
}
```

</details>

- ### Gradio Demo

We also provide a UI for testing our method that is built with gradio. This demo also supports pose&prompt guided human image generation! Running the following command in a terminal will launch the demo:

```bash
git checkout main
python app.py
```

This demo is also hosted on HuggingFace Space <a href="https://huggingface.co/spaces/Yuliang/ECON"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ECON-orange'></a>

- ### Full Texture Generation

#### Method 1: ECON+TEXTure

Please firstly follow the [TEXTure's installation](https://github.com/YuliangXiu/TEXTure#installation-floppy_disk) to setup the env of TEXTure.

```bash

# generate required UV atlas
python -m apps.avatarizer -n <file_name> -uv

# generate new texture using TEXTure
git clone https://github.com/YuliangXiu/TEXTure
cd TEXTure
ln -s ../ECON/results/econ/cache
python -m scripts.run_texture --config_path=configs/text_guided/avatar.yaml
```

Then check `./experiments/<file_name>/mesh` for the results.

<details><summary>Please consider cite <strong>TEXTure</strong> if it also helps on your project</summary>

```bibtex
@article{richardson2023texture,
  title={Texture: Text-guided texturing of 3d shapes},
  author={Richardson, Elad and Metzer, Gal and Alaluf, Yuval and Giryes, Raja and Cohen-Or, Daniel},
  journal={ACM Transactions on Graphics (TOG)},
  publisher={ACM New York, NY, USA},
  year={2023}
}
```
</details>

#### Method 2: TeCH

Please check out our new paper, *TeCH: Text-guided Reconstruction of Lifelike Clothed Humans* ([Page](https://huangyangyi.github.io/TeCH/), [Code](https://github.com/huangyangyi/TeCH))

<details><summary>Please consider cite <strong>TeCH</strong> if it also helps on your project</summary>

```bibtex
@inproceedings{huang2024tech,
  title={{TeCH: Text-guided Reconstruction of Lifelike Clothed Humans}},
  author={Huang, Yangyi and Yi, Hongwei and Xiu, Yuliang and Liao, Tingting and Tang, Jiaxiang and Cai, Deng and Thies, Justus},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}
```

</details>

<br/>

## More Qualitative Results

|   ![OOD Poses](assets/OOD-poses.jpg)   |
| :------------------------------------: |
|          _Challenging Poses_           |
| ![OOD Clothes](assets/OOD-outfits.jpg) |
|            _Loose Clothes_             |

<br/>
<br/>

## Citation

```bibtex
@inproceedings{xiu2023econ,
  title     = {{ECON: Explicit Clothed humans Optimized via Normal integration}},
  author    = {Xiu, Yuliang and Yang, Jinlong and Cao, Xu and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```

<br/>

## Acknowledgments

We thank [Lea Hering](https://is.mpg.de/person/lhering) and [Radek Daněček](https://is.mpg.de/person/rdanecek) for proof reading, [Yao Feng](https://ps.is.mpg.de/person/yfeng), [Haven Feng](https://is.mpg.de/person/hfeng), and [Weiyang Liu](https://wyliu.com/) for their feedback and discussions, [Tsvetelina Alexiadis](https://ps.is.mpg.de/person/talexiadis) for her help with the AMT perceptual study.

Here are some great resources we benefit from:

- [ICON](https://github.com/YuliangXiu/ICON) for SMPL-X Body Fitting
- [BiNI](https://github.com/hoshino042/bilateral_normal_integration) for Bilateral Normal Integration
- [MonoPortDataset](https://github.com/Project-Splinter/MonoPortDataset) for Data Processing, [MonoPort](https://github.com/Project-Splinter/MonoPort) for fast implicit surface query
- [rembg](https://github.com/danielgatis/rembg) for Human Segmentation
- [MediaPipe](https://google.github.io/mediapipe/getting_started/python.html) for full-body landmark estimation
- [PyTorch-NICP](https://github.com/wuhaozhe/pytorch-nicp) for non-rigid registration
- [smplx](https://github.com/vchoutas/smplx), [PyMAF-X](https://www.liuyebin.com/pymaf-x/), [PIXIE](https://github.com/YadiraF/PIXIE) for Human Pose & Shape Estimation
- [CAPE](https://github.com/qianlim/CAPE) and [THuman](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset) for Dataset
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for Differential Rendering

Some images used in the qualitative examples come from [pinterest.com](https://www.pinterest.com/).

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No.860768 ([CLIPE Project](https://www.clipe-itn.eu)).

## Contributors

Kudos to all of our amazing contributors! ECON thrives through open-source. In that spirit, we welcome all kinds of contributions from the community.

<a href="https://github.com/yuliangxiu/ECON/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yuliangxiu/ECON" />
</a>

_Contributor avatars are randomly shuffled._

---

<br>

## License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

## Disclosure

MJB has received research gift funds from Adobe, Intel, Nvidia, Meta/Facebook, and Amazon. MJB has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While MJB is a part-time employee of Meshcapade, his research was performed solely at, and funded solely by, the Max Planck Society.

## Contact

For technical questions, please contact yuliang.xiu@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de
