## CAPE Testset

![CAPE testset](../assets/cape.png)

```bash
# 1. Register at http://icon.is.tue.mpg.de/ or https://cape.is.tue.mpg.de/
# 2. Download CAPE testset (Easy: 50, Hard: 100)
bash fetch_cape.sh 

# 3. Check CAPE testset via 3D visualization
python -m lib.dataloader_demo -v -c ./configs/train/icon-filter.yaml -d cape
```

## Command

```bash
conda activate icon

# model_type: 
#   "pifu"            reimplemented PIFu
#   "pamir"           reimplemented PaMIR
#   "icon-filter"     ICON w/ global encoder (continous local wrinkles)
#   "icon-nofilter"   ICON w/o global encoder (correct global pose)

python -m apps.train -cfg ./configs/train/icon-filter.yaml -test

# TIP: reduce "mcube_res" as 128 in apps/train.py for faster evaluation
```

The qualitative results are located at `./results/icon-filter`

<br>

## Benchmark (train on THuman2.0, test on CAPE)

|Method|PIFu|PaMIR|ICON|ICON-filter|ICON-keypoint[1]|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Easy-Chamfer|2.396|-|1.522|1.297|**1.291**|
|Easy-P2S|1.098|-|1.424|1.26|**1.088**|
|Easy-NC|0.137|-|**0.092**|0.094|0.104|
|Hard-Chamfer|4.162|-|1.538|**1.487**|1.663|
|Hard-P2S|1.675|-|1.434|**1.396**|1.493|
|Hard-NC|0.210|-|**0.089**|0.104|0.112|

[1] Mihajlovic, Marko, et al. "KeypointNeRF: Generalizing image-based volumetric avatars using relative spatial encoding of keypoints." ECCV 2022.

<br>

## Citation

:+1: Please cite these CAPE-related papers

```
@inproceedings{CAPE:CVPR:20,
  title = {{Learning to Dress 3D People in Generative Clothing}},
  author = {Ma, Qianli and Yang, Jinlong and Ranjan, Anurag and Pujades, Sergi and Pons-Moll, Gerard and Tang, Siyu and Black, Michael J.},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  month = June,
  year = {2020},
  month_numeric = {6}
}

@article{Pons-Moll:Siggraph2017,
  title = {ClothCap: Seamless 4D Clothing Capture and Retargeting},
  author = {Pons-Moll, Gerard and Pujades, Sergi and Hu, Sonny and Black, Michael},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)},
  volume = {36},
  number = {4},
  year = {2017},
  note = {Two first authors contributed equally},
  crossref = {},
  url = {http://dx.doi.org/10.1145/3072959.3073711}
}
```