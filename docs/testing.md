# Evaluation

## Testing Data

![dataset](../assets/dataset.png)

- OOD pose (CAPE, [download](https://github.com/YuliangXiu/ICON/blob/master/docs/evaluation.md#cape-testset)): [`pose.txt`](../pose.txt)
- OOD outfits (RenderPeople, [link](https://renderpeople.com/)): [`loose.txt`](../loose.txt)

## Run the evaluation

```bash
# Benchmark of ECON_{IF}, which uses IF-Net+ for completion
export CUDA_VISIBLE_DEVICES=0; python -m apps.benchmark -ifnet

# Benchmark of ECON_{EX}, which uses registered SMPL for completion
export CUDA_VISIBLE_DEVICES=1; python -m apps.benchmark

```

## Benchmark

|   Method    |  $\text{ECON}_\text{IF}$  | $\text{ECON}_\text{EX}$ |
| :---------: | :-----------------------: | :---------------------: |
|             |     OOD poses (CAPE)      |                         |
| Chamfer(cm) |           0.996           |        **0.926**        |
|   P2S(cm)   |           0.967           |        **0.917**        |
| Normal(L2)  |          0.0413           |       **0.0367**        |
|             | OOD oufits (RenderPeople) |                         |
| Chamfer(cm) |           1.401           |        **1.342**        |
|   P2S(cm)   |         **1.422**         |          1.458          |
| Normal(L2)  |          0.0516           |       **0.0478**        |

**\*OOD: Out-of-Distribution**

## Citation

:+1: Please cite these CAPE-related papers

```

@inproceedings{xiu2022icon,
  title     = {{ICON}: {I}mplicit {C}lothed humans {O}btained from {N}ormals},
  author    = {Xiu, Yuliang and Yang, Jinlong and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {13296-13306}
}

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
