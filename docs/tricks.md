## Technical tricks to improve or accelerate ECON

### If the reconstructed geometry is not satisfying, play with the adjustable parameters in _config/econ.yaml_

- `use_smpl: ["hand"]`
  - [ ]: don't use either hands or face parts from SMPL-X
  - ["hand"]: only use the **visible** hands from SMPL-X
  - ["hand", "face"]: use both **visible** hands and face from SMPL-X
- `thickness: 2cm`
  - could be increased accordingly in case final reconstruction **xx_full.obj** looks flat
- `k: 4`
  - could be reduced accordingly in case the surface of **xx_full.obj** has discontinous artifacts
- `hps_type: PIXIE`
  - "pixie": more accurate for face and hands
  - "pymafx": more robust for challenging poses
- `texture_src: image`
  - "image": direct mapping the aligned pixels to final mesh
  - "SD": use Stable Diffusion to generate full texture (TODO)

### To accelerate the inference, you could

- `use_ifnet: False`
  - True: use IF-Nets+ for mesh completion ( $\text{ECON}_\text{IF}$ - Better quality, **~2min / img**)
  - False: use SMPL-X for mesh completion ( $\text{ECON}_\text{EX}$ - Faster speed, **~1.8min / img**)

```bash
# For single-person image-based reconstruction (w/o all visualization steps, 1.5min)
python -m apps.infer -cfg ./configs/econ.yaml -in_dir ./examples -out_dir ./results -novis
```
