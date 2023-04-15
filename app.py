# install

import glob
import gradio as gr
import os
import numpy as np

os.environ["ICON"] = "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"

import subprocess

# if os.getenv('SYSTEM') == 'spaces':
#     subprocess.run('pip install pyembree'.split())
#     subprocess.run(
#         'pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html'.split())
#     subprocess.run(
#         'pip install https://download.is.tue.mpg.de/icon/HF/pytorch3d-0.7.0-cp38-cp38-linux_x86_64.whl'.split())

from apps.infer import generate_model

# running

description = '''
# Fully-textured Clothed Human Digitization (ECON + TEXTure) 
### ECON: Explicit Clothed humans Optimized via Normal integration (CVPR 2022, Highlight)

<table>
<th>
<ul>
<li><strong>Homepage</strong> <a href="https://econ.is.tue.mpg.de/">econ.is.tue.mpg.de</a></li>
<li><strong>Code</strong> <a href="https://github.com/YuliangXiu/ECON">YuliangXiu/ECON</a></li>
<li><strong>Paper</strong> <a href="https://arxiv.org/abs/2212.07422">arXiv</a>, <a href="https://readpaper.com/paper/4736821012688027649">ReadPaper</a></li>
<li><strong>Chatroom</strong> <a href="https://discord.gg/Vqa7KBGRyk">Discord</a></li>
</ul>
<br>
<ul>
<li><strong>Colab Notebook</strong> <a href="https://colab.research.google.com/drive/1YRgwoRCZIrSB2e7auEWFyG10Xzjbrbno?usp=sharing">Google Colab</a></li>
<li><strong>Blender Plugin</strong> <a href="https://carlosedubarreto.gumroad.com/l/CEB_ECON">Blender</a></li>
<li><strong>Docker Image</strong> <a href="https://github.com/YuliangXiu/ECON/blob/master/docs/installation-docker.md">Docker</a></li>
<li><strong>Windows Setup</strong> <a href="https://github.com/YuliangXiu/ECON/blob/master/docs/installation-windows.md">Windows</a></li>
</ul>

<a href="https://twitter.com/yuliangxiu"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/yuliangxiu?style=social"></a>
<iframe src="https://ghbtns.com/github-btn.html?user=yuliangxiu&repo=ECON&type=star&count=true&v=2&size=small" frameborder="0" scrolling="0" width="100" height="20"></iframe>
<a href="https://youtu.be/j5hw4tsWpoY"><img alt="YouTube Video Views" src="https://img.shields.io/youtube/views/j5hw4tsWpoY?style=social"></a>
</th>
<th>
<iframe width="560" height="315" src="https://www.youtube.com/embed/j5hw4tsWpoY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</th>
</table>

<h4> The reconstruction takes ~2min for single image. <span style="color:red"> If ERROR, try "Submit Image" again.</span></h4>

<details>

<summary>More</summary>

#### Citation
```
@inproceedings{xiu2023econ,
  title     = {{ECON: Explicit Clothed humans Optimized via Normal integration}},
  author    = {Xiu, Yuliang and Yang, Jinlong and Cao, Xu and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
} 
```

#### Acknowledgments:
- [controlnet-openpose](https://huggingface.co/spaces/diffusers/controlnet-openpose)
- [TEXTure](https://huggingface.co/spaces/TEXTurePaper/TEXTure)


#### Image Credits

* [Pinterest](https://www.pinterest.com/search/pins/?q=parkour&rs=sitelinks_searchbox)

#### Related works

* [ICON @ MPI](https://icon.is.tue.mpg.de/)
* [MonoPort @ USC](https://xiuyuliang.cn/monoport)
* [Phorhum @ Google](https://phorhum.github.io/)
* [PIFuHD @ Meta](https://shunsukesaito.github.io/PIFuHD/)
* [PaMIR @ Tsinghua](http://www.liuyebin.com/pamir/pamir.html)

</details>
'''

from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import gradio as gr
import torch
import base64
from io import BytesIO
from PIL import Image

# live conditioning
canvas_html = "<pose-canvas id='canvas-root' style='display:flex;max-width: 500px;margin: 0 auto;'></pose-canvas>"
load_js = """
async () => {
  const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/pose-gradio.js"
  fetch(url)
    .then(res => res.text())
    .then(text => {
      const script = document.createElement('script');
      script.type = "module"
      script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
      document.head.appendChild(script);
    });
}
"""
get_js_image = """
async (image_in_img, prompt, image_file_live_opt, live_conditioning) => {
  const canvasEl = document.getElementById("canvas-root");
  const data = canvasEl? canvasEl._data : null;
  return [image_in_img, prompt, image_file_live_opt, data]
}
"""

# Constants
low_threshold = 100
high_threshold = 200

# Models
pose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# This command loads the individual model components on GPU on-demand. So, we don't
# need to explicitly call pipe.to("cuda").
pipe.enable_model_cpu_offload()

# xformers
pipe.enable_xformers_memory_efficient_attention()

# Generator seed,
generator = torch.manual_seed(0)


def get_pose(image):
    return pose_model(image)


def generate_images(image, prompt, image_file_live_opt='file', live_conditioning=None):
    if image is None and 'image' not in live_conditioning:
        raise gr.Error("Please provide an image")
    try:
        if image_file_live_opt == 'file':
            pose = get_pose(image)
        elif image_file_live_opt == 'webcam':
            base64_img = live_conditioning['image']
            image_data = base64.b64decode(base64_img.split(',')[1])
            pose = Image.open(BytesIO(image_data)).convert('RGB').resize((512, 512))
        output = pipe(
            prompt,
            pose,
            generator=generator,
            num_images_per_prompt=3,
            num_inference_steps=20,
        )
        all_outputs = []
        all_outputs.append(pose)
        for image in output.images:
            all_outputs.append(image)
        return all_outputs, all_outputs
    except Exception as e:
        raise gr.Error(str(e))


def toggle(choice):
    if choice == "file":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    elif choice == "webcam":
        return gr.update(visible=False, value=None), gr.update(visible=True, value=canvas_html)


examples_pose = glob.glob('examples/pose/*')
examples_cloth = glob.glob('examples/cloth/*')

default_step = 50

with gr.Blocks() as demo:
    gr.Markdown(description)

    out_lst = []
    with gr.Row():
        with gr.Column():
            with gr.Row():
                
                live_conditioning = gr.JSON(value={}, visible=False)

                with gr.Column():
                    image_file_live_opt = gr.Radio(["file", "webcam"],
                                                   value="file",
                                                   label="How would you like to upload your image?")

                    with gr.Row():
                        image_in_img = gr.Image(source="upload", visible=True, type="pil")
                        canvas = gr.HTML(None, elem_id="canvas_html", visible=False)

                    image_file_live_opt.change(
                        fn=toggle,
                        inputs=[image_file_live_opt],
                        outputs=[image_in_img, canvas],
                        queue=False
                    )
                    prompt = gr.Textbox(
                        label="Enter your prompt",
                        max_lines=1,
                        placeholder="best quality, extremely detailed",
                    )
                with gr.Column():
                    gallery = gr.Gallery().style(grid=[2], height="auto")
                    gallery_cache = gr.State()
                    inp = gr.Image(type="filepath", label="Input Image")
                    fitting_step = gr.inputs.Slider(
                        10, 100, step=10, label='Fitting steps', default=default_step
                    )

            with gr.Row():
                btn_sample = gr.Button("Generate Image")
                btn_submit = gr.Button("Submit Image")

            btn_sample.click(
                fn=generate_images,
                inputs=[image_in_img, prompt, image_file_live_opt, live_conditioning],
                outputs=[gallery, gallery_cache],
                _js=get_js_image
            )
            
            def get_select_index(cache, evt: gr.SelectData):
                return cache[evt.index]

            
            gallery.select(
                fn=get_select_index,
                inputs=[gallery_cache],
                outputs=[inp],
            )

            with gr.Row():

                gr.Examples(
                    examples=list(examples_pose),
                    inputs=[inp],
                    cache_examples=False,
                    fn=generate_model,
                    outputs=out_lst
                )
                gr.Examples(
                    examples=list(examples_cloth),
                    inputs=[inp],
                    cache_examples=False,
                    fn=generate_model,
                    outputs=out_lst
                )

            out_vid = gr.Video(label="Image + SMPL Body + Clothed Human")
            out_vid_download = gr.File(label="Download Video, welcome share on Twitter with #ECON")

        with gr.Column():
            overlap_inp = gr.Image(type="filepath", label="Image Normal Overlap")
            out_final = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Clothed human")
            out_final_download = gr.File(label="Download clothed human mesh")
            out_smpl = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="SMPL body")
            out_smpl_download = gr.File(label="Download SMPL body mesh")
            out_smpl_npy_download = gr.File(label="Download SMPL params")

    out_lst = [
        out_smpl, out_smpl_download, out_smpl_npy_download, out_final, out_final_download,
        overlap_inp
    ]

    btn_submit.click(fn=generate_model, inputs=[inp, fitting_step], outputs=out_lst)

    demo.load(None, None, None, _js=load_js)

if __name__ == "__main__":

    # demo.launch(debug=False, enable_queue=False,
    #             auth=(os.environ['USER'], os.environ['PASSWORD']),
    #             auth_message="Register at icon.is.tue.mpg.de to get HuggingFace username and password.")

    demo.launch(debug=True, enable_queue=True)