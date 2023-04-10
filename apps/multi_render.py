import argparse

import torch

from lib.common.render import Render

root = "./results/econ/vid"

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

in_tensor = torch.load(f"{root}/{args.name}_in_tensor.pt")

render = Render(size=512, device=torch.device(f"cuda:{args.gpu}"))

# visualize the final results in self-rotation mode
verts_lst = in_tensor["body_verts"] + in_tensor["BNI_verts"]
faces_lst = in_tensor["body_faces"] + in_tensor["BNI_faces"]

# self-rotated video
render.load_meshes(verts_lst, faces_lst)
render.get_rendered_video_multi(in_tensor, f"{root}/{args.name}_cloth.mp4")
