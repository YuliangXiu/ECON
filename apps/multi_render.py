import torch

from lib.common.render import Render


def generate_video(name):

    root = "./results/econ/vid"

    in_tensor = torch.load(f"{root}/{name}_in_tensor.pt")

    render = Render(size=512, device=torch.device(f"cuda:0"))

    # visualize the final results in self-rotation mode
    verts_lst = in_tensor["body_verts"] + in_tensor["BNI_verts"]
    faces_lst = in_tensor["body_faces"] + in_tensor["BNI_faces"]

    # self-rotated video
    render.load_meshes(verts_lst, faces_lst)
    render.get_rendered_video_multi(in_tensor, f"{root}/{name}_cloth.mp4")
