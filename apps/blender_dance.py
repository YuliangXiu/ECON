import bpy
import os
import sys
from tqdm import tqdm
import numpy as np

argv = sys.argv
argv = argv[argv.index("--") + 1:]    # get all args after "--"
render_normal = True if argv[0] == 'normal' else False
avatar_name = argv[1]
duration = int(argv[2])

# use-defined parameters
n_start = 0
fps = 25
n_end = fps * duration
n_step = 1


class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0


class Character:
    def __init__(self, v_seq, faces, colors, name):

        self.v_seq = v_seq
        self.faces = faces
        self.colors = np.hstack((colors / 255.0, np.ones((colors.shape[0], 1), dtype=np.float32)))
        self.material = bpy.data.materials['vertex-color']
        self.name = name

    def load_frame(self, index, delta_trans):

        name = f"{self.name}_{str(index).zfill(4)}"

        # create mesh
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            self.v_seq[index][:, [0, 2, 1]] * np.array([1., -1., 1.]), [],
            self.faces.view(ndarray_pydata)
        )
        mesh.vertex_colors.new(name="vcol")
        mesh.vertex_colors["vcol"].active = True
        mloops = np.zeros((len(mesh.loops)), dtype=np.int)
        mesh.loops.foreach_get("vertex_index", mloops)
        mesh.vertex_colors["vcol"].data.foreach_set("color", self.colors[mloops].flatten())
        mesh.validate()

        # create object
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        obj.active_material = self.material
        obj.delta_location = delta_trans
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

        return obj, mesh

    def __len__(self):
        return self.v_seq.shape[0]


root_dir = os.path.join(os.path.dirname(__file__), "..")
avatar_pos = {
    avatar_name: np.array([-0.7, -7.5, 0.]),
}
avatar_names = avatar_pos.keys()

# load blend file
blend_path = f"{root_dir}/econ_empty.blend"
bpy.ops.wm.open_mainfile(filepath=blend_path)

# rendering settings
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.eevee.taa_render_samples = 128

# load all the large motion data
avatar_data = {}
pbar = tqdm(avatar_names)
for key in pbar:
    pbar.set_description(f"Loading {key}")
    motion_path = f"{root_dir}/results/econ/seq/{key}_motion.npz"
    motion = np.load(motion_path, allow_pickle=True)
    if render_normal:
        avatar_data[key] = Character(motion['v_seq'], motion['f'], motion['normal'], name=key)
    else:
        avatar_data[key] = Character(motion['v_seq'], motion['f'], motion['rgb'], name=key)

export_dir = f"{root_dir}/results/econ/render/{argv[0]}"
os.makedirs(export_dir, exist_ok=True)

# start rendering
for fid in tqdm(range(n_start, n_end, n_step)):
    objs = []
    meshes = []
    for key in avatar_names:
        obj, mesh = avatar_data[key].load_frame(fid, avatar_pos[key])
        objs.append(obj)
        meshes.append(mesh)

    bpy.context.scene.frame_set(fid)
    bpy.context.scene.render.filepath = f"{export_dir}/{str(fid).zfill(5)}.png"
    bpy.ops.render.render(use_viewport=True, write_still=True)
    bpy.ops.object.select_all(action='SELECT')

    for mesh in meshes:
        bpy.data.meshes.remove(mesh)

    # Debug: you can open the blend file to check the blender scene
    # if fid == 10:
    #     bpy.ops.wm.save_as_mainfile(filepath=f"{root_dir}/test.blend")

# combine all the rendering images into a video
from moviepy.editor import ImageSequenceClip
from glob import glob
from os import cpu_count

mpy_conf = {
    "codec": "libx264",
    "remove_temp": True,
    "preset": "ultrafast",
    "ffmpeg_params": ['-crf', '0', '-pix_fmt', 'yuv444p', '-profile:v', 'high444'],
    "logger": "bar",
    "fps": fps,
    "threads": cpu_count(),
}

video_lst = sorted(glob(f"{root_dir}/results/econ/render/{argv[0]}/*.png"))
video_clip = ImageSequenceClip(video_lst, fps=fps)
video_clip = video_clip.set_duration(duration)
video_clip.write_videofile(f"{root_dir}/results/econ/render/{argv[0]}.mp4", **mpy_conf)
