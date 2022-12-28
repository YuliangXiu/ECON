import bpy
import sys, os
from math import radians
import mathutils
import bmesh

print(sys.exec_prefix)
from tqdm import tqdm
import numpy as np

##################################################
# Globals
##################################################

views = 120

render = 'eevee'
cycles_gpu = False

quality_preview = False
samples_preview = 16
samples_final = 256

resolution_x = 512
resolution_y = 512

shadows = False

# diffuse_color = (57.0/255.0, 108.0/255.0, 189.0/255.0, 1.0)
# diffuse_color = (18/255., 139/255., 142/255.,1)     #correct
# diffuse_color = (251/255., 60/255., 60/255.,1)    #wrong

smooth = False

wireframe = False
line_thickness = 0.1
quads = False

object_transparent = False
mouth_transparent = False

compositor_background_image = False
compositor_image_scale = 1.0
compositor_alpha = 0.7

##################################################
# Helper functions
##################################################


def blender_print(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def using_app():
    ''' Returns if script is running through Blender application (GUI or background processing)'''
    return (not sys.argv[0].endswith('.py'))


def setup_diffuse_transparent_material(target, color, object_transparent, backface_transparent):
    ''' Sets up diffuse/transparent material with backface culling in cycles'''

    mat = target.active_material
    if mat is None:
        # Create material
        mat = bpy.data.materials.new(name='Material')
        target.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    node_geometry = nodes.new('ShaderNodeNewGeometry')

    node_diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    node_diffuse.inputs[0].default_value = color

    node_transparent = nodes.new('ShaderNodeBsdfTransparent')
    node_transparent.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

    node_emission = nodes.new('ShaderNodeEmission')
    node_emission.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)

    node_mix = nodes.new(type='ShaderNodeMixShader')
    if object_transparent:
        node_mix.inputs[0].default_value = 1.0
    else:
        node_mix.inputs[0].default_value = 0.0

    node_mix_mouth = nodes.new(type='ShaderNodeMixShader')
    if object_transparent or backface_transparent:
        node_mix_mouth.inputs[0].default_value = 1.0
    else:
        node_mix_mouth.inputs[0].default_value = 0.0

    node_mix_backface = nodes.new(type='ShaderNodeMixShader')

    node_output = nodes.new(type='ShaderNodeOutputMaterial')

    links = mat.node_tree.links

    links.new(node_geometry.outputs[6], node_mix_backface.inputs[0])

    links.new(node_diffuse.outputs[0], node_mix.inputs[1])
    links.new(node_transparent.outputs[0], node_mix.inputs[2])
    links.new(node_mix.outputs[0], node_mix_backface.inputs[1])

    links.new(node_emission.outputs[0], node_mix_mouth.inputs[1])
    links.new(node_transparent.outputs[0], node_mix_mouth.inputs[2])
    links.new(node_mix_mouth.outputs[0], node_mix_backface.inputs[2])

    links.new(node_mix_backface.outputs[0], node_output.inputs[0])
    return


##################################################


def setup_scene():
    global render
    global cycles_gpu
    global quality_preview
    global resolution_x
    global resolution_y
    global shadows
    global wireframe
    global line_thickness
    global compositor_background_image

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    scene = bpy.data.scenes['Scene']

    # Setup render engine
    if render == 'cycles':
        scene.render.engine = 'CYCLES'
    else:
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    if quality_preview:
        scene.cycles.samples = samples_preview
    else:
        scene.cycles.samples = samples_final

    # Setup Cycles CUDA GPU acceleration if requested
    if render == 'cycles':
        if cycles_gpu:
            print('Activating GPU acceleration')
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

            if bpy.app.version[0] >= 3:
                cuda_devices = bpy.context.preferences.addons[
                    'cycles'].preferences.get_devices_for_type(compute_device_type='CUDA')
            else:
                (cuda_devices, opencl_devices
                ) = bpy.context.preferences.addons['cycles'].preferences.get_devices()

            if (len(cuda_devices) < 1):
                print('ERROR: CUDA GPU acceleration not available')
                sys.exit(1)

            for cuda_device in cuda_devices:
                if cuda_device.type == 'CUDA':
                    cuda_device.use = True
                    print('Using CUDA device: ' + str(cuda_device.name))
                else:
                    cuda_device.use = False
                    print('Igoring CUDA device: ' + str(cuda_device.name))

            scene.cycles.device = 'GPU'
            if bpy.app.version[0] < 3:
                scene.render.tile_x = 256
                scene.render.tile_y = 256
        else:
            scene.cycles.device = 'CPU'
            if bpy.app.version[0] < 3:
                scene.render.tile_x = 64
                scene.render.tile_y = 64

    # Disable Blender 3 denoiser to properly measure Cycles render speed
    if bpy.app.version[0] >= 3:
        scene.cycles.use_denoising = False

    # Setup camera
    camera = bpy.data.objects['Camera']
    camera.location = (0.0, -3, 1.8)
    camera.rotation_euler = (radians(74), 0.0, 0)
    bpy.data.cameras['Camera'].lens = 55

    # Setup light

    # Setup lights
    light = bpy.data.objects['Light']
    light.location = (-2, -3.0, 0.0)
    light.rotation_euler = (radians(90.0), 0.0, 0.0)
    bpy.data.lights['Light'].type = 'POINT'
    bpy.data.lights['Light'].energy = 2
    light.data.cycles.cast_shadow = False

    if 'Sun' not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN')
        light_sun = bpy.context.active_object
        light_sun.location = (0.0, -3, 0.0)
        light_sun.rotation_euler = (radians(45.0), 0.0, radians(30))
        bpy.data.lights['Sun'].energy = 2
        light_sun.data.cycles.cast_shadow = shadows
    else:
        light_sun = bpy.data.objects['Sun']

    if shadows:
        # Setup shadow catcher
        bpy.ops.mesh.primitive_plane_add()
        plane = bpy.context.active_object
        plane.scale = (5.0, 5.0, 1)

        plane.cycles.is_shadow_catcher = True

        # Exclude plane from diffuse cycles contribution to avoid bright pixel noise in body rendering
        # plane.cycles_visibility.diffuse = False

        if wireframe:
            # Unmark freestyle edges
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.mark_freestyle_edge(clear=True)
            bpy.ops.object.mode_set(mode='OBJECT')

    # Setup freestyle mode for wireframe overlay rendering
    if wireframe:
        scene.render.use_freestyle = True
        scene.render.line_thickness = line_thickness
        bpy.context.view_layer.freestyle_settings.linesets[0].select_edge_mark = True

        # Disable border edges so that we don't see contour of shadow catcher plane
        bpy.context.view_layer.freestyle_settings.linesets[0].select_border = False
    else:
        scene.render.use_freestyle = False

    if compositor_background_image:
        # Setup compositing when using background image
        setup_compositing()
    else:
        # Output transparent image when no background is used
        scene.render.image_settings.color_mode = 'RGBA'


##################################################


def setup_compositing():

    global compositor_image_scale
    global compositor_alpha

    # Node editor compositing setup
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Create input image node
    image_node = tree.nodes.new(type='CompositorNodeImage')

    scale_node = tree.nodes.new(type='CompositorNodeScale')
    scale_node.inputs[1].default_value = compositor_image_scale
    scale_node.inputs[2].default_value = compositor_image_scale

    blend_node = tree.nodes.new(type='CompositorNodeAlphaOver')
    blend_node.inputs[0].default_value = compositor_alpha

    # Link nodes
    links = tree.links
    links.new(image_node.outputs[0], scale_node.inputs[0])

    links.new(scale_node.outputs[0], blend_node.inputs[1])
    links.new(tree.nodes['Render Layers'].outputs[0], blend_node.inputs[2])

    links.new(blend_node.outputs[0], tree.nodes['Composite'].inputs[0])


def render_file(input_file, input_dir, output_file, output_dir, yaw, correct):
    '''Render image of given model file'''
    global smooth
    global object_transparent
    global mouth_transparent
    global compositor_background_image
    global quads

    path = input_dir + input_file

    # Import object into scene
    bpy.ops.import_scene.obj(filepath=path)
    object = bpy.context.selected_objects[0]

    object.rotation_euler = (radians(90.0), 0.0, radians(yaw))
    z_bottom = np.min(np.array([vert.co for vert in object.data.vertices])[:, 1])
    # z_top = np.max(np.array([vert.co for vert in object.data.vertices])[:,1])
    # blender_print(radians(90.0), z_bottom, z_top)
    object.location -= mathutils.Vector((0.0, 0.0, z_bottom))

    if quads:
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode='OBJECT')

    if smooth:
        bpy.ops.object.shade_smooth()

    # Mark freestyle edges
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    if correct:
        diffuse_color = (18 / 255., 139 / 255., 142 / 255., 1)    #correct
    else:
        diffuse_color = (251 / 255., 60 / 255., 60 / 255., 1)    #wrong

    setup_diffuse_transparent_material(object, diffuse_color, object_transparent, mouth_transparent)

    if compositor_background_image:
        # Set background image
        image_path = input_dir + input_file.replace('.obj', '_original.png')
        bpy.context.scene.node_tree.nodes['Image'].image = bpy.data.images.load(image_path)

    # Render
    bpy.context.scene.render.filepath = os.path.join(output_dir, output_file)

    # Silence console output of bpy.ops.render.render by redirecting stdout to file
    # Note: Does not actually write the output to file (Windows 7)
    sys.stdout.flush()
    old = os.dup(1)
    os.close(1)
    os.open('blender_render.log', os.O_WRONLY | os.O_CREAT)

    # Render
    bpy.ops.render.render(write_still=True)

    # Remove temporary output redirection
    #    sys.stdout.flush()
    #    os.close(1)
    #    os.dup(old)
    #    os.close(old)

    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()


def process_file(input_file, input_dir, output_file, output_dir, correct=True):
    global views
    global quality_preview

    if not input_file.endswith('.obj'):
        print('ERROR: Invalid input: ' + input_file)
        return

    print('Processing: ' + input_file)
    if output_file == '':
        output_file = input_file[:-4]

    if quality_preview:
        output_file = output_file.replace('.png', '-preview.png')

    angle = 360.0 / views
    pbar = tqdm(range(0, views))
    for view in pbar:
        pbar.set_description(f"{os.path.basename(output_file)} | View:{str(view)}")
        yaw = view * angle
        output_file_view = f"{output_file}/{view:03d}.png"
        if not os.path.exists(os.path.join(output_dir, output_file_view)):
            render_file(input_file, input_dir, output_file_view, output_dir, yaw, correct)

    cmd = "ffmpeg -loglevel quiet -r 30 -f lavfi -i color=c=white:s=512x512 -i " + os.path.join(output_dir, output_file, '%3d.png') + \
        " -shortest -filter_complex \"[0:v][1:v]overlay=shortest=1,format=yuv420p[out]\" -map \"[out]\" -y " + output_dir+"/"+output_file+".mp4"
    os.system(cmd)
