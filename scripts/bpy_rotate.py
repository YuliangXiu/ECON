#
# Joachim Tesch, Max Planck Institute for Intelligent Systems, Perceiving Systems
#
# Render object at different object rotations
#
# Version: 20190307
#
# Notes:
#
#   + Eevee
#     + Fast physically based rendering on GPU
#     + Needs OpenGL hardware context for rendering and therefore only available when running script through Blender app (GUI or background)
#
#   + Cycles
#     + Fast low-quality preview render: quality_preview=True
#     + Slow high-quality final render: quality_preview=False
#
#   + Code can also run from Blender application Text Editor via 'Run Script'
#

# pylint: disable=invalid-name

import argparse
import os
import sys
import time
from glob import glob

# project related libs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
from lib.common.blender_utils import *

# input_path = '/home/yxiu/Downloads/econ_pinterest/select/econ_meshes'
# output_path = '/home/yxiu/Downloads/econ_pinterest/select/blender'


##############################################################################
# Main
##############################################################################
if __name__== '__main__':
    try:
        if bpy.app.background:
            parser = argparse.ArgumentParser(description='Generate object rotation images from obj.',
                                                formatter_class=argparse.RawTextHelpFormatter,
                                                epilog = 'Usage: %s --input body.obj --output body.png\n       %s --input ./data --output ./data-render' % (__file__, __file__))
            parser.add_argument('--input', dest='input_path', default="", type=str, help='Input file or directory')
            parser.add_argument('--subject', dest='subject', default="", type=str, help='Input file or directory')
            parser.add_argument('--output', dest='output_path', default="", type=str, help='Output file or directory')
            parser.add_argument('--views', type=int, default=views, help='Number of views to render')
            parser.add_argument("--correct", action="store_true", help="vis sampler 3D")

            parser.add_argument('--quality_preview', type=int, default=quality_preview, help='Use preview quality')
            parser.add_argument('--resolution_x', type=int, default=resolution_x, help='Render image X resolution')
            parser.add_argument('--resolution_y', type=int, default=resolution_y, help='Render image Y resolution')
            parser.add_argument('--shadows', type=int, default=shadows, help='Show shadows')

            parser.add_argument('--smooth', type=int, default=smooth, help='Smooth mesh')
            parser.add_argument('--wireframe', type=int, default=wireframe, help='Show wireframe overlay')
            parser.add_argument('--line_thickness', type=float, default=line_thickness, help='Wireframe line thickness')
            parser.add_argument('--quads', type=int, default=quads, help='Convert triangles to quads')
            parser.add_argument('--object_transparent', type=int, default=object_transparent, help='Render face transparent')
            parser.add_argument('--mouth_transparent', type=int, default=mouth_transparent, help='Render mouth transparent')

            parser.add_argument('--compositor_background_image', type=int, default=compositor_background_image, help='Use background image')
            parser.add_argument('--compositor_image_scale', type=float, default=compositor_image_scale, help='Input image scale')
            parser.add_argument('--compositor_alpha', type=float, default=compositor_alpha, help='Rendered object alpha value')

            parser.add_argument('--render', type=str, default=render, help='Render engine (cycles|eevee)')
            parser.add_argument('--cycles_gpu', type=int, default=cycles_gpu, help='Use CUDA GPU acceleration for Cycles')
            
            parser.add_argument('--background', action='store_true', help='Use CUDA GPU acceleration for Cycles')
            parser.add_argument('--python', type=str, default=cycles_gpu, help='Use CUDA GPU acceleration for Cycles')

            args = parser.parse_args()
            # TODO make input/output positional and not optional

            # if (args.input_path is None) or (args.output_path is None):
            #     parser.print_help()
            #     print('-----\n')
            #     sys.exit(1)

            # input_path = os.path.join(args.input_path, args.subject)
            # output_path = args.output_path

            # if not os.path.exists(input_path):
            #     print('ERROR: Invalid input path')
            #     sys.exit(1)

            views = args.views
            quality_preview = args.quality_preview
            resolution_x = args.resolution_x
            resolution_y = args.resolution_y
            shadows = args.shadows

            smooth = args.smooth
            wireframe = args.wireframe
            line_thickness = args.line_thickness

            # Always use quads in wireframe mode
            if wireframe:
                quads = True
            else:
                quads = args.quads

            object_transparent = args.object_transparent
            mouth_transparent = args.mouth_transparent

            compositor_background_image = args.compositor_background_image
            compositor_image_scale = args.compositor_image_scale
            compositor_alpha = args.compositor_alpha

            render = args.render
            cycles_gpu = args.cycles_gpu
        # end if bpy.app.background

        print('Render engine: ' + render)
        if ((render != 'cycles') and (render != 'eevee')):
            print('ERROR: Unsupported render engine')
            sys.exit(1)

        if (render == 'eevee'):
            if not using_app():
                print('ERROR: Eevee rendering only supported when running through Blender app')
                sys.exit(1)
            else:
                quality_preview = False

        startTime = time.perf_counter()

        setup_scene()

        # Process data
        cwd = os.getcwd()

        # if not os.path.isfile(input_path):
        #     if not input_path.endswith(os.path.sep):
        #         input_path += os.path.sep

        # if not input_path.startswith(os.path.sep):
        #     input_dir = os.path.join(cwd, input_path)
        # else:
        #     input_dir = os.path.dirname(input_path)

        # if not output_path.endswith('.png'):
        #     if not output_path.endswith(os.path.sep):
        #         output_path += os.path.sep

        # if not output_path.startswith(os.path.sep):
        #     if not output_path.endswith('.png'):
        #         output_dir = os.path.join(cwd, output_path)
        #     else:
        #         output_dir = os.path.join(cwd, os.path.dirname(output_path))
        # else:
        #     output_dir = os.path.dirname(output_path)

        # if not input_dir.endswith(os.path.sep):
        #     input_dir += os.path.sep

        # if not output_dir.endswith(os.path.sep):
        #     output_dir += os.path.sep
        
        if args.subject != "":
            
            input_path = glob(f"{args.input_path}/*{args.subject}*.obj")[0]
                
            input_dir = os.path.dirname(input_path)+os.path.sep
            
            output_dir = input_dir.replace("meshes","blender")+os.path.sep
            output_path = os.path.join(output_dir, args.subject)
        
        print('Input path: ' + input_path)
        print('Input directory: ' + input_dir)
        print('Output path: ' + output_path)
        print('Output directory: ' + output_dir)

        print('--------------------------------------------------')

        if os.path.isfile(input_path):
            # Process single file
            input_file = os.path.basename(input_path)
            output_file = os.path.basename(output_path)
            process_file(input_file, input_dir, output_file, output_dir, args.correct)
        else:
            # Process directory
            for input_file in sorted(os.listdir(input_dir)):
                if input_file.endswith('.obj'):
                    output_file = os.path.basename(output_path)
                    process_file(input_file, input_dir, '', output_dir, args.correct)

        print('--------------------------------------------------')
        print('Rendering finished. Processing time: %0.2f s' % (time.perf_counter() - startTime) )
        print('--------------------------------------------------')
        sys.exit(0)

    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else:
            exit_status = ex.code

        print('Exiting. Exit status: ' + str(exit_status))

        # Only exit to OS when we are not running in Blender GUI
        if bpy.app.background:
            sys.exit(exit_status)
