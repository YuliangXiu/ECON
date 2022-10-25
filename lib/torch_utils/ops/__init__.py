# # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.

# # empty
# try:
#     from .fused_act import FusedLeakyReLU, fused_leaky_relu
#     # from .upfirdn2d import upfirdn2d

#     print('Using custom CUDA kernels')
# except Exception as e:
#     print(str(e))
#     print('There was something wrong with the CUDA kernels')
#     print('Reverting to native PyTorch implementation')
#     from .native_ops import FusedLeakyReLU, fused_leaky_relu
