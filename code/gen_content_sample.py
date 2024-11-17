# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import matplotlib.pyplot as plt
import scipy.misc
from torchvision.transforms import transforms


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from training.volumetric_rendering.feature_match import FeatureMatch, StyleEncoder


#----------------------------------------------------------------------------

# def parse_range(s: Union[str, List]) -> List[int]:
#     '''Parse a comma separated list of numbers or ranges and return a list of ints.

#     Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
#     '''
#     if isinstance(s, list): return s
#     ranges = []
#     range_re = re.compile(r'^(\d+)-(\d+)$')
#     for p in s.split(','):
#         if m := range_re.match(p):
#             ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
#         else:
#             ranges.append(int(p))
#     return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--outdir', help='Where to save the out images', type=str, required=True, metavar='DIR')
# modify
@click.option('--outdir', help='Where to save the out images', type=str, required=False, metavar='DIR', default='/data1/sch/EG3D-projector-master/out_style', show_default=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', type=str, required=False, metavar='DIR', default='/data1/sch/EG3D-projector-master/eg3d/networks/ffhq_stdcrop-128.pkl', show_default=True)
@click.option('--style_encoder', 'style_pkl', help='Network pickle filename', type=str, required=False, metavar='DIR', default='/data1/sch/EG3D-projector-master/eg3d/check/process_50/feature_match_final_16.pth', show_default=True)
@click.option('--style_img', 'Is_path', help='Network pickle filename', type=str, required=False, metavar='DIR', default='/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Caricature_FFHQ/caricature_generate_1997_01.png', show_default=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False, default='0-3', show_default=True)

@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
def generate_images(
    network_pkl: str,
    style_pkl: str,
    Is_path: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    sampling_multiplier: float,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=out --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    sampling_multiplier = 2
    device = torch.device('cuda:0')
    if 'pkl' in network_pkl:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    # else:
    #     init_args = ()
    #     init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
    #                    'channel_max': 512, 'fused_modconv_default': 'inference_only',
    #                    'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48,
    #                                         'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
    #                                         'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
    #                                         'disparity_space_sampling': False, 'clamp_mode': 'softplus',
    #                                         'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
    #                                         'c_gen_conditioning_zero': False, 'c_scale': 1.0,
    #                                         'superresolution_noise_mode': 'none', 'density_reg': 0.25,
    #                                         'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
    #                                         'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
    #                    'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
    #                                  'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25,
    #                    'img_resolution': 512, 'img_channels': 3}
    #     rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
    #                         'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
    #                         'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
    #                         'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
    #                         'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
    #                         'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
    #                         'sr_antialias': True}

    #     # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    #     print("Reloading Modules!")
    #     G = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)

    #     ckpt = torch.load(network_pkl)
    #     G.load_state_dict(ckpt['G_ema'], strict=False)
    #     G.neural_rendering_resolution = 128

    #     G.rendering_kwargs = rendering_kwargs

    #     G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    #     G.rendering_kwargs['depth_resolution_importance'] = int(
    #         G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    
    # Style_En = torch.load(style_pkl).eval().requires_grad_(False).to(device)

    # style_im = PIL.Image.open(Is_path).convert('RGB')
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # style_image = trans(style_im).unsqueeze(0).to(device)

    # w_s = Style_En(style_image)
    # w_s = w_s.repeat([1, G.backbone.mapping.num_ws, 1])

    # add_weight = torch.ones(1, 14, 1).to(device)
    # add_weight[:, 7:, :] = 0

    # 测试
    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/label/img00000024.npy'
    # val_c = np.load(c_path)
    # val_c = np.reshape(val_c, (1, 25))
    # val_c = torch.FloatTensor(val_c).to(device)

    # os.makedirs(outdir, exist_ok=True)
    # imgs = []

    # img = G.synthesis(w_s, val_c, style_out=w_s, noise_mode='const')['image']

    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # imgs.append(img)

    # im = torch.cat(imgs, dim=2)

    # PIL.Image.fromarray(im[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed200.png')
    # print("Finish!!")

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    imgs = []
    angle_p = -0.2
    z_1 = torch.from_numpy(np.random.RandomState(11).randn(1, G.z_dim)).to(device)
    # z_2 = torch.from_numpy(np.random.RandomState(12).randn(1, G.z_dim)).to(device)
    j = 0

    for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        # w_1 = G.mapping(z_1, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # w_2 = G.mapping(z_2, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        w_potential_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/emb/img00000510_w_plus.npy'
        w_c = torch.from_numpy(np.load(w_potential_path)).to(device)

        # w_cs = w_c * add_weight + w_s * (1 - add_weight)

        # img = G.synthesis(w_cs, camera_params, style_out=w_s, noise_mode='const')['image']
        img = G.synthesis(w_c, camera_params, noise_mode='const')['image']

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/510_{j}.png')
        j = j + 1

        imgs.append(img)

    img = torch.cat(imgs, dim=2)

    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/510.png')

    # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    #     imgs = []
    #     depth_imgs = []
    #     angle_p = -0.2
    #     for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
    #         cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    #         cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    #         cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    #         conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    #         camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    #         conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    #         # ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    #         w_potential_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/embeddings/img00000004/img00000004_w_plus.npy'
    #         ws = torch.from_numpy(np.load(w_potential_path)).to(device)
    #         img = G.synthesis(ws, camera_params)['image']

    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    #         imgs.append(img)

    #     img = torch.cat(imgs, dim=2)

    #     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        # if shapes:
        #     # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        #     max_batch=1000000

        #     samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        #     samples = samples.to(z.device)
        #     sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        #     transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        #     transformed_ray_directions_expanded[..., -1] = -1

        #     head = 0
        #     with tqdm(total = samples.shape[1]) as pbar:
        #         with torch.no_grad():
        #             while head < samples.shape[1]:
        #                 torch.manual_seed(0)
        #                 sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
        #                 sigmas[:, head:head+max_batch] = sigma
        #                 head += max_batch
        #                 pbar.update(max_batch)

        #     sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        #     sigmas = np.flip(sigmas, 0)

        #     # Trim the border of the extracted cube
        #     pad = int(30 * shape_res / 256)
        #     pad_value = -1000
        #     sigmas[:pad] = pad_value
        #     sigmas[-pad:] = pad_value
        #     sigmas[:, :pad] = pad_value
        #     sigmas[:, -pad:] = pad_value
        #     sigmas[:, :, :pad] = pad_value
        #     sigmas[:, :, -pad:] = pad_value

        #     if shape_format == '.ply':
        #         from shape_utils import convert_sdf_samples_to_ply
        #         convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
        #     elif shape_format == '.mrc': # out mrc
        #         with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        #             mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
