import json
import numpy as np
from torch import float32
import os
import torch
import PIL.Image
import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

# c1_path = '/data1/sch/EG3D-projector-master/eg3d/projector_test_data/00000.npy'
# c1 = np.load(c1_path)
# c1 = np.reshape(c1,(1,25))
# print(c1)

# c2_path = '/data1/sch/EG3D-projector-master/eg3d/projector_test_data/FFHQ/label/00000.npy'
# c2 = np.load(c2_path)
# c2 = np.reshape(c2,(1,25))
# print(c2)

# w2_path = '/data1/sch/EG3D-projector-master/eg3d/projector_out/00000_w/00000_w.npy'
# w2 = np.load(c2_path)
# print(w2.shape)
# # if self.initial_w is not None:
# #     if os.path.isdir(self.initial_w):
# #         initial_w = torch.load(os.path.join(self.initial_w, image_name, 'rec_ws.pt')).cpu().numpy()
# #     else:
# #         initial_w = torch.load(self.initial_w).cpu().numpy()
# initial_w = torch.load('/data1/sch/EG3D-projector-master/eg3d/initial_w/rec_ws.pt').cpu().numpy()
# print(initial_w.shape)

device = torch.device('cuda')
network_pkl = '/data1/sch/EG3D-projector-master/eg3d/networks/ffhq512-128.pkl'
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

fov_deg = 18.837
intrinsics = FOV_to_intrinsics(fov_deg, device=device)
cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
print(cam_pivot)
cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
print(conditioning_cam2world_pose)