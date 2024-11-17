from __future__ import print_function
import argparse
import os
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms


import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d/')
from training.volumetric_rendering.feature_match import IDLoss
sys.path.append('/data1/sch/EG3D-projector-master/eg3d/projector/PTI')
from models.vgg import net


def initialize_vgg_each(vgg):

    enc_layers = list(vgg.children())
    enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
    mse_loss = nn.MSELoss()


device = torch.device("cuda:0")
a_dir = "/data1/sch/evaluation/EG3D/Ours/illustration/"
# b_dir = "/data1/sch/DualStyleGAN-main/output_11/pixar_512/"
# a_image_filenames = [x for x in os.listdir(a_dir) if is_image_file(x)]
a_image_filenames = [x for x in os.listdir(a_dir)]
n = len(a_image_filenames)
trans = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# val_style = trans(style_im).unsqueeze(0).to(global_config.device)
# id_loss
id_loss = IDLoss().to(device).eval().requires_grad_(False)

# content and style loss
# Initialize vgg_style
network_pkl = '/data1/sch/EG3D-projector-master/eg3d/networks/vgg_normalised.pth'
print('Loading vgg_networks from "%s"...' % network_pkl)
vgg = net.vgg.eval()
vgg.load_state_dict(torch.load(network_pkl))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
decoder = net.decoder.to(device)
network = net.Net(vgg, decoder)
network.to(device)


# load style 
path_s = f'/data1/sch/evaluation/EG3D/Style/illustration_6_01051.png'
style = PIL.Image.open(path_s).convert('RGB')
style = trans(style).unsqueeze(0).to(device)

sum_id = 0
sum_c = 0
sum_s = 0
for image_name in a_image_filenames:
    img_a = a_dir + image_name
    b_image_filenames = [x for x in os.listdir(img_a)]
    L_id = 0
    L_c = 0
    L_s = 0
    for name in b_image_filenames:
        # read data
        path_gen = img_a + '/' + name
        path_c = f'/data1/sch/evaluation/EG3D/Content/{image_name}/{name}'
        gen = PIL.Image.open(path_gen).convert('RGB')
        content = PIL.Image.open(path_c).convert('RGB')
        gen = trans(gen).unsqueeze(0).to(device)
        content = trans(content).unsqueeze(0).to(device)

        # id loss
        loss_id = id_loss(gen, content)
        L_id = L_id + loss_id

        # content and style loss
        loss_c, loss_s = network(gen, content, style)
        L_c = L_c + loss_c
        L_s = L_s + loss_s
    sum_id = sum_id + (L_id / 3)
    sum_c = sum_c + (L_c / 3)
    sum_s = sum_s + (L_s / 3)

avg_id = sum_id / n
avg_c = sum_c / n
avg_s = sum_s / n
print("ID Loss:")
print(avg_id)
print("Content Loss:")
print(avg_c)
print("Style Loss:")
print(avg_s)
    
