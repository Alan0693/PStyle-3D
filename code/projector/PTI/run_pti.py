import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# import os
from configs import global_config, paths_config
import wandb
import torch
import numpy as np

from training_1.coaches.multi_id_coach import MultiIDCoach
from training_1.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset, LabelsDataset
from utils.sampler import InfiniteSamplerWrapper



def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    # original
    # dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    
    # modify: add style_dataset
    
    content_dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),                                           # 转为Tensor，并归一化至[0-1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))        # 标准化为均值为0，标准差为1
    
    style_dataset = ImagesDataset(paths_config.style_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    
    emb_dataset = LabelsDataset(paths_config.w_plus_data_path)

    # original
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # modify: add style_dataset

    # content_dataloader = DataLoader(content_dataset, batch_size=1, shuffle=False)
    # style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=False)

    content_dataloader = DataLoader(content_dataset, sampler=InfiniteSamplerWrapper(content_dataset), batch_size=1, shuffle=False)
    style_dataloader = DataLoader(style_dataset, sampler=InfiniteSamplerWrapper(style_dataset), batch_size=1, shuffle=False)
    emb_dataloader = DataLoader(emb_dataset, sampler=InfiniteSamplerWrapper(emb_dataset), batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(content_dataloader, style_dataloader, emb_dataloader, use_wandb)
    else:
        coach = SingleIDCoach(content_dataloader, style_dataloader, use_wandb)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=True)
    # batch = 5
    # emb_dataset = LabelsDataset(paths_config.w_plus_data_path)
    # emb_dataloader = DataLoader(emb_dataset, sampler=InfiniteSamplerWrapper(emb_dataset), batch_size=batch, shuffle=False)
    # emb_iter = iter(emb_dataloader)

    # for i in range(1,10):
    #     w_fname, w, c = next(emb_iter)
    #     # w = torch.from_numpy(w).to(global_config.device)
    #     ws = w.squeeze(dim=1)
    #     c = c.squeeze(dim=1).reshape(batch, -1)
    #     # c = torch.FloatTensor(c).to(global_config.device)
    #     a = 1

    # w = torch.from_numpy(np.load(w_potential_path)).to(global_config.device)
