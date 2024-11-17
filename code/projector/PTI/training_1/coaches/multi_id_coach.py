import os
from pathlib import Path

import torch
from tqdm import tqdm
import PIL.Image
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

from configs import paths_config, hyperparameters, global_config
from training_1.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import numpy as np

import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d/')
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


def cam_sampler(batch, device, cam_pivot, cam_radius, intrinsics):
    c_samples_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device, horizontal_stddev=0.3, vertical_stddev=0.155, batch_size=batch)
    intrinsics = intrinsics.reshape(1, -1).repeat(batch, 1)
    c = torch.cat([c_samples_pose.reshape(batch, -1), intrinsics], -1)
    return c


class MultiIDCoach(BaseCoach):

    def __init__(self, content_data_loader, style_data_loader, emb_dataloader, use_wandb):
        super().__init__(content_data_loader, style_data_loader, emb_dataloader, use_wandb)

    def train(self):
        # self.G.synthesis.train()
        # self.G.mapping.train()

        # w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        # use_ball_holder = True
        # w_pivots = []
        # images = []
        # cs = []
        # style_images = []

        log_dir = Path(paths_config.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        
        image_log_step = 5000

        # modify: change dataloder to iter
        # content_iter = iter(self.content_data_loader)
        style_iter = iter(self.style_data_loader)
        emb_iter = iter(self.emb_dataloader)

        self.restart_training()

        # seeds = 0

        # cam_pivot = [0, 0, 0.2]
        # cam_radius = 2.7
        # intrinsics = FOV_to_intrinsics(18.837, device=global_config.device)

        for i in tqdm(range(hyperparameters.max_iter)):
            # load content and style image
            # content_fname, content_image = next(content_iter)

            # style_image===============================================
            style_fname, style_image, style_c = next(style_iter)
            style_c = style_c.squeeze(dim=1).reshape(1, -1)    # [1, 25]
            style_name = style_fname[0]
            #===========================================================

            # content_image=============================================
            w_fname, w_pivot, c = next(emb_iter)
            # content_name = content_fname[0]
            w_pivot = w_pivot.squeeze(dim=1)         # [1, 14, 512]
            c = c.squeeze(dim=1).reshape(1, -1)      # [1, 25]

            # modify
            # z = torch.from_numpy(np.random.RandomState(seeds).randn(1, 512)).to(global_config.device)
            # c = cam_sampler(1, global_config.device, cam_pivot, cam_radius, intrinsics)
            #============================================================
            

            # content_image
            # w_path_dir = f'{paths_config.embedding_base_dir}/{content_name}'
            # c_path = os.path.join(paths_config.input_c_path,f'{content_name}.npy')
            # # print("content_image_name: ", content_fname, 'content_c_path', c_path)
            # c = np.load(c_path)
            # c = np.reshape(c, (1, 25))
            # c = torch.FloatTensor(c).cuda()
            # w_pivot = self.get_inversion(w_path_dir, content_name, content_image)
            # content_image_batch = content_image.to(global_config.device)

            # style_image
            style_image_batch = style_image.to(global_config.device)   # [1, 3, 512, 512]

            # generate stylized image and content_image
            generated_images_cs, generated_images_cs_raw, generated_images_c, sigma = self.forward(w_pivot, c, style_image_batch)

            if i % image_log_step == 0:
                with torch.no_grad():
                    imgs = []
                    # modify
                    c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/label/img00000024.npy'
                    val_c = np.load(c_path)
                    val_c = np.reshape(val_c, (1, 25))
                    val_c = torch.FloatTensor(val_c).to(global_config.device)
                    w_potential_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/emb/img00000024_w_plus.npy'
                    val_w = torch.from_numpy(np.load(w_potential_path)).to(global_config.device)
                    Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_2_FFHQ/illustration_6_01051.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_FFHQ/Tomer_Hanuka-0823.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Fantasy_FFHQ/Peter_Mohrbacher-0114.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Pixar_FFHQ/pixar_generate_43_01.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Caricature_FFHQ/caricature_generate_1997_01.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Comic_FFHQ/comic_generate_14.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Arance_FFHQ/arance_generate_39.png'
                    # Is_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Impasto_FFHQ/Gregory Manchess-0346.png'
                    style_im = PIL.Image.open(Is_path).convert('RGB')
                    trans = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                    val_style = trans(style_im).unsqueeze(0).to(global_config.device)
                    val_images_cs, val_images_cs_raw, val_images_c, sigma_val = self.forward(val_w, val_c, val_style)

                    vis_img_cs = (val_images_cs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imgs.append(vis_img_cs)
                    vis_img_c = (val_images_c.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imgs.append(vis_img_c)
                    vis_img_s = (val_style.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imgs.append(vis_img_s)

                    img = torch.cat(imgs, dim=2)

                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'/data1/sch/EG3D-projector-master/eg3d/out/process_74/iter_{i}.png')

            # loss1, loss_c1, loss_s1 = self.calc_content_style_loss(generated_images_cs, generated_images_c, style_image_batch)

            # self.optimizer_style.zero_grad()
            # self.optimizer_G.zero_grad()
            # loss1.backward()
            # self.optimizer_G.step()
            # self.optimizer_style.step()

            # train the discriminator
            # d_loss = self.compute_d_loss(style_image_batch, generated_images_cs.detach())
            # 消融========================================
            d_loss = self.compute_d_loss(style_image_batch, style_c, generated_images_cs.detach(), generated_images_cs_raw.detach(), c)
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()
            # 消融========================================

            # train the discriminator
            g_loss, loss_adv, loss_c, loss_s, TVloss, loss_id = self.compute_g_loss(generated_images_cs, generated_images_cs_raw, c, generated_images_c, style_image_batch, sigma)
            # 反传梯度给Style Encoder
            self.optimizer_style.zero_grad()
            self.optimizer_G.zero_grad()
            
            # # loss_style_en = loss1
            g_loss.backward()
            self.optimizer_G.step()
            self.optimizer_style.step()

            # 反传梯度给G
            # self.optimizer_G.zero_grad()

            # loss2, loss_c2, loss_s2 = self.calc_content_style_loss(generated_images_cs, generated_images_c, style_image_batch)

            # # loss_style_G = loss2
            # loss2.backward()
            # self.optimizer_G.step()

            # 记录
            # writer.add_scalar('loss_total', loss1.item(), i + 1)
            # writer.add_scalar('loss_content', loss_c1.item(), i + 1)
            # writer.add_scalar('loss_style', loss_s1.item(), i + 1)

            writer.add_scalar('loss_total', g_loss.item(), i + 1)
            writer.add_scalar('loss_content', loss_c.item(), i + 1)
            writer.add_scalar('loss_style', loss_s.item(), i + 1)
            # 消融========================================
            writer.add_scalar('loss_adv', loss_adv.item(), i + 1)
            # 消融========================================
            writer.add_scalar('TVloss', TVloss.item(), i + 1)
            writer.add_scalar('loss_id', loss_id.item(), i + 1)
            # writer.add_scalar('loss_id', loss_id.item(), i + 1)

            # if i % image_log_step == 0:
            #     checkpoint_path = f'/data1/sch/EG3D-projector-master/eg3d/check/process_22/feature_match_{i}.pth'
            #     torch.save(self.feature_match, checkpoint_path)

            #     save_dict = {
            #                     'G_ema': self.G.state_dict()
            #             }
            #     checkpoint_path = f'/data1/sch/EG3D-projector-master/eg3d/check/process_22/model_G_style_{i}.pth'
            #     print('final model ckpt save to ', checkpoint_path)
            #     torch.save(save_dict, checkpoint_path)

        
        checkpoint_path = f'/data1/sch/EG3D-projector-master/eg3d/check/process_74/feature_match_final_16.pth'
        torch.save(self.feature_match, checkpoint_path)

        save_dict = {
                        'G_ema': self.G.state_dict()
                }
        checkpoint_path = f'/data1/sch/EG3D-projector-master/eg3d/check/process_74/model_G_style.pth'
        print('final model ckpt save to ', checkpoint_path)
        torch.save(save_dict, checkpoint_path)

        # for fname, image in self.content_data_loader:
        #     if self.image_counter >= hyperparameters.max_images_to_invert:
        #         break

        #     image_name = fname[0]
        #     w_path_dir = f'{paths_config.embedding_base_dir}/{image_name}'
        #     c_path = os.path.join(paths_config.input_c_path,f'{image_name}.npy')
        #     print("image_name: ", fname, 'c_path', c_path)
        #     c = np.load(c_path)

        #     c = np.reshape(c, (1, 25))

        #     c = torch.FloatTensor(c).cuda()

        #     # if hyperparameters.first_inv_type == 'w+':
        #     #     embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
        #     # else:
        #     #     embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        #     # embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        #     # os.makedirs(embedding_dir, exist_ok=True)

        #     cs.append(c)
        #     w_pivot = self.get_inversion(w_path_dir, image_name, image)
        #     w_pivots.append(w_pivot)
        #     images.append((image_name, image))
        #     self.image_counter += 1
        
        # self.restart_training()

        # for i in tqdm(range(hyperparameters.max_pti_steps)):
        #     self.image_counter = 0

        #     for c, data, w_pivot in zip(cs, images, w_pivots):
        #         image_name, image = data

        #         if self.image_counter >= hyperparameters.max_images_to_invert:
        #             break

        #         real_images_batch = image.to(global_config.device)

        #         generated_images = self.forward(w_pivot, c)
        #         loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
        #                                 self.G, use_ball_holder, w_pivot)

        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        #         use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

        #         global_config.training_step += 1
        #         self.image_counter += 1

        # # if self.use_wandb:
        # #     log_images_from_w(w_pivots, self.G, [image[0] for image in images])

        # torch.save(self.G,
        #            f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')
        # save_dict = {
        #                     'G_ema': self.G.state_dict()
        #             }
        # checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pth'
        # print('final model ckpt save to ', checkpoint_path)
        # torch.save(save_dict, checkpoint_path)
