import abc
import os
import numpy as np
from argparse import Namespace
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
import torch.nn as nn
from torchvision import transforms
from lpips import LPIPS
from training_1.projectors import w_projector
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
# from models.e4e.psp import pSp
from models.vgg import net
from utils.log_utils import log_image_from_w
from utils.models_utils import toogle_grad, load_D, load_FF_G

import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d/')
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from training.triplane import TriPlaneGenerator
from training.dual_discriminator import DualDiscriminator
from training.volumetric_rendering.feature_match import FeatureMatch, StyleEncoder, FeatureMatch_p, IDLoss, pSp, StyleTransformer
from training.volumetric_rendering.disriminator import StarDiscriminator, StarDiscriminator_pose, adv_loss, r1_reg
from training.volumetric_rendering.function import calc_mean_std
from training.volumetric_rendering.psp_options import TrainOptions

# import sys
# sys.path.append('/data1/sch/EG3D-projector-master/eg3d/projector/PTI')
# from models.vgg import net


class BaseCoach:
    def __init__(self, content_data_loader, style_data_loader, emb_dataloader, use_wandb):

        self.use_wandb = use_wandb
        self.content_data_loader = content_data_loader
        self.style_data_loader = style_data_loader
        self.emb_dataloader = emb_dataloader
        self.w_pivots = {}
        self.image_counter = 0


        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()
        self.id_loss = IDLoss().to(global_config.device).eval().requires_grad_(False)

        # Initialize vgg_style
        network_pkl = '/data1/sch/EG3D-projector-master/eg3d/networks/vgg_normalised.pth'
        print('Loading vgg_networks from "%s"...' % network_pkl)
        self.vgg = net.vgg.eval()
        self.vgg.load_state_dict(torch.load(network_pkl))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(global_config.device)

        # Initialize vgg的各个层
        self.initialize_vgg_each()

        # self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.add_weight = torch.ones(1, 14, 1).to(global_config.device)

    # modify to train stlyle
    def restart_training(self):
        reload_modules = True

        # Initialize networks
        # G = load_old_G()

        if reload_modules:
            print("Reloading Modules!")
            init_args = ()
            init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                        'channel_max': 512, 'fused_modconv_default': 'inference_only',
                        'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48,
                                                'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                                                'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                                                'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                                'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                                'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                                                'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                                                'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                                'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                        'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
                                        'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25,
                        'img_resolution': 512, 'img_channels': 3}
            rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
                                'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
                                'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
                                'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                'sr_antialias': True}
            G = TriPlaneGenerator(*init_args, **init_kwargs).requires_grad_(False).to(global_config.device)

            ckpt = torch.load(paths_config.eg3d_ffhq_pth)
            G.load_state_dict(ckpt['G_ema'], strict=False)
            G.neural_rendering_resolution = 128

            G.rendering_kwargs = rendering_kwargs

            G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
            G.rendering_kwargs['depth_resolution_importance'] = int(
                G.rendering_kwargs['depth_resolution_importance'] * 2)

            # G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).requires_grad_(False).to(global_config.device)
            # misc.copy_params_and_buffers(G, G_new, require_all=True)
            # G_new.neural_rendering_resolution = G.neural_rendering_resolution
            # G_new.rendering_kwargs = G.rendering_kwargs
            # G_new.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
            # G_new.rendering_kwargs['depth_resolution_importance'] = int(
            #     G.rendering_kwargs['depth_resolution_importance'] * 2)
            # G = G_new
        
        self.G = G
        toogle_grad(self.G, False)
        print("G")
        # only train backbone.synthesis
        # for name, parms in self.G.named_parameters():	
        #     if name[:18] == 'backbone.synthesis':
        #         parms.requires_grad = True

        # only train backbone.synthesis and superresolution
        for name, parms in self.G.named_parameters():	
            if (name[:18] == 'backbone.synthesis') or (name[:15] == 'superresolution'):
                parms.requires_grad = True

        for name, parms in self.G.named_parameters():	
            print('-->name:', name)
            # print('-->name:', name[:18])
            print('-->grad_requirs:',parms.requires_grad)
        
        self.original_G = load_FF_G()
        toogle_grad(self.original_G, False)
        print("original_G")
        # for name, parms in self.original_G.named_parameters():	
        #     print('-->name:', name)
        #     print('-->grad_requirs:',parms.requires_grad)
        

        print("Initialize D")
        # print("StarDiscriminator()")
        # self.D = StarDiscriminator().to(global_config.device)

        # print("StarDiscriminator_pose()")
        # self.D = StarDiscriminator_pose().to(global_config.device)

        # 消融========================================
        # print("No adv loss")
        print("pretrain_D")
        self.D = load_D()
        toogle_grad(self.D, True)
        # 消融========================================

        # print("No pretrain_D")
        # self.D = DualDiscriminator(c_dim=25, img_resolution=512, img_channels=3).to(global_config.device)
        # for name, parms in self.D.named_parameters():	
        #     print('-->name:', name)
        #     print('-->grad_requirs:',parms.requires_grad)

        # print("original_G")
        # for name, parms in self.original_G.named_parameters():	
        #     print('-->name:', name)
        #     print('-->grad_requirs:',parms.requires_grad)
        
        # print("G")
        # for name, parms in self.G.named_parameters():	
        #     print('-->name:', name)
        #     print('-->grad_requirs:',parms.requires_grad)
        
        # Initialize feature_match
        print("Initialize feature_match")
        # self.feature_match = FeatureMatch(sampling_multiplier = 2).to(global_config.device)
        # self.feature_match = StyleEncoder(style_dim=512, n_mlp=4, load_pretrained_vgg=True).to(global_config.device)
        # self.feature_match = FeatureMatch_p().to(global_config.device)
        opts = TrainOptions().parse()
        opts.device = global_config.device
        self.feature_match = pSp(opts).to(global_config.device)
        # self.feature_match = StyleTransformer(opts).to(global_config.device)

        # self.optimizer_style = torch.optim.Adam(self.feature_match.parameters(), lr=0.002, betas=(0, 0.99))
        self.optimizer_style = torch.optim.Adam(self.feature_match.parameters(), lr=0.00001, betas=(0, 0.999))
        # 消融========================================
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=0.00001, betas=(0, 0.99), eps=1e-8)
        # 消融========================================
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate, betas=(0.9, 0.999))

    # def restart_training(self):

    #     # Initialize networks
    #     self.G = load_old_G()
    #     toogle_grad(self.G, True)

    #     self.original_G = load_old_G()

    #     self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
    #     self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def initialize_vgg_each(self):

        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()
    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{image_name}_w_plus.npy'
        else:
            w_potential_path = f'{w_path_dir}/{image_name}_w.npy'

        # print('load pre-computed w from ', w_potential_path)
        if not os.path.isfile(w_potential_path):
            print(w_potential_path, 'is not exist!')
            return None

        w = torch.from_numpy(np.load(w_potential_path)).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name,c):

        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            w = w_projector.project(self.G, c,id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                    num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                    use_wandb=self.use_wandb)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer
    

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_content_style_loss(self, generated_images_cs, generated_images_c, style_image):
        loss = 0.0
        content_feats = self.vgg(generated_images_c)
        style_feats = self.encode_with_intermediate(style_image)
        g_t_feats = self.encode_with_intermediate(generated_images_cs)

        # style_loss
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        loss = hyperparameters.content_weight * loss_c + hyperparameters.style_weight * loss_s
        
        return loss, loss_c, loss_s

    def compute_d_loss(self, style_image, style_label, generated_images_cs, generated_images_cs_raw, generated_images_cs_label):
        loss = 0.0
        # with real images
        style_image.requires_grad_()
        # StarGAN_D==========================================
        # loss_Dreal = 0
        # loss_reg = 0
        # out_real = self.D(style_image)
        # loss_Dreal = adv_loss(out_real, 1)
        # loss_reg = r1_reg(out_real, style_image)

        # with fake images
        # loss_Dgen = 0
        # out_fake = self.D(generated_images_cs)
        # loss_Dgen = adv_loss(out_fake, 0)
        # loss = (loss_Dgen + loss_Dreal) * 0.5 + 1 * loss_reg
        #====================================================

        # StarGAN_D_pose==========================================
        # with real images
        # loss_Dreal = 0
        # loss_reg = 0
        # out_real = self.D(style_image, style_label)
        # loss_Dreal = adv_loss(out_real, 1)
        # loss_reg = r1_reg(out_real, style_image)

        # with fake images
        # loss_Dgen = 0
        # out_fake = self.D(generated_images_cs, generated_images_cs_label)
        # loss_Dgen = adv_loss(out_fake, 0)
        # loss = (loss_Dgen + loss_Dreal) * 0.5 + 1 * loss_reg
        #====================================================

        # EG3D_D==============================================================================================
        # Dmain: Maximize logits for real images.
        loss_Dreal = 0
        loss_reg = 0
        style_image_raw = nn.functional.interpolate(style_image, size=128, mode="bilinear")
        real_style_img = {'image': style_image, 'image_raw': style_image_raw}
        real_logits = self.D(real_style_img, style_label)
        loss_Dreal = nn.functional.softplus(-real_logits)    # 希望判别器将real_style_img判别为越大的值（真）
        loss_Dreal = loss_Dreal.squeeze()

        # Dr1: Apply R1 regularization
        loss_reg = r1_reg(real_logits, style_image)
        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_style_img['image'], real_style_img['image_raw']], create_graph=True, only_inputs=True)
        r1_grads_image = r1_grads[0]
        r1_grads_image_raw = r1_grads[1]
        loss_reg = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        gen_style_img = {'image': generated_images_cs, 'image_raw': generated_images_cs_raw}
        gen_logits = self.D(gen_style_img, generated_images_cs_label)
        loss_Dgen = torch.nn.functional.softplus(gen_logits)  # 希望判别器将gen_style_img判别为越小的值（假）
        loss_Dgen = loss_Dgen.squeeze()

        loss = (loss_Dgen + loss_Dreal) * 0.5 + 5 * loss_reg
        #=====================================================================================================
        return loss
    
    def compute_g_loss(self, generated_images_cs, generated_images_cs_raw, generated_images_cs_label, generated_images_c, style_image, sigma):
        loss = 0.0
        # adversarial loss
        # StarGAN_D==================================
        # loss_Gmain = 0
        # out = self.D(generated_images_cs)
        # loss_Gmain = adv_loss(out, 1)
        #============================================

        # StarGAN_D_pose==================================
        # loss_Gmain = 0
        # out = self.D(generated_images_cs, generated_images_cs_label)
        # loss_Gmain = adv_loss(out, 1)
        #============================================

        # EG3D_D===========================================================================================
        # Gmain: Maximize logits for generated images.
        # 消融========================================
        loss_Gmain = 0
        gen_style_img = {'image': generated_images_cs, 'image_raw': generated_images_cs_raw}
        gen_logits = self.D(gen_style_img, generated_images_cs_label)
        loss_Gmain = nn.functional.softplus(-gen_logits)    # 希望判别器将gen_style_img判别为越大的值（真）
        loss_Gmain = loss_Gmain.squeeze()
        # 消融========================================
        #==================================================================================================

        # style_loss and content_loss
        content_feats = self.vgg(generated_images_c)
        style_feats = self.encode_with_intermediate(style_image)
        g_t_feats = self.encode_with_intermediate(generated_images_cs)

        
        loss_c = 0
        loss_s = 0
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        # Density Regularization
        TVloss = 0
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        # TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)

        # id_loss
        loss_id = self.id_loss(generated_images_cs, generated_images_c)
        
        # 消融========================================
        # loss = hyperparameters.content_weight * loss_c + hyperparameters.style_weight * loss_s + hyperparameters.TV_weight * TVloss + hyperparameters.id_weight * loss_id
        # 消融========================================
        loss = hyperparameters.adv_weight * loss_Gmain + hyperparameters.content_weight * loss_c + hyperparameters.style_weight * loss_s + hyperparameters.TV_weight * TVloss + hyperparameters.id_weight * loss_id
        return loss, loss_Gmain, loss_c, loss_s, TVloss, loss_id

    # def forward(self, w,c):

    #     if w.shape[1]!= self.G.backbone.mapping.num_ws:
    #         w = w.repeat([1, self.G.backbone.mapping.num_ws, 1])
    #     # print("Hello1")
    #     generated_images = self.G.synthesis(w,c, noise_mode='const')['image']

    #     return generated_images

    # modify: generate cs_image and c_image
    def forward(self, w, c, style_image=None):
        add_weight_index = 7
        if w.shape[1]!= self.G.backbone.mapping.num_ws:
            w = w.repeat([1, self.G.backbone.mapping.num_ws, 1])
        # print("Hello1")
        if style_image != None:
            # style_f = self.vgg(style_image)
            # style_out = self.feature_match.forward(style_f)
            w_s = self.feature_match.forward(style_image)
            if w_s.shape[1]!= self.G.backbone.mapping.num_ws:
                w_s = w_s.repeat([1, self.G.backbone.mapping.num_ws, 1])
            # 消融===========================================================================
            # add_weight_new = self.add_weight.clone()
            # add_weight_new[:, add_weight_index:, :] = 0
            # w_cs = w * add_weight_new + w_s * (1 - add_weight_new)
            # images_cs = self.G.synthesis(ws=w_cs, c=c, style_out=w_s, noise_mode='const')
            #============================================================================

            # Our method==================================================================
            images_cs = self.G.synthesis(ws=w, c=c, style_out=w_s, noise_mode='const')
            # ============================================================================
            generated_images_cs = images_cs['image']
            generated_images_cs_raw = images_cs['image_raw']

            # modify
            initial_coordinates = torch.rand((w.shape[0], 1000, 3), device=w.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

            # 消融==============================================================================================================
            # sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), w_cs, update_emas=False)['sigma']
            # ==================================================================================================================
            # Our Method================================================================================================================
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), w, style_out=w_s, update_emas=False)['sigma']
            # ============================================================================================================================
        generated_images_c = self.original_G.synthesis(w,c, noise_mode='const')['image']

        return generated_images_cs, generated_images_cs_raw, generated_images_c, sigma


    # def initilize_e4e(self):
    #     ckpt = torch.load(paths_config.e4e, map_location='cpu')
    #     opts = ckpt['opts']
    #     opts['batch_size'] = hyperparameters.train_batch_size
    #     opts['checkpoint_path'] = paths_config.e4e
    #     opts = Namespace(**opts)
    #     self.e4e_inversion_net = pSp(opts)
    #     self.e4e_inversion_net.eval()
    #     self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
    #     toogle_grad(self.e4e_inversion_net, False)
        

    # def get_e4e_inversion(self, image):
    #     image = (image + 1) / 2
    #     new_image = self.e4e_image_transform(image[0]).to(global_config.device)
    #     _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
    #                                   input_code=False)
    #     if self.use_wandb:
    #         log_image_from_w(w, self.G, 'First e4e inversion')
    #     return w
