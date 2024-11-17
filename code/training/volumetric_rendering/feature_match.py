# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
from turtle import forward
from collections import namedtuple
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d')
from training.volumetric_rendering.function import adaptive_instance_normalization as adain
from training.volumetric_rendering.function import calc_mean_std
from training.volumetric_rendering.op import fused_leaky_relu
from training.volumetric_rendering.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm
from training.networks_stylegan2 import FullyConnectedLayer
from training.volumetric_rendering import psp_encoders, style_transformer_encoders
from training.volumetric_rendering.psp_options import TrainOptions


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class FeatureMatch(nn.Module):
    def __init__(self, sampling_multiplier = 1):
        super(FeatureMatch, self).__init__()
        self.hidden_dim = 64
        self.sampling_multiplier = sampling_multiplier
        self.iden = Upsample(512, True)
        self.res1 = nn.Sequential(
            Upsample(512, True),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512)
        )
        self.active = nn.LeakyReLU(0.01, True)

        self.conv1 = nn.Conv2d(512, 512*3, 3, padding=1)
        self.conv2 = nn.Conv2d(512*3, 512*6, 3, padding=1)

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(32, self.hidden_dim, lr_multiplier=0.01),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + 32, lr_multiplier=0.01)
        )
    
    def forward(self, style):
        x = self.iden(style)
        y = self.res1(style)
        res = self.res2(y)
        out = x + res
        out = self.active(out)
        style_features = self.conv1(out)
        if self.sampling_multiplier == 2:
            style_features = self.conv2(style_features)
        
        sf = torch.cat([ sf_i.unsqueeze(4) for sf_i in style_features.chunk(48*self.sampling_multiplier, 1) ], dim=4)
        N, C, H, W, D = sf.shape
        sf = sf.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)

        N1, M1, C1 = sf.shape
        sf = sf.view(N1*M1, C1)

        sf = self.net(sf)
        sf = sf.view(N1, M1, -1)     # net(x): [1, 786432, 33]
        sf_rgb = torch.sigmoid(sf[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sf_sigma = sf[..., 0:1]

        return {'rgb': sf_rgb, 'sigma': sf_sigma}
    
    # def get_params(self):
    #     wd_params, nowd_params = [], []
    #     for name, module in self.named_modules():
    #         if isinstance(module, (nn.Linear, nn.Conv2d)):
    #             wd_params.append(module.weight)
    #             if not module.bias is None:
    #                 nowd_params.append(module.bias)
    #         elif isinstance(module,  nn.BatchNorm2d):
    #             nowd_params += list(module.parameters())
    #     return wd_params, nowd_params

# BlendGAN

FeatureOutput = namedtuple(
    "FeatureOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class FeatureExtractor(nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self, load_pretrained_vgg=True):
        super(FeatureExtractor, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=load_pretrained_vgg).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return FeatureOutput(**output)


class StyleEmbedder(nn.Module):
    def __init__(self, load_pretrained_vgg=True):
        super(StyleEmbedder, self).__init__()
        self.feature_extractor = FeatureExtractor(load_pretrained_vgg=load_pretrained_vgg)
        self.feature_extractor.eval()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, img):
        N = img.shape[0]
        features = self.feature_extractor(self.avg_pool(img))

        grams = []
        for feature in features:
            gram = gram_matrix(feature)
            grams.append(gram.view(N, -1))
        out = torch.cat(grams, dim=1)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


# without pretain
# class StyleEncoder(nn.Module):
#     def __init__(
#         self,
#         style_dim=512,
#         n_mlp=4,
#         load_pretrained_vgg=True,
#     ):
#         super().__init__()

#         self.style_dim = style_dim

#         e_dim = 610304

#         # fixed vgg19 to extract feature
#         self.embedder = StyleEmbedder(load_pretrained_vgg=load_pretrained_vgg).requires_grad_(False)

#         # feature to z_s
#         layers = []

#         layers.append(EqualLinear(e_dim, style_dim, lr_mul=1, activation='fused_lrelu'))
#         for i in range(n_mlp - 2):
#             layers.append(
#                 EqualLinear(
#                     style_dim, style_dim, lr_mul=1, activation='fused_lrelu'
#                 )
#             )
#         layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation=None))
#         self.embedder_mlp = nn.Sequential(*layers)

#         # z_s to w_s
#         layers = [PixelNorm()]

#         for i in range(n_mlp):
#             layers.append(
#                 EqualLinear(
#                     style_dim, style_dim, lr_mul=0.01, activation='fused_lrelu'
#                 )
#             )
#         self.embedding = nn.Sequential(*layers) 

#     def forward(self, image):
#         z_embed = self.embedder_mlp(self.embedder(image))  # [N, 512]
#         w_embed = self.embedding(z_embed)
#         return w_embed


# without pretain
class StyleEncoder(nn.Module):
    def __init__(
        self,
        style_dim=512,
        n_mlp=4,
        load_pretrained_vgg=True,
    ):
        super().__init__()

        self.style_dim = style_dim

        e_dim = 610304
        self.embedder = StyleEmbedder(load_pretrained_vgg=load_pretrained_vgg).requires_grad_(False)

        layers = []

        layers.append(EqualLinear(e_dim, style_dim, lr_mul=1, activation='fused_lrelu'))
        for i in range(n_mlp - 2):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=1, activation='fused_lrelu'
                )
            )
        layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation=None))
        self.embedder_mlp = nn.Sequential(*layers)

        print("load embedder")
        checkpoint1 = torch.load('/data1/sch/BlendGAN-main/other_models/model_embedder_mlp.pth')
        embedder_dict = checkpoint1['embedder_mlp']
        self.embedder_mlp.load_state_dict(embedder_dict)

    def forward(self, image):
        z_embed = self.embedder_mlp(self.embedder(image))  # [N, 512]
        return z_embed


class FeatureMatch_p(nn.Module):
    def __init__(
        self,
        style_dim=512,
        n_mlp=8,
        lr_mlp=0.01,
        load_pretrained_vgg=True,
    ):
        super(FeatureMatch_p, self).__init__()

        self.style_dim = style_dim

        self.embedder = StyleEncoder(style_dim=512, n_mlp=4, load_pretrained_vgg=load_pretrained_vgg)
        # for name, parms in self.embedder.named_parameters():
        #     print("-->name:", name)
        #     # print("para:", parms)
        #     print("-->grad_requirs", parms.requires_grad)

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.embedding = nn.Sequential(*layers)                      # MLPs: z_s to w_s

        print("load embedding")
        checkpoint2 = torch.load('/data1/sch/BlendGAN-main/other_models/model_embedding.pth')
        embedder_dict = checkpoint2['embedding']
        self.embedding.load_state_dict(embedder_dict)

        # for name, parms in self.embedding.named_parameters():
        #     print("-->name:", name)
        #     # print("para:", parms)
        #     print("-->grad_requirs", parms.requires_grad)
    
    def forward(self, image):
        z_embed = self.embedder(image)  # [N, 512]
        w_embed = self.embedding(z_embed)
        return w_embed


class Backbone(nn.Module):
	def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
		super(Backbone, self).__init__()
		assert input_size in [112, 224], "input_size should be 112 or 224"
		assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
		assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
		blocks = get_blocks(num_layers)
		if mode == 'ir':
			unit_module = bottleneck_IR
		elif mode == 'ir_se':
			unit_module = bottleneck_IR_SE
		self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  nn.BatchNorm2d(64),
									  nn.PReLU(64))
		if input_size == 112:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               Flatten(),
			                               nn.Linear(512 * 7 * 7, 512),
			                               nn.BatchNorm1d(512, affine=affine))
		else:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               Flatten(),
			                               nn.Linear(512 * 14 * 14, 512),
			                               nn.BatchNorm1d(512, affine=affine))

		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = nn.Sequential(*modules)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.body(x)
		x = self.output_layer(x)
		return l2_norm(x)


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/data1/sch/EG3D-projector-master/eg3d/networks/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count



def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load('/data1/sch/EG3D-projector-master/eg3d/networks/model_ir_se50.pth')
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)

	def forward(self, x, input_code=False):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		return codes

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None


class StyleTransformer(nn.Module):

	def __init__(self, opts):
		super(StyleTransformer, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = style_transformer_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		self.encoder = nn.DataParallel(self.encoder)
		layers = [PixelNorm()]
		for i in range(8):
			layers.append(EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'))
		self.mapping = nn.Sequential(*layers)
		self.mapping = nn.DataParallel(self.mapping)
		# self.decoder = nn.DataParallel(Generator(self.opts.output_size, 512, 8))
		# for name, parms in self.mapping.named_parameters():
		# 	print("-->name:", name)
		# 	print("-->grad_requirs", parms.requires_grad)
		# self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading style transformer from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load((self.opts.checkpoint_path), map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load('/data1/sch/EG3D-projector-master/eg3d/networks/model_ir_se50.pth')
			self.encoder.module.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(('/data1/sch/EG3D-projector-master/eg3d/networks/stylegan2-ffhq-config-f.pt'), map_location='cpu')
			self.mapping.module.load_state_dict(ckpt['g_ema'], strict=False)
			# self.decoder.module.load_state_dict(get_keys(ckpt, 'decoder'), strict=True) # For cars dataset.

	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):
		if input_code:
			codes = x
		else:
			# Get w from MLP
			z = self.encoder.module.z                # [1, 18, 512]
			n, c = z.shape[1], z.shape[2]
			b = x.shape[0]
			z = z.expand(b, n, c).flatten(0, 1)      # [18, 512]
			query = self.mapping.module(z).reshape(b, n, c)    # [1, 18, 512]
			codes = self.encoder(x, query)

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        
		return codes

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None


if __name__ == "__main__":
    # net = FeatureMatch(sampling_multiplier = 2)
    # # x = torch.randn(10, 1536, 64, 64)
    # # x = torch.cat([ x_i.unsqueeze(1) for x_i in x.chunk(48, 1) ], dim=1)
    # x = torch.randn(1, 512, 64, 64)
    # out = net(x)
    # style_colors = out['rgb']
    # style_densities = out['sigma']
    # print(style_colors.shape)
    # print(style_densities.shape)

    # # 参数
    # batch_size = 1
    # num_rays = 16384
    # samples_per_ray = 96
    # # 特征融合
    # style_colors = style_colors.reshape(batch_size, num_rays, samples_per_ray, style_colors.shape[-1]) 
    # style_colors = style_colors.permute(0, 3, 2, 1).reshape(batch_size, 32*samples_per_ray, 128, 128)
    # content_colors = torch.randn(1, 32*samples_per_ray, 128, 128)
    # print(style_colors.shape)

    # colors = adain(content_colors, style_colors)
    # print(colors.shape)
    # colors = colors.reshape(batch_size, 32, samples_per_ray, 128*128).permute(0, 3, 2, 1)
    # print(colors.shape)

    # BlendGAN
    # device = 'cuda'
    # device = 'cpu'
    # net = FeatureMatch_p().to(device)
    # print("Information")
    # for name, parms in net.named_parameters():
    #     print("-->name:", name)
    #     # print("para:", parms)
    #     print("-->grad_requirs", parms.requires_grad)
    # x = torch.randn(5, 3, 512, 512).to(device)
    # y = net(x)
    # print(y.shape)

    # id_loss = IDLoss().to(device).eval().requires_grad_(False)
    # for name, parms in id_loss.named_parameters():
    #     print("-->name:", name)
    #     # print("para:", parms)
    #     print("-->grad_requirs", parms.requires_grad)
    # y_hat = torch.randn(5, 3, 512, 512).to(device)
    # y = torch.randn(5, 3, 512, 512).to(device)
    # loss = id_loss(y_hat, y)
    # print(loss)

    opts = TrainOptions().parse()
    device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
    opts.device = device
    net = StyleTransformer(opts).to(device)
	# for name, parms in net.named_parameters():
	# 	print("-->name:", name)
	# 	print("-->grad_requirs", parms.requires_grad)
    x = torch.randn(1, 3, 512, 512).to(device)
    latent = net.forward(x)
    print(latent)


