## Architechture
lpips_type = 'alex'
first_inv_type = 'w+'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
content_weight = 1.0
style_weight = 1.0
adv_weight = 0.7
id_weight = 0.5
TV_weight = 0.1

## Steps
LPIPS_value_threshold = 0.06
max_pti_steps = 350
first_inv_steps = 450
max_images_to_invert = 30

# modify: 训练步数
max_iter = 120001

## Optimization
pti_learning_rate = 1e-5
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = True
