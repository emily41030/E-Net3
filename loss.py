# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import copy
import torchvision.models as models
from base_networks import *
import torch.nn.functional as F
import tensorflow as tf
import numpy as np


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


############  def get_style_model_and_losses 介紹  ################
# 將風格影像切成patch 例如：16x16
# 計算style影像的feature 新增成style_loss層到model
# >> model
# Sequential(
#   (0): Normalization()
#   (conv_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (style_loss_0): StyleLoss()
#   (relu_1): ReLU()
#   (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (relu_2): ReLU()
#   (pool_2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv_3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (style_loss_1): StyleLoss()
#   (conv_4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (relu_4): ReLU()
#   (pool_4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv_5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (style_loss_2): StyleLoss()
########################################
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def get_style_model_and_losses(self, vgg, normalization_mean, normalization_std,
                                   style_img, content_img,):
        # 要取maxpooling後面的conv  作者取conv1.1  conv2.1  conv3.1
        style_layers = ['conv_1', 'conv_3', 'conv_5']
        # normalization module
        normalization = Normalization(
            normalization_mean, normalization_std)
        if self.gpu_mode:
            normalization.cuda()

        style_losses = []
        T_model = nn.Sequential(normalization)
        i = 0  # increment every time we see a conv
        styleLoss_num = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))

            T_model.add_module(name, layer)

            if name in style_layers:
                # add style loss:
                if self.gpu_mode:
                    T_model.cuda()
                styleLoss_num = patch_styleLoss(self, style_losses, T_model,
                                                style_img, styleLoss_num)
                # print(T_model)

        ###############################################
        # 將最後一個StyleLoss找出並將後面layer去掉
        # range(16-1, -1, -1)  假設len=16  表示範圍16到-1 每次-1
        for i in range(len(T_model) - 1, -1, -1):
            if isinstance(T_model[i], StyleLoss):
                # isinstance('22',int) -> False
                # isinstance(22,int) -> True
                break
        T_model = T_model[:(i + 1)]
        ###############################################

        return T_model, style_losses


def crop_to_patch(self, image):
    patch_size = self.patch_size
    interval = patch_size-self.overlap

    crop_img = None
    for w in range(0, self.crop_size, interval):
        for h in range(0, self.crop_size, interval):
            # [batch,3,16,16]
            patch = image[:, :, w:w+patch_size, h:h+patch_size]
            if w == 0 and h == 0:
                crop_img = patch
            else:
                crop_img = torch.cat(
                    ([crop_img, patch]), 0)
    # if crop_img :
    #     print("Error crop image to patch")
    return crop_img


def patch_styleLoss(self, style_losses, T_model, style_img, styleLoss_num):
    target_list = []
    if self.patchloss:
        # crop_style_img = crop_to_patch(self, style_img)
        style_img_resize = style_img.view(-1, 64, 16, 16)
        target_feature = T_model(style_img_resize).detach()
    else:
        target_feature = T_model(style_img).detach()
    target_list.append(target_feature)
    t_len = target_list.__len__()

    for j in range(t_len):
        # [384,384]             [b,64,16,16]
        style_loss = StyleLoss(target_list[j])
        T_model.add_module(
            "style_loss_{}".format(j+styleLoss_num), style_loss)
        style_losses.append(style_loss)
    x = t_len+styleLoss_num
    return x


def mse_loss(input, target):
    return (torch.sum((input - target)**2) / input.data.nelement())


def normalize(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    return v / tf.reduce_mean(v, axis=[1, 2, 3], keep_dims=True)


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    v = tf.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return tf.matmul(v, v, transpose_a=True)


def tf_op(patch_size, x):
    # [256,64,16,16]  ->  [256,16,16,64]
    x = torch.transpose(x, 1, 3)
    a, b, crop_size, c = x.size()
    c = x.shape[3]
    # x.numpy()  <-  pytorch_Tensor 轉 numpy
    # tf.convert_to_tensor()   <- numpy 轉 tf
    tf_x = tf.convert_to_tensor(x.cpu().numpy())

    #tf_x = normalize(tf_x)
    assert crop_size % patch_size == 0 and crop_size % patch_size == 0

    # [b * ?, h/p, w/p, c] [32,128,128,64]->[8192,8,8,64]
    tf_x = tf.space_to_batch_nd(
        tf_x, [patch_size, patch_size], [[0, 0], [0, 0]])
    # [p, p, b, h/p, w/p, c]  [8192,8,8,64]->[16,16,32,8,8,64]
    tf_x = tf.reshape(tf_x, [patch_size, patch_size, -1,
                             crop_size // patch_size, crop_size // patch_size, c])
    # [b * ?, p, p, c]  [16,16,32,8,8,64]->[32,8,8,16,16,64]
    tf_x = tf.transpose(tf_x, [2, 3, 4, 0, 1, 5])
    patches_tf_x = tf.reshape(tf_x, [-1, patch_size, patch_size, c])

    return patches_tf_x


def T_loss_op(patch_size, model, recon, x):
    recon = model(recon).detach()
    x = model(x)
    recon_r = tf_op(patch_size, recon)
    x_r = tf_op(patch_size, x)

    loss = tf.losses.mean_squared_error(
        gram_matrix(recon_r),
        gram_matrix(x_r),
        reduction=tf.losses.Reduction.MEAN
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # tf 轉 numpy
    loss = loss.eval(session=sess)
    # numpy 轉 pytorch_Tensor
    loss_temp = torch.from_numpy(np.array(loss, dtype=np.float64))
    # loss= Variable(loss)
    tf.reset_default_graph()
    return loss


class Loss:
    def loss_op(self, self_E,  recon_image, x_):
        loss_a = 0
        loss_output_m2 = 0
        loss_output_m5 = 0
        style_score = 0
        loss_G = 0
        loss_T = []
        if 'A' in self_E.model_loss:
            if self_E.loss_F == "BCEWithLogitsLoss":
                loss_a = self_E.criterion_GAN(
                    self_E.discriminator(recon_image), self_E.discriminator(x_))
                loss_G = self_E.criterion_GAN(
                    self_E.discriminator(recon_image), self_E.fake)
            elif self_E.loss_F == "MSE":
                loss_a = self.mse_loss(
                    self_E.discriminator(recon_image), self_E.discriminator(x_))
                loss_G = self.mse_loss(
                    self_E.discriminator(recon_image), self_E.fake)

        if 'P' in self_E.model_loss:
            # print("creat P loss")
            ############## VGG maxpooling_2 #################
            recon_loss_m2 = self_E.VGG_m2_model(recon_image)
            xs_loss_m2 = self_E.VGG_m2_model(x_)

            for re_m2, xs_m2 in zip(recon_loss_m2, xs_loss_m2):
                loss_output_m2 += mse_loss(re_m2, xs_m2)
            ############## VGG maxpooling_5 #################
            recon_loss_m5 = self_E.VGG_m5_model(recon_image)
            xs_loss_m5 = self_E.VGG_m5_model(x_)

            for re_m5, xs_m5 in zip(recon_loss_m5, xs_loss_m5):
                loss_output_m5 += mse_loss(re_m5, xs_m5)

        if 'T' in self_E.model_loss:
            #################################################
            # 輸入整張圖到model，將conv1_1，conv2_1，conv3_1的feature map取出
            # 並切成16*16大小做loss運算
            ###############################################
            # print("creat T loss")
            # self.T_model, style_losses = StyleLoss.get_style_model_and_losses(self, self.vgg19,
            #                                                                   self.normalization_mean, self.normalization_std, x_, recon_image)
            # self.T_model(recon_image)
            # # utils.print_network(self.T_model)
            # i = 0
            # for sl in style_losses:
            #     if i == style_losses.__len__():
            #         style_score += sl.loss*0.3
            #         i += 1
            #     else:
            #         style_score += sl.loss

            # style_score = style_score.cuda() if self.gpu_mode else style_score

            loss_conv1_1 = T_loss_op(self.patch_size,
                                     self_E.conv1_1, recon_image, x_)
            loss_conv2_1 = T_loss_op(self.patch_size,
                                     self_E.conv2_1, recon_image, x_)
            loss_conv3_1 = T_loss_op(self.patch_size,
                                     self_E.conv3_1, recon_image, x_)

            style_score = loss_conv1_1*0.3+loss_conv2_1+loss_conv3_1
            temp = style_score*1e-6
            tf.reset_default_graph()
            # if not style_score==0:
            #     print("style=%.4f   style=%.4f" % (style_score, temp))
            # style_score = loss_conv1_1*0.3s
            loss_T.append(loss_conv1_1)
            loss_T.append(loss_conv2_1)
            loss_T.append(loss_conv3_1)
        return loss_a, loss_output_m2, loss_output_m5, style_score, loss_G, loss_T
