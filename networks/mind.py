import torch
import numpy as np
import torchvision
import SimpleITK as sitk
import torch.nn.functional as F
import matplotlib.pyplot as plt
def gaussian_kernel(sigma, sz):
    xpos_vec = np.arange(sz)
    ypos_vec = np.arange(sz)
    output = np.ones([1, 1,sz, sz], dtype=np.single)
    midpos = sz // 2
    for xpos in xpos_vec:
        for ypos in ypos_vec:
            output[:,:,xpos,ypos] = np.exp(-((xpos-midpos)**2 + (ypos-midpos)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return output
def torch_image_translate(input_, tx, ty, interpolation='nearest'):
    # got these parameters from solving the equations for pixel translations
    # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
    translation_matrix = torch.zeros([input_.shape[0], 3, 3], dtype=torch.float)
    translation_matrix[:, 0, 0] = 1.0
    translation_matrix[:, 1, 1] = 1.0
    translation_matrix[:, 0, 2] = -2*tx/(input_.size()[2]-1)
    translation_matrix[:, 1, 2] = -2*ty/(input_.size()[3]-1)
    translation_matrix[:, 2, 2] = 1.0
    grid = F.affine_grid(translation_matrix[:, 0:2, :], input_.size()).to(input_.device)
    wrp = F.grid_sample(input_.to(torch.float32), grid, mode=interpolation)
    return wrp
def Dp(image, xshift, yshift, sigma, patch_size):
    shift_image = torch_image_translate(image, xshift, yshift, interpolation='nearest')#将image在x、y方向移动I`(a)
    diff = torch.sub(image, shift_image).cuda()#计算差分图I-I`(a)
    diff_square = torch.mul(diff, diff).cuda()#(I-I`(a))^2
    res = torch.conv2d(diff_square, weight =torch.from_numpy(gaussian_kernel(sigma, patch_size)).cuda(), stride=1, padding=3)#C*(I-I`(a))^2
    return res
 
def MIND(image, patch_size = 7, neigh_size = 9, sigma = 2.0, eps = 1e-5,image_size0=256,image_size1=256,  name='MIND'):
    # compute the Modality independent neighbourhood descriptor (MIND) of input image.
    # suppose the neighbor size is R, patch size is P.
    # input image is 384 x 256 x input_c_dim
    # output MIND is (384-P-R+2) x (256-P-R+2) x R*R
    reduce_size = int((patch_size + neigh_size - 2) / 2)#卷积后减少的size
 
    # estimate the local variance of each pixel within the input image.
    Vimg = torch.add(Dp(image, -1, 0, sigma, patch_size), Dp(image, 1, 0, sigma, patch_size))
    Vimg = torch.add(Vimg, Dp(image, 0, -1, sigma, patch_size))
    Vimg = torch.add(Vimg, Dp(image, 0, 1, sigma, patch_size))#sum(Dp)
    Vimg = torch.div(Vimg,4) + torch.mul(torch.ones_like(Vimg), eps)#防除零
    # estimate the (R*R)-length MIND feature by shifting the input image by R*R times.
    xshift_vec = np.arange( -(neigh_size//2), neigh_size - (neigh_size//2))#邻域计算
    yshift_vec = np.arange(-(neigh_size // 2), neigh_size - (neigh_size // 2))#邻域计算
    iter_pos = 0
    for xshift in xshift_vec:
        for yshift in yshift_vec:
            if (xshift,yshift) == (0,0):
                continue
            MIND_tmp = torch.exp(torch.mul(torch.div(Dp(image, xshift, yshift,  sigma, patch_size), Vimg), -1))#exp(-D(I)/V(I))
            tmp = MIND_tmp[:, :, reduce_size:(image_size0 - reduce_size), reduce_size:(image_size1 - reduce_size)]
            if iter_pos == 0:
                output = tmp
            else:
                output = torch.cat([output,tmp], 1)
            iter_pos = iter_pos + 1
 
    # normalization.
    input_max, input_indexes = torch.max(output, dim=1)
    output = torch.div(output,input_max.unsqueeze(1))
 
    return output
def abs_criterion(in_, target):
    return torch.mean(torch.abs(in_ - target))
if __name__ == '__main__':
    patch_size=7
    neigh_size=9
    sigma=2.0
    eps=1e-5
    image_size0=256
    image_size1=256



    A = torch.randn(16, 1,256, 256).cuda()
    B = torch.randn(16, 1,256, 256).cuda()
 
   
    A_MIND = MIND(A,  patch_size, neigh_size, sigma, eps,image_size0,image_size1,  name='realA_MIND')
    B_MIND = MIND(B,  patch_size, neigh_size, sigma, eps,image_size0,image_size1,  name='realA_MIND')
    g_loss_MIND = abs_criterion(A_MIND, B_MIND)
    print('g_loss_MIND', g_loss_MIND)

