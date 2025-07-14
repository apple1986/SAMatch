import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

def compute_loss(fusion, img_cat,confidence = None,img_map = None ):

   
    loss = structure_loss(fusion, img_cat,confidence,img_map)

    return  loss 



class Structure_Tensor(nn.Module):
    def __init__(self):
        super(Structure_Tensor, self).__init__()
        self.gradient_X = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            padding_mode='reflect'
        )
        self.X_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 1, 3)
        self.gradient_X.weight.data = self.X_kernel

        self.gradient_Y = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            padding_mode='reflect'
        )
        self.Y_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 3, 1)
        self.gradient_Y.weight.data = self.Y_kernel

    def forward(self, x, img_map = None):
        if x.shape[1] == 1:
            gray = x.squeeze(1)
        elif x.shape[1] == 2:
            r, g = x.unbind(dim=-3)
            gray = (r +  g ) / 2
        elif x.shape[1] == 4:
            r, g, b, a = x.unbind(dim=-3)
            gray = (r +  g + b + a) / 4 
        else:
            gray = torch.mean(x,dim=1)
        if  img_map :
            gray = gray.unsqueeze(dim=-3) * (img_map) * 255.0
        else:
            gray = gray.unsqueeze(dim=-3) * 255.0
        # 计算梯度
        Ix = self.gradient_X(gray)
        Iy = self.gradient_Y(gray)
        Ix2 = torch.pow(Ix, 2)
        Iy2 = torch.pow(Iy, 2)
        Ixy = Ix * Iy

        # 计算行列式和迹
        #  Ix2, Ixy
        #  Ixy, Iy2
        H = Ix2 + Iy2
        K = Ix2 * Iy2 - Ixy * Ixy

        # Flat平坦区域：H = 0;
        # Edge边缘区域：H > 0 & & K = 0;
        # Corner角点区域：H > 0 & & K > 0;

        h_ = 100

        Flat = torch.zeros_like(H)
        Flat[H < h_] = 1.0

        Edge = torch.zeros_like(H)
        Edge[(H >= h_) * (K.abs() <= 1e-6)] = 1.0

        Corner = torch.zeros_like(H)
        Corner[(H >= h_) * (K.abs() > 1e-6)] = 1.0

       # return 1.0 - Flat
        return 1.0 - Flat


test_transform = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(p=1.0),
        ],
        additional_targets={
            "image1": "image",
            "image2": "image",
        },
    )

def image_gradient(input, gray=False):
    input_device = input.device
    base_kernel = torch.tensor([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # base_kernel = torch.tensor([[0,  1,  0],
    #                             [1, -4,  1],
    #                             [0,  1,  0]], dtype=torch.float32).to(device)
    if gray:
        conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = base_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = kernel
        return conv_op(input)
    else:
        conv_op = nn.Conv2d(3, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = torch.zeros((1, 3, 3, 3), dtype=torch.float32).to(input_device)
        for i in range(3):
            kernel[:, i] = base_kernel
        conv_op.weight.data = kernel
        return conv_op(input)


def gradient_Gray(input):
    input_device = input.device
    conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # kernel = torch.tensor([[0,  1, 0],
    #                        [1, -4, 1],
    #                        [0,  1, 0]], dtype=torch.float32).to(input_device)
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = kernel

    return conv_op(input)

# plt.rcParams['figure.figsize']=(19.405, 14.41)
# # 显示图片

# def show_numpy(image, name):
#     plt.figure(name)
#    # plt.figure(figsize=(15.0, 10.0), dpi=100)
#     if len(image.shape) == 2:
#         plt.imshow(image, cmap='gray')
#     else:
#         plt.imshow(image)

#     plt.axis('off')  # 关掉坐标轴为off
#     plt.savefig("./gt_st.png", bbox_inches='tight', pad_inches=0)
#   #  plt.title('text title')  # 图像标题
#     plt.show()



# if __name__ == '__main__':
#     ST = Structure_Tensor()

#     a = ST.gradient_Y.weight.data
#     print(a)

#     # 原始图片
#     image = cv.imread(r'C:\Users\Administrator\Desktop\HSTHdr-main\FFT_page-0001.jpg', cv.IMREAD_UNCHANGED)
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     print(image.shape)


#     # 转为张量
#     img_tensor = test_transform(image=image)['image'].unsqueeze(0)
#     img_tensor.requires_grad = True
#     print(img_tensor.requires_grad)
#     print(img_tensor.shape, img_tensor.max(), img_tensor.min())

#     Ix = ST(img_tensor)
#     Ix = Ix[0, 0, :, :]
#     print(Ix.shape)

#    # G = image_gradient(img_tensor, gray=False)
#    # G = G[0, 0, :, :]

#     img_tensor_num = Ix.detach().numpy()
#    # img_tensor_num2 = G.detach().numpy()

#     print(img_tensor_num.shape, img_tensor_num.max(), img_tensor_num.min())
#     show_numpy(img_tensor_num, name='2')
#   # show_numpy(img_tensor_num2, name='1')



def gradient(x):

    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top 

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def create_structure(inputs):

    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W) # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:,0,:,:] = torch.sum(a_00,dim=1)
    structure[:,1,:,:] = torch.sum(a_01,dim=1)
    structure[:,2,:,:] = torch.sum(a_10,dim=1)
    structure[:,3,:,:] = torch.sum(a_11,dim=1)

    return structure

def structure_loss(fusion, img_cat,confidence,img_map):
    ST = Structure_Tensor().cuda()
    # st_fusion = create_structure(fusion)
    # st_input = create_structure(img_cat)
    # fusion.requires_grad = True
    # img_cat.requires_grad = True
    st_fusion = ST(fusion,img_map) 
    st_input = ST(img_cat,img_map) 
    # Frobenius norm
    losses = 0.0
    if confidence is not None:
        for i,(fus,inp) in enumerate(zip(st_fusion,st_input)):
            loss = torch.norm(fus - inp) * max(confidence[i],0.001)
            losses += loss
    else:
        losses = torch.norm(st_fusion - st_input)
    #loss = torch.norm(st_fusion - st_input)

    return losses


if __name__ == "__main__":
    fusion = torch.rand(5,3,4,4)
    img_1 = torch.rand(5,3,4,4)
    img_2 = torch.rand(5,1,4,4)
    img_cat = torch.cat([img_1,img_2],dim=1)

    loss = compute_loss(fusion, img_cat)

    print(loss)
