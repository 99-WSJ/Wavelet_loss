#coding:utf-8
import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
import os, torchvision
from PIL import Image
from torchvision import transforms as trans
import torch.nn.functional as F


def blur(self, x):
    batch = x.size()[0]
    channel = x.size()[1]

    # sum = self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para7 + self.para8
    # self.kernel = torch.tensor([[self.para1, self.para2, self.para3],
    #                             [self.para4, self.para5, self.para6],
    #                             [self.para7, self.para8, 1 - sum]], requires_grad=True)
    data = [[1.0, 2.5, 1.0],
            [2.5, 5.0, 2.5],
            [1.0, 2.5, 1.0]]
    self.kernel = torch.tensor(data, requires_grad=True, dtype=torch.float)

    weights = self.kernel.expand(batch, channel, 3, 3).cuda()
    x = F.conv2d(x, weights, stride=1, padding=1)
    x = x / 14
    return x


def test3():

    #J为分解的层次数,wave表示使用的变换方法
    xfm = DWTForward(J=3, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')

    img = Image.open('./2.jpeg')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    Yl, Yh = xfm(img)
    print(Yl.shape)
    print(len(Yh))
    Yh[0] = Yh[0] * 3
    Yh[1] = Yh[1] * 1
    Yh[2] = Yh[2] * 1
    Yl = Yl * 1
    # print(Yh[0].shape)
    img1 = ifm((Yl,Yh))
    img_grid1 = torchvision.utils.make_grid(img1, 1)
    torchvision.utils.save_image(img_grid1, 'img.jpg')

    for i in range(len(Yh)):
        print(Yh[i].shape)
        if i == len(Yh)-1:
            h = torch.zeros([4,3,Yh[i].size(3),Yh[i].size(3)]).float()
            h[0,:,:,:] = Yl*1
        else:
            h = torch.zeros([3,3,Yh[i].size(3),Yh[i].size(3)]).float()
        for j in range(3):
            if i == len(Yh)-1:
                h[j+1,:,:,:] = Yh[i][:,:,j,:,:]*1
            else:
                h[j,:,:,:] = Yh[i][:,:,j,:,:]*1
        if i == len(Yh)-1:
            print(h.shape)
            img_grid = torchvision.utils.make_grid(h, 2) #一行2张图片
        else:
            print(h.shape)
            img_grid = torchvision.utils.make_grid(h, 3)
        torchvision.utils.save_image(img_grid, 'img_grid1_{}.jpg'.format(i))

if __name__ == '__main__':
    test3()