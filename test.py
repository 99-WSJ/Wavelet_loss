#coding:utf-8
import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
import os, torchvision
from PIL import Image
from torchvision import transforms as trans
import torch.nn.functional as F


def test():
    xfm = DWTForward(J=3, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='haar')

    img = Image.open('./ando.jpeg')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    Yl, Yh = xfm(img)
    # print(Yl.shape)
    # print(len(Yh))
    Yh[0] = Yh[0] * 1
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
            img_grid = torchvision.utils.make_grid(h, 2)
        else:
            print(h.shape)
            img_grid = torchvision.utils.make_grid(h, 3)
        torchvision.utils.save_image(img_grid, 'img_grid1_{}.jpg'.format(i))

if __name__ == '__main__':
    test()
