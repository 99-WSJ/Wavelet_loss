import torch.nn as nn
import torch
from pytorch_wavelets.pytorch_wavelets import DWTForward
import torch.nn.functional as F


class Wavelet_Net(nn.Module):
    def __init__(self):
        super(Wavelet_Net, self).__init__()
        self.criterion_MSE = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        # 可训练的 blur 核
        # kernel = torch.ones((3, 3), requires_grad=True)
        # self.kernel = torch.nn.Parameter(kernel)
        # 手工设定的 blur 核

        data = [[1.0, 2.5, 1.0],
                [2.5, 5.0, 2.5],
                [1.0, 2.5, 1.0]]
        self.kernel = torch.tensor(data, requires_grad=True, dtype=torch.float)

    def blur_train(self, x):
        batch = x.size()[0]
        channel = x.size()[1]

        # 尝试 使用 para9 = 1 - (para1 + .. + para8)
        # sum = self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para7 + self.para8
        # self.kernel = torch.tensor([[self.para1, self.para2, self.para3],
        #                             [self.para4, self.para5, self.para6],
        #                             [self.para7, self.para8, 1 - sum]], requires_grad=True)

        weights = self.kernel.expand(batch, channel, 3, 3).cuda()
        x = F.conv2d(x, weights, stride=1, padding=1)
        x = x / torch.sum(self.kernel)
        # x = x / 25
        return x

    def blur(self, x):
        batch = x.size()[0]
        channel = x.size()[1]

        # sum = self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para1 + self.para7 + self.para8
        # self.kernel = torch.tensor([[self.para1, self.para2, self.para3],
        #                             [self.para4, self.para5, self.para6],
        #                             [self.para7, self.para8, 1 - sum]], requires_grad=True)

        weights = self.kernel.expand(batch, channel, 3, 3)
        x = F.conv2d(x, weights, stride=1, padding=1)
        x = x / 14
        return x

    def forward(self, im, out):
        xfm = DWTForward(J=3, wave='db1', mode='zero')
        Yl_im, Yh_im = xfm(im)
        Yl_de, Yh_de = xfm(out)


        # smoothIm_0 = self.blur_train(torch.norm(Yh_im[0], dim=2))
        # smoothDe_0 = self.blur_train(torch.norm(Yh_de[0], dim=2))

        smoothIm_0 = self.blur(torch.norm(Yh_im[0], dim=2))
        smoothDe_0 = self.blur(torch.norm(Yh_de[0], dim=2))
        # print(self.kernel)

        loss01 = self.criterion_MSE(smoothIm_0 * 0.5, smoothDe_0 * 0.5)
        loss02 = self.criterion_L1(Yh_im[0] * 0.5, Yh_de[0] * 0.5)
        loss0 = 40 * (2 * loss01 + loss02)

        # loss0 = 30 * (2 * loss01 + loss02)

        loss12 = self.criterion_L1(Yh_im[1] * 0.25, Yh_de[1] * 0.25)
        loss1 = 40 * loss12

        loss22 = self.criterion_L1(Yh_im[2] * 0.125, Yh_de[2] * 0.125)
        loss2 = 20 * loss22

        # 最后的下采样结果
        loss3 = 15 * self.criterion_MSE(Yl_im * 0.125, Yl_de * 0.125)

        loss = loss0 + loss1 + loss2 + loss3

        return loss




