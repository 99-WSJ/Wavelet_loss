import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward
import torch.nn.functional as F


class Wavelet_Net(nn.Module):
    def __init__(self):
        super(Wavelet_Net, self).__init__()
        self.criterion_MSE = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()
        
        data = [[1.0, 2.5, 1.0],
                [2.5, 5.0, 2.5],
                [1.0, 2.5, 1.0]]
        self.kernel = torch.tensor(data, requires_grad=True, dtype=torch.float)

    def blur_train(self, x):
        batch = x.size()[0]
        channel = x.size()[1]

        weights = self.kernel.expand(batch, channel, 3, 3).cuda()
        x = F.conv2d(x, weights, stride=1, padding=1)
        x = x / torch.sum(self.kernel)
        return x

    def blur(self, x):
        batch = x.size()[0]
        channel = x.size()[1]

        weights = self.kernel.expand(batch, channel, 3, 3)
        x = F.conv2d(x, weights, stride=1, padding=1)
        x = x / 14
        return x

    def forward(self, im, out):
        xfm = DWTForward(J=3, wave='db1', mode='zero')
        Yl_im, Yh_im = xfm(im)
        Yl_de, Yh_de = xfm(out)
        smoothIm_0 = self.blur(torch.norm(Yh_im[0], dim=2))
        smoothDe_0 = self.blur(torch.norm(Yh_de[0], dim=2))

        loss1 = self.criterion_MSE(smoothIm_0 * 0.5, smoothDe_0 * 0.5)
        loss2 = self.criterion_L1(Yh_im[0] * 0.5, Yh_de[0] * 0.5)
        loss3 = self.criterion_L1(Yh_im[1] * 0.25, Yh_de[1] * 0.25)
        loss4 = self.criterion_L1(Yh_im[2] * 0.125, Yh_de[2] * 0.125)

        loss5 = self.criterion_MSE(Yl_im * 0.125, Yl_de * 0.125)

        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return loss




