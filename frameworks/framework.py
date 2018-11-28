import torch as t
from torch import nn
# from matplotlib import pyplot as plt
# from dataset.helpers import *
import numpy as np
from utils.visualize import Visualizer
# import visdom
# from utils.visualize import Visualizer
class MyFrame():
    def __init__(self, net, loss, K):
        self.net = net
        self.loss = loss
        self.downsample = nn.MaxPool2d(kernel_size=(8,8), padding=1)
        self.K = K
        # self.vis = Visualizer(env='VideoMatch')

    def set_data(self,img_1, img_t, gt_1, gt_t):
        '''

        :param img_1: the first frame
        :param img_t: the t-th frame
        :param gt_1: the groudtruth of the first frame
        :param gt_t: the groudtruth of t-th frame
        :return: no return
        '''
        self.img_1 = img_1
        self.img_t = img_t
        self.gt_1 = gt_1
        self.gt_t = gt_t

    def forward(self):
        '''
        :function: forward compute
        :return: {output:loss, IoU:IoU}
        '''
        H, W = self.img_t.shape[2], self.img_t.shape[3]
        # plt.figure()

        X1 = self.net(self.img_1)
        Xt = self.net(self.img_t)
        FG = self.downsample(self.gt_1)
        mF = X1[:, :, FG[0][0] > 0]
        BG = 1 - FG
        mB = X1[:, :, BG[0][0] > 0]

        S_fg = self.soft_match(mF, Xt, self.K)
        S_bg = self.soft_match(mB, Xt, self.K)

        upsample = nn.UpsamplingBilinear2d(size=(H, W))
        S1 = upsample(S_fg)
        S2 = upsample(S_bg)
        S = t.cat((S2, S1), dim=1)
        pred = t.argmax(nn.functional.softmax(S, dim=1), dim=1)


        # self.vis.img(name='VideoMatch', img_=overlay_mask(im_normalize(tens2image(self.img_t.cpu())), tens2image(pred.cpu())).transpose(2,0,1))
        loss_out = self.loss(S, self.gt_t.squeeze(1).long())
        IoU = self.compute_IoU(pred.float().squeeze(), self.gt_t.squeeze())

        return loss_out, IoU, pred

    def soft_match(self, m, x, k):
        '''
        :param m: mF or mB, (1, 2048, |m|)
        :param x: the first frame x1 or the t-th frame xt, (1, 2048, 17, 25)
        :param k: the top K
        :return: matching score, (1, 1, h, w)
        '''
        h, w = x.shape[2], x.shape[3]
        x = x.squeeze()
        '''
        let x.shape = 17,25,2048
        '''
        x = x.unsqueeze(dim=3).transpose(0,3).squeeze()
        '''
        let x.shape = 405,2048
        '''
        x = x.view(-1, 2048)
        '''
        m.shape = 1x2048x|m|
        let m.shape = 2048x|m|
        '''
        m = m.squeeze()
        x = nn.functional.normalize(x, 2, 1)
        m = nn.functional.normalize(m, 2, 0)
        '''
        print(x.shape,x.sum())
        print(m.shape,m.sum())
        '''
        A = t.mm(x, m)
        m_abs = m.shape[1]
        k_adjust = min(k, m_abs)
        A, _ = t.topk(A, k_adjust, dim=1)
        A = t.mean(A, dim=1, keepdim = True)
        S = A.view(1, 1, h, w)
        return S

    def compute_IoU(self, pred, groundtruth):
        intersection = (pred*groundtruth).sum()
        union = pred.sum() + groundtruth.sum() - intersection
        return intersection/union
    def save_net(self, model_name='best_model.pth'):
        model_save_path = '/root/PycharmProjects/VideoMatch/checkpoint/' + model_name
        t.save(self.net.state_dict(), model_save_path)









