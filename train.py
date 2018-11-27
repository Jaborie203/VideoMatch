import torch
import dataset.custom_transforms as tr
import torch as t
from torchvision import transforms
from dataset.davis_2016 import DAVIS2016
from dataset.test_overfit_data import DAVIS_OVER_FIT_TEST
from torch.utils.data import DataLoader
from models import Net
from models import Dilated_Net
from frameworks import MyFrame
from torch import nn
from torchnet import meter
from dataset.helpers import *
from PIL import Image
import numpy as np
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu" )
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))
K = 20
model = Dilated_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
frame = MyFrame(model, criterion, K)
transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])
dataset = DAVIS_OVER_FIT_TEST(db_root_dir='/root/DAVIS-2016',
                    train=True, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
loss_meter = meter.AverageValueMeter()
iou_meter = meter.AverageValueMeter()
for i in range(1000):
    loss_meter.reset()
    iou_meter.reset()
    for ii, data in enumerate(dataloader):
        img_1 = data['image_1'].to(device)
        img_t = data['image_t'].to(device)
        gt_1  = data['gt_1'].to(device)
        gt_t = data['gt_t'].to(device)
        if(gt_1.sum() == 0):
            continue

        frame.set_data(img_1, img_t, gt_1, gt_t)
        optimizer.zero_grad()
        loss, IoU = frame.forward()
        loss.backward()
        optimizer.step()
        # loss_temp = loss
        # loss_temp = loss_temp.detach().numpy
        loss_meter.add(loss.item())
        iou_meter.add(IoU.item())
        if(ii%5==0):
            print("loss:{}".format(loss_meter.value()[0]))
    print("train iter:{}, IoU:{}".format(i, iou_meter.value()[0]))




