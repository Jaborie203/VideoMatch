import torch
import dataset.custom_transforms as tr
import torch as t
from torchvision import transforms
from dataset.davis_2016 import DAVIS2016
from torch.utils.data import DataLoader
from models import Dilated_Net
from frameworks import MyFrame
from torch import nn
from torchnet import meter
from utils.visualize import Visualizer
import os
from dataset.helpers import *
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))
K = 20
vis = Visualizer(env='VideoMatch28')

def train():

    model = Dilated_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
    frame = MyFrame(model, criterion, K)
    transform = transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])
    dataset = DAVIS2016(db_root_dir='/root/DAVIS-2016',
                        train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    loss_meter = meter.AverageValueMeter()
    iou_meter = meter.AverageValueMeter()
    max_iou=0
    for i in range(1000):
        loss_meter.reset()
        iou_meter.reset()
        for ii, data in enumerate(dataloader):
            img_1 = data['image_1'].to(device)
            img_t = data['image_t'].to(device)
            gt_1 = data['gt_1'].to(device)
            gt_t = data['gt_t'].to(device)
            if gt_1.sum() == 0:
                continue

            frame.set_data(img_1, img_t, gt_1, gt_t)
            optimizer.zero_grad()
            loss, IoU, pred = frame.forward()
            loss.backward()
            optimizer.step()
            vis.img(name='train', img_=overlay_mask(im_normalize(tens2image(img_t.cpu())),tens2image(pred.cpu())).transpose(2, 0, 1))
            # loss_temp = loss
            # loss_temp = loss_temp.detach().numpy
            loss_meter.add(loss.item())
            iou_meter.add(IoU.item())
        print("train epoch:{}, loss:{} IoU:{}".format(i, loss_meter.value()[0], iou_meter.value()[0]))
        vis.log("train epoch:{}, loss:{} IoU:{}".format(i, loss_meter.value()[0], iou_meter.value()[0]),
                'loss_iou')
        vis.plot('train_loss', loss_meter.value()[0])
        vis.plot('train_iou', iou_meter.value()[0])
        if (i+1) % 5 == 0:
            with torch.no_grad():
                iou_val = evaluation(frame=frame)
            if max_iou < iou_val:
                max_iou = iou_val
                frame.save_net('train11_28.pth')


def evaluation(frame=None):
    with torch.no_grad():
        if frame is None:
            from models.dilated_net import Dilated_Net
            model = Dilated_Net(pretrained=False)
            model.load_state_dict(t.load("/root/PycharmProjects/VideoMatch/checkpoint/train.pth"))
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            frame = MyFrame(model, criterion, K)
        loss_meter_val = meter.AverageValueMeter()
        iou_meter_val = meter.AverageValueMeter()
        db_root_dir = '/root/DAVIS-2016'
        fname = 'val_seqs'
        with open(os.path.join(db_root_dir, fname + '.txt')) as f:
            seqs = f.readlines()
            val_files_len = len(seqs)
        for val_index in range(val_files_len):
            transform = transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])
            dataset_val = DAVIS2016(db_root_dir='/root/DAVIS-2016',
                                    train=False, transform=transform, val_index=val_index)
            dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=1)
            for ii, data in enumerate(dataloader_val):
                img_1 = data['image_1'].to(device)
                img_t = data['image_t'].to(device)
                gt_1 = data['gt_1'].to(device)
                gt_t = data['gt_t'].to(device)
                if (gt_1.sum() == 0):
                    continue
                frame.set_data(img_1, img_t, gt_1, gt_t)
                loss_val, iou_val, pred = frame.forward()
                vis.img(name='val',
                        img_=overlay_mask(im_normalize(tens2image(img_t.cpu())), tens2image(pred.cpu())).transpose(2, 0,
                                                                                                                   1))
                loss_meter_val.add(loss_val.item())
                iou_meter_val.add(iou_val.item())

            print("evaluation on sequence:{}, loss: {}, IoU: {}".format(seqs[val_index], loss_meter_val.value()[0],
                                                                        iou_meter_val.value()[0]))
            vis.log("evaluation on sequence:{}, loss: {}, IoU: {}".format(seqs[val_index], loss_meter_val.value()[0],
                                                                          iou_meter_val.value()[0]), 'loss_iou')
        vis.plot('val_loss', loss_meter_val.value()[0])
        vis.plot('val_iou', iou_meter_val.value()[0])
        vis.log("avg of all sequences:{}".format(iou_meter_val.value()[0]), 'loss_iou')
        if frame is not None:
            return iou_meter_val.value()[0]

if __name__ == '__main__':
    import fire
    fire.Fire()

