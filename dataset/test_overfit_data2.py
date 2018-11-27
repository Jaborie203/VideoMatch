import os
import numpy as np
import cv2
from scipy.misc import imresize
from torchvision import transforms
from dataset.helpers import *
from torch.utils.data import Dataset
from itertools import combinations


class DAVIS_OVER_FIT_TEST2(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 db_root_dir='../../../DAVIS-2016',
                 meanval=(104.00699, 116.66877, 122.67892),
                 transform=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        inputRes:inputResize scalar?
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.meanval = meanval
        self.transform = transform
        file_name = "drift-chicane"
        images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', file_name)))
        img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', file_name, x), images))
        lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', file_name)))
        labels = list(map(lambda x: os.path.join('Annotations/480p/', file_name, x), lab))

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        img_num = len(self.img_list)
        self.combin = list(combinations(list(range(img_num)), 2))


    def __len__(self):
        if self.train:
            return 600
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        if self.train:
            img_1, img_t, gt_1, gt_t = self.make_img_gt_pair_train(idx)
        else:
            img_1, img_t, gt_1, gt_t = self.make_img_gt_pair_val(idx)

        sample = {'image_1': img_1, 'image_t':img_t, 'gt_1': gt_1, 'gt_t':gt_t}
        #array->tensor
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair_train(self, idx):
        """
        Make the image-ground-truth pair
        """

        index = self.combin[idx]
        img_1 = cv2.imread(os.path.join(self.db_root_dir, self.img_list[index[0]]))
        img_t = cv2.imread(os.path.join(self.db_root_dir, self.img_list[index[1]]))
        label_1 = cv2.imread(os.path.join(self.db_root_dir, self.labels[index[0]]), 0)
        label_t = cv2.imread(os.path.join(self.db_root_dir, self.labels[index[1]]), 0)
        #img->array
        img_1 = np.array(img_1, dtype=np.float32)
        img_1 = (img_1 - img_1.min())/(img_1.max() - img_1.min())
        img_t = np.array(img_t, dtype=np.float32)
        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min())

        gt_1 = np.array(label_1, dtype=np.float32)
        gt_1 = gt_1/np.max([gt_1.max(), 1e-8])
        gt_t = np.array(label_t, dtype=np.float32)
        gt_t = gt_t / np.max([gt_t.max(), 1e-8])
        # name = self.img_list[idx].split('/')[-1].split('.')[0]
        return img_1, img_t, gt_1, gt_t
    def make_img_gt_pair_val(self, idx):
        """
        Make the image-ground-truth pair
        """
        img_1 = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        img_t = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        label_1 = cv2.imread(os.path.join(self.db_root_dir, self.labels[0]), 0)
        label_t = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        # img->array
        img_1 = np.array(img_1, dtype=np.float32)
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        img_t = np.array(img_t, dtype=np.float32)
        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min())

        gt_1 = np.array(label_1, dtype=np.float32)
        gt_1 = gt_1 / np.max([gt_1.max(), 1e-8])
        gt_t = np.array(label_t, dtype=np.float32)
        gt_t = gt_t / np.max([gt_t.max(), 1e-8])
        # name = self.img_list[idx].split('/')[-1].split('.')[0]
        return img_1, img_t, gt_1, gt_t

if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt
    from models import net
    from torch import nn

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = DAVIS_OVER_FIT_TEST1(db_root_dir='../../../DAVIS-2016',
                        train=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        # b = tens2image(data['image_1'])
        # plt.imshow(b)
        # print(data['image_1'][:, :, 3:15, 3:15])
        x = data['image_1']
        xt = data['image_t']
        lab = data['gt']
        # x[:, 0, :, :] = x[:, 0, :, :]*lab
        # x[:, 1, :, :] = x[:, 1, :, :]*lab
        # x[:, 2, :, :] = x[:, 2, :, :]*lab
        #or x[:, :, lab[0][0] < 1] = 0
        # x = x*lab
        # plt.imshow(tens2image(x))
        net1 = net.Net()
        x = net1(x)
        xt = net1(xt)
        downsample = nn.MaxPool2d(kernel_size=(32, 32),padding=5)
        lab = downsample(lab)
        print(x.shape)
        print(lab.shape)
        #lab = 1-lab
        m = x[:, :, lab[0][0]>0]
        m = m.squeeze()
        xt = xt.squeeze()
        print(xt.shape)
        # xt = xt.transpose((1,2,0))
        xt = xt.unsqueeze(dim=3).transpose(0, 3).squeeze()
        xt = xt.view(-1,2048)

        y = torch.mm(xt, m)
        y,_ = torch.topk(y, 5, dim=1)
        y = torch.mean(y, dim=1, keepdim = True)
        print(xt.shape, m.shape, y.shape)
        # plt.imshow(tens2image(lab))
        #data['image'].dtype = float32
        print(data['image_1'].shape, data['image_t'].shape, data['gt'].shape)
        if i == 10:
            break
        plt.show(block=True)