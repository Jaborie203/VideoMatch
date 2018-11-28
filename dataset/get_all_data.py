import os
import numpy as np
import cv2
from scipy.misc import imresize
from torchvision import transforms
from dataset.helpers import *
from torch.utils.data import Dataset


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 db_root_dir='../../../DAVIS-2016',
                 meanval=(104.00699, 116.66877, 122.67892),
                 transform=None,
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        inputRes:inputResize scalar?
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.meanval = meanval
        self.transform = transform
        self.seq_name = seq_name
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'
        if self.seq_name == None:
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = {}
                labels = {}
                for i, seq in enumerate(seqs):
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list[i] = images_path
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels[i] = lab_path
        else:
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = list(map(lambda x: os.path.join('Annotations/480p/', str(seq_name), x), lab))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]
        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_1, img_t, gt_1, gt_t = self.make_img_gt_pair(idx)

        sample = {'image_1': img_1, 'image_t':img_t, 'gt_1': gt_1, 'gt_t':gt_t}
        #array->tensor
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        l = len(self.img_list[idx])
        if self.train:
            idx_list = np.random.choice(l, 2, replace=False)
        else:
            idx_list = np.random.choice(l, 2, replace=False)
            idx_list[0]=0
        img_1 = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx][idx_list[0]]))
        img_t = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx][idx_list[1]]))
        label_1 = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx][idx_list[0]]), 0)
        label_t = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx][idx_list[1]]), 0)
        #img->array
        img_1 = np.array(img_1, dtype=np.float32)
        img_1 = (img_1 - img_1.min())/(img_1.max() - img_1.min())
        img_t = np.array(img_t, dtype=np.float32)
        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min())

        gt_1 = np.array(label_1, dtype=np.float32)
        gt_1 = gt_1/np.max([gt_1.max(), 1e-8])
        gt_t = np.array(label_t, dtype=np.float32)
        gt_t = gt_t / np.max([gt_t.max(), 1e-8])
        return img_1, img_t, gt_1, gt_t


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = DAVIS2016(db_root_dir='../../../DAVIS-2016',
                        train=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        # b = tens2image(data['image_1'])
        # plt.imshow(b)
        # print(data['image_1'][:, :, 3:15, 3:15])
        x = data['image_1']
        xt = data['image_t']
        lab = data['gt_1']
        # x[:, 0, :, :] = x[:, 0, :, :]*lab
        # x[:, 1, :, :] = x[:, 1, :, :]*lab
        # x[:, 2, :, :] = x[:, 2, :, :]*lab
        #or x[:, :, lab[0][0] < 1] = 0
        # x = x*lab
        plt.imshow(np.random.rand(32, 32))

        #plt.imshow(overlay_mask(im_normalize(tens2image(x)), tens2image(lab)))
        if i == 10:
            break
    plt.show(block=True)