from torchvision import datasets
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        # camera_id = filename.split('_')[2][0:2]
        return int(camera_id) - 1

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = random.randint(0, len(pos_index) - 1)
        return self.samples[pos_index[rand]]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        cam = self.cams[index]
        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)
        neg_path = self._get_neg_sample(target)

        sample = self.loader(path)
        pos = self.loader(pos_path[0])
        neg = self.loader(neg_path[0])

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, pos, neg


class SiameseDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, root, transform):
        super(SiameseDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        siamese_target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index].item()
        if siamese_target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2, label2 = self.data[siamese_index], self.labels[siamese_index].item()

        img1 = default_loader(img1)
        img2 = default_loader(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), siamese_target, (int(label1), int(label2))

    def __len__(self):
        return len(self.imgs)


class SggDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly 4 images
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform):
        super(SggDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1].astype(int)
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        cams = []
        for s in self.imgs:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img_num = 4
        label = self.labels[index].item()
        img, label = self.__getimgs_bylabel__(label, img_num)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __getimgs_bylabel__(self, label, img_num):
        if len(self.label_to_indices[label]) >= img_num:
            index = np.random.choice(self.label_to_indices[label], size=img_num, replace=False)
        else:
            index1 = np.random.choice(self.label_to_indices[label], size=len(self.label_to_indices[label]), replace=False)
            index2 = np.random.choice(self.label_to_indices[label], size=img_num - len(self.label_to_indices[label]),
                                      replace=True)
            index = np.concatenate((index1, index2))
        for i in range(img_num):
            img_temp = (self.data[index[i]])
            label_temp = (self.labels[index[i]])
            if type(label_temp) not in (tuple, list):
                label_temp = (label_temp,)
            label_temp = torch.LongTensor(label_temp)
            img_temp = default_loader(img_temp)
            if self.transform is not None:
                img_temp = self.transform(img_temp)
                img_temp = img_temp.unsqueeze(0)
            if i == 0:
                img = img_temp
                label = label_temp
            else:
                img = torch.cat((img, img_temp), 0)
                label = torch.cat((label, label_temp), 0)

        return img, label

