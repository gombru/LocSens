from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import image_processing
from PIL import Image


class YFCC_Dataset_Images_Test(Dataset):
    def __init__(self, root_dir, split, central_crop=224):

        self.root_dir = root_dir
        self.split = split
        self.central_crop = central_crop

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/splits/' + split))
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)

        # Read data
        print("Reading data ...")
        for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            data = line.split(';')
            self.img_ids[i] = int(data[0])
        print("Data read. Set size: " + str(len(self.img_ids)))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = '{}/{}{}'.format(self.root_dir, self.img_ids[idx], '.jpg')
        # Load and transform image
        image = Image.open(img_name)

        if self.central_crop != 0:
            image = image_processing.CenterCrop(image, self.central_crop, self.central_crop)

        im_np = np.array(image, dtype=np.float32)
        im_np = image_processing.PreprocessImage(im_np)
        # Build tensors
        img_tensor = torch.from_numpy(np.copy(im_np))
        img_id = str(self.img_ids[idx])

        return img_id, img_tensor