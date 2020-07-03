from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import time


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, img_backbone_model, split):

        self.root_dir = root_dir
        self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + img_backbone_model + '/test.txt'
        self.split = split

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.img_embeddings_path))
        # self.num_elements = 100
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.img_embeddings = {}

        # Read data
        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 100000 == 0 and i != 0: print(i)
            # if i == 10000: break
            img_em_c += 1
            d = line.split(',')
            img_id = int(d[0])
            self.img_ids[i] = img_id
            img_em = np.asarray(d[1:], dtype=np.float32)
            img_em = img_em / np.linalg.norm(img_em, 2)
            self.img_embeddings[img_id] = img_em
        print("Img embeddings loaded: " + str(img_em_c))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        try:
            img = self.img_embeddings[self.img_ids[idx]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            img = np.zeros(300, dtype=np.float32)

            # Build tensors
        img = torch.from_numpy(img)
        img_id = str(self.img_ids[idx])

        return img_id, img