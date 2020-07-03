from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import numpy as np
import math


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split, img_backbone_model):

        self.root_dir = root_dir
        self.split = split
        self.img_backbone_model = img_backbone_model
        self.num_negatives = 6  # Number of negatives per anchor
        self.distance_thresholds = [2500, 750, 200, 25, 1]
        self.current_threshold = 1

        if 'train' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/train_filtered.txt'
            images_per_tag_file = '../../../datasets/YFCC100M/' + 'splits/images_per_tag_train_filtered.json'
        elif 'val' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/val.txt'
            images_per_tag_file = '../../../datasets/YFCC100M/' + 'splits/images_per_tag_val.json'
        else:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/test.txt'
            images_per_tag_file = '../../../datasets/YFCC100M/' + 'splits/images_per_tag_test.json'

        # Load img ids per tag
        print("Loading img ids per tag ...")
        self.images_per_tag = json.load(open(images_per_tag_file))

        # Load tag embeddings
        text_model_path = '../../../hd/datasets/YFCC100M/results/YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55/tags_embeddings.json'
        print("Loading textual model from: " + str(text_model_path))
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))
        print("Normalizing vocab")
        for k, v in self.text_model.items():
            v = np.asarray(v, dtype=np.float32)
            self.text_model[k] = v / np.linalg.norm(v, 2)
        self.tags_list = list(self.text_model.keys())

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/' + '/splits/' + split))
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.tags = []
        self.latitudes_or = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes_or = np.zeros(self.num_elements, dtype=np.float32)
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.img_embeddings = {}
        self.img_ids2idx_map = {}

        # Read data
        print("Reading split data ...")
        for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            tags_array = data[1].split(',')
            self.tags.append(tags_array)

            self.latitudes_or[i] = float(data[4])
            self.longitudes_or[i] = float(data[5])

            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])
            # Coordinates normalization
            self.latitudes[i] = (self.latitudes[i] + 90) / 180
            self.longitudes[i] = (self.longitudes[i] + 180) / 360

            self.img_ids2idx_map[int(data[0])] = i


        print("Data read. Set size: " + str(len(self.tags)))

        print("Latitudes min and max: " + str(min(self.latitudes)) + ' ; ' + str(max(self.latitudes)))
        print("Longitudes min and max: " + str(min(self.longitudes)) + ' ; ' + str(max(self.longitudes)))

        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 100000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            img_em_c += 1
            d = line.split(',')
            img_id = int(d[0])
            img_em = np.asarray(d[1:], dtype=np.float32)
            img_em = img_em / np.linalg.norm(img_em, 2)
            img_em[img_em != img_em] = 0
            self.img_embeddings[img_id] = img_em
        print("Img embeddings loaded: " + str(img_em_c))

    def __len__(self):
        return len(self.img_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding

    def __getdistanceFast__(self, lat1, lon1, lat2, lon2):
        deglen = 110.25
        x = lat1 - lat2
        y = (lon1 - lon2)*math.cos(lat2)
        return deglen*math.sqrt(x*x + y*y)

    def __getItemNotSharingTag__(self, idx, tag_str):
        while True:
            img_n_index = random.randint(0, self.num_elements - 1)
            if img_n_index != idx and tag_str not in self.tags[img_n_index]:
                break
        return img_n_index

    def __get_random_negative_triplet__(self, img_a_idx, img_n, tag_n, lat_n, lon_n, tag_str):

        # Select randomly the element to change

        # element_picker = random.randint(0, 2)
        # element_picker = random.randint(0, 1)
        element_picker = 0

        if element_picker == 0:  # Change image
            while True:
                negative_img_idx = random.randint(0, self.num_elements - 1)
                if negative_img_idx != img_a_idx and tag_str not in self.tags[negative_img_idx]:
                    break
                negative_img_idx = random.randint(0, self.num_elements - 1)
            try:
                img_n = self.img_embeddings[self.img_ids[negative_img_idx]]
            except:
                print("Couldn't find img embedding for negative image: " + str(
                    self.img_ids[negative_img_idx]) + ". Using 0s." + str())
                img_n = np.zeros(300, dtype=np.float32)

        elif element_picker == 1:  # Change tag
            while True:  # Check that image does not have the randlomly selected tag
                cur_tag_neg = random.choice(self.tags_list)
                if cur_tag_neg not in self.tags[img_a_idx]:
                    break
            tag_n = self.__getwordembedding__(cur_tag_neg)

        # TODO Change the location by a random (existing one) further away than a threshold
        elif element_picker == 3:  # Change location by a random one further away than a threshold
            lat_a_or = self.latitudes_or[img_a_idx]
            lon_a_or = self.longitudes_or[img_a_idx]
            while True:
                negative_location_idx = random.randint(0, self.num_elements - 1)
                lat_n_or = self.latitudes_or[negative_location_idx]
                lon_n_or = self.longitudes_or[negative_location_idx]
                # Check distance is further away than a threshold
                locations_distance = self.__getdistanceFast__(lat_a_or, lon_a_or, lat_n_or, lon_n_or)
                if locations_distance > self.distance_thresholds[self.current_threshold]:
                    lat_n = self.latitudes[negative_location_idx]
                    lon_n = self.longitudes[negative_location_idx]
                    break

        # TODO: Create hard triplets replacing the image with another sharing the tag but with a different location
        else:
            lat_a_or = self.latitudes_or[img_a_idx]
            lon_a_or = self.longitudes_or[img_a_idx]
            img_with_cur_tag = self.images_per_tag[tag_str]
            distances_checked = 0
            while True:
                img_n_index = self.img_ids2idx_map[random.choice(img_with_cur_tag)]
                if img_n_index == img_a_idx:
                    continue
                distances_checked += 1
                lat_n_or = self.latitudes_or[img_n_index]
                lon_n_or = self.longitudes_or[img_n_index]
                locations_distance = self.__getdistanceFast__(lat_a_or, lon_a_or, lat_n_or, lon_n_or)
                if locations_distance > self.distance_thresholds[self.current_threshold]:
                    break
                if distances_checked > 20:
                    img_n_index = self.__getItemNotSharingTag__(img_a_idx, tag_str)
                    break
            try:
                img_n = self.img_embeddings[self.img_ids[img_n_index]]
            except:
                print("Couldn't find img embedding for negative image: " + str(
                    self.img_ids[img_n_index]) + ". Using 0s.")
                img_n = np.zeros(300, dtype=np.float32)

        return img_n, tag_n, lat_n, lon_n

    def __getitem__(self, idx):

        # Initialize tensors container. In position 0 I put the anchor element. Others are negatives
        # But we initialize all with anchor info
        images = np.zeros((self.num_negatives + 1, 300), dtype=np.float32)
        tags = np.zeros((self.num_negatives + 1, 300), dtype=np.float32)
        latitudes = np.zeros((self.num_negatives + 1, 1), dtype=np.float32)
        longitudes = np.zeros((self.num_negatives + 1, 1), dtype=np.float32)

        try:
            images[:,:] = self.img_embeddings[self.img_ids[idx]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            images[:, :] = np.zeros(300, dtype=np.float32)

        # Select a random positive tag
        tag_p = random.choice(self.tags[idx])
        tags[:] = self.__getwordembedding__(tag_p)

        latitudes[:] = self.latitudes[idx]
        longitudes[:] = self.longitudes[idx]


        #### Negatives selection
        ### Multiple Random negative selection
        for n_i in range(1,self.num_negatives+1):
            images[n_i,:], tags[n_i,:], latitudes[n_i], longitudes[n_i] = self.__get_random_negative_triplet__(idx, images[0,:], tags[0,:], latitudes[0], longitudes[0], tag_p)

        # Build tensors
        images = torch.from_numpy(images)
        tags = torch.from_numpy(tags)

        latitudes = torch.from_numpy(latitudes)
        longitudes = torch.from_numpy(longitudes)

        return images, tags, latitudes, longitudes