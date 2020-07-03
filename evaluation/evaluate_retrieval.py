# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K

import aux
import torch
import torch.nn.functional as F
import torch.nn as nn
import operator
import random
from shutil import copyfile
import os
import json
import numpy as np

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_MCC'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + 'results/' + model_name + '/images_embeddings_test.json'
tags_embeddings_path = dataset + 'results/' + model_name + '/tags_embeddings.json'
# If using GloVe embeddings directly
# print("Using GloVe embeddings")
# tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results

measure =  'distance' # 'distance'

distance_norm = 2 # 2
if measure == 'distance':
    print("Using pairwise distance with norm: " + str(distance_norm))

normalize = True # Normalize img embeddings and tag embeddings using L2 norm
print("Normalize tags and img embeddings: " + str(normalize))

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ...")
test_images_tags = aux.read_tags(test_split_path)

if normalize:
    print("Using L2 normalization on img AND tag embeddings")

print("Get tags with at least k appearances in test images")
tags_test_histogram = {}
for id, tags in test_images_tags.items():
    for tag in tags:
        if tag not in tags_test_histogram:
            tags_test_histogram[tag] = 1
        else:
            tags_test_histogram[tag] += 1

print("Total tags in test images: " + str(len(tags_test_histogram)))

print("Filtering vocab")
tags_test_histogram_filtered = {}
for k, v in tags_test_histogram.items():
    if v >= precision_k:
        tags_test_histogram_filtered[k] = v

print("Total tags in test images with more than " + str(precision_k) + " appearances: " + str(
    len(tags_test_histogram_filtered)))

print("Putting image embeddings in a tensor")
# Put img embeddings in a tensor
img_embeddings_tensor = torch.zeros([len(img_embeddings), embedding_dim], dtype=torch.float32).cuda()
img_ids = []
for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):
    img_ids.append(img_id)
    img_np_embedding = np.asarray(img_embedding, dtype=np.float32)
    if normalize:
        img_np_embedding /= np.linalg.norm(img_np_embedding)
    img_embeddings_tensor[i, :] = torch.from_numpy(img_np_embedding)
del img_embeddings

print("Starting per-tag evaluation")
dist = nn.PairwiseDistance(p=distance_norm)
total_precision = 0.0
for i, (tag, test_appearances) in enumerate(tags_test_histogram_filtered.items()):
    if i % 100 == 0 and i > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*total_precision/i))

    tag_np_embedding = np.asarray(tags_embeddings[tag], dtype=np.float32)
    if normalize:
        tag_np_embedding /= np.linalg.norm(tag_np_embedding)

    tag_embedding_tensor = torch.from_numpy(tag_np_embedding).cuda()

    if measure == 'distance':
        distances = dist(img_embeddings_tensor, tag_embedding_tensor)
        indices_sorted = np.array(distances.sort(descending=False)[1][0:precision_k].cpu())

    else:
        print("Measure not found: " + str(measure))
        break

    # Compute Precision at k
    correct = False
    precision_tag = 0.0
    for idx in indices_sorted:
        if tag in test_images_tags[int(img_ids[idx])]:
            correct = True
            precision_tag += 1

    precision_tag /= precision_k
    total_precision += precision_tag

    # Save img
    if save_img and correct and random.randint(0, 100) < 5:
        print("Saving results for: " + tag)
        if not os.path.isdir(dataset + '/retrieval_results/' + model_name + '/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + model_name + '/' + tag + '/')

        for idx in indices_sorted:
            copyfile('../../../datasets/YFCC100M/test_img/' + img_ids[idx] + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + img_ids[idx] + '.jpg')

total_precision /= len(tags_test_histogram_filtered)

print("Precision at " + str(precision_k) + ": " + str(total_precision*100))