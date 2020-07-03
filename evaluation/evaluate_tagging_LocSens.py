# Image tagging evaluation for classification models
# Measures Accuracy at K, typically k=1,10
# Accuracy at k: Is one of the K top predicted hashtags in the vocab?

import aux
import torch
import torch.nn as nn
import random
from shutil import  copyfile
import os
import json
import numpy as np

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_MLC.pth'
model_name = model_name.replace('.pth', '')
print(model_name)
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
results_path = dataset + 'results/' + model_name + '/images_test.json'
accuracy_k = 10 # Compute accuracy at k (will also compute it at 1)
save_img = False # Save some random image tagging resultsYFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55.pth

# Also compute the number of different correcty predicted tags
predicted_tags = []
correctly_predicted_tags = []
test_set_tags = []

print("Loading tag list ...")
tags_list = []
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
for line in open(tags_file):
    tags_list.append(line.replace('\n', ''))
print("Vocabulary size: " + str(len(tags_list)))


print("Reading results ...")
results = json.load(open(results_path))
print("Len of results: " +str(len(results)))
print("Reading tags of testing images ... ")
test_images_tags = aux.read_tags(test_split_path)



print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0
total_precision_at_k = 0.0

f_a = 0
f_b = 0
f_c = 0
img_evaluated = 0

for i, (img_id, img_result) in enumerate(results.items()):

    if i == 2000:
        break

    if i % 1000 == 0: 
        print(i)
    result_img_tags = img_result['tags_indices']

    ### REMOVE PLACES
    # cur_img_tags = result_img_tags

    if len(cur_img_tags) <=1 and len(cur_img_tags[0]) == 0:
        continue

    # Check frequency of most frequent tags in results
    if 0 in cur_img_tags:
        f_a +=1
    if 1 in cur_img_tags:
        f_b +=1
    if 2 in cur_img_tags:
        f_c +=1


    img_id = int(img_id.replace('\'','').replace('[','').replace(']',''))

    #### COMPUTE TAGS DIVERSITY
    # Gather total test set tags
    for tag in test_images_tags[img_id]:
        if tag not in test_set_tags:
            test_set_tags.append(tag)
    # Gather total predicted tags
    for cur_img_tag in cur_img_tags[0:accuracy_k]:
        if tags_list[cur_img_tag] in test_images_tags[img_id] and tags_list[cur_img_tag] not in correctly_predicted_tags:
                correctly_predicted_tags.append(tags_list[cur_img_tag])
        if tags_list[cur_img_tag] not in predicted_tags:
            predicted_tags.append(tags_list[cur_img_tag])

    # Compute Accuracy at 1
    if tags_list[cur_img_tags[0]] in test_images_tags[img_id]:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    aux = []
    correct = False
    for cur_img_tag in cur_img_tags[0:accuracy_k]:
        aux.append(tags_list[cur_img_tag])
        if tags_list[cur_img_tag] in test_images_tags[img_id]:
            total_accuracy_at_k += 1
            correct = True
            break

    cur_p = 0
    for cur_img_tag in cur_img_tags[0:accuracy_k]:
        if tags_list[cur_img_tag] in test_images_tags[img_id]:
            cur_p += 1
    cur_p/=len(test_images_tags[img_id])
    total_precision_at_k += cur_p

    # Save img
    if save_img:# and correct and random.randint(0,10) < 1:
        # print("Saving")
        img_id = str(img_id)
        if not os.path.isdir(dataset + '/tagging_results/' + model_name + '/' + img_id + '/'):
            os.makedirs(dataset + '/tagging_results/' + model_name + '/' + img_id + '/')
        copyfile('../../../datasets/YFCC100M/test_img/' + img_id + '.jpg', dataset + '/tagging_results/' + model_name + '/' + img_id + '/' + img_id + '.jpg')
        # Save txt file with gt and predicted tags
        with open(dataset + '/tagging_results/' + model_name + '/' + img_id + '/tags.txt','w') as outfile:
            outfile.write('GT_tags\n')
            for tag in test_images_tags[int(img_id)]:
                outfile.write(tag + ' ')
            outfile.write('\nPredicted_tags\n')
            for idx in cur_img_tags:
                outfile.write(tags_list[idx] + ' ')

    img_evaluated += 1



total_accuracy_at_1 /= img_evaluated
total_accuracy_at_k /= img_evaluated

total_precision_at_k /= img_evaluated


print("Accuracy at 1:" + str(total_accuracy_at_1*100))
print("Accuracy at " + str(accuracy_k) + " :" + str(total_accuracy_at_k*100))

print("Precision at " + str(accuracy_k) + " :" + str(total_precision_at_k*100))


print(f_a)
print(f_b)
print(f_c)


print("Num of different correctly predicted tags: " + str(len(correctly_predicted_tags)))
print("Num of different test set tags: " + str(len(test_set_tags)))
print("Percentage of correctly predicted hashtags compared to total different test set hashtags:" + str(float(len(correctly_predicted_tags)) / len(test_set_tags)))

# print("Num of different predicted tags: " + str(len(predicted_tags)))
# print("Percentage of  predicted hashtags compared to total different test set hashtags:" + str(float(len(predicted_tags)) / len(test_set_tags)))

predicted_tags_filtered = []
for tag in predicted_tags:
    if tag in test_set_tags:
        predicted_tags_filtered.append(tag)

print("Num of different predicted tags filtered: " + str(len(predicted_tags_filtered)))
print("Percentage of  predicted hashtags compared to total different test set hashtags:" + str(float(len(predicted_tags_filtered)) / len(test_set_tags)))
