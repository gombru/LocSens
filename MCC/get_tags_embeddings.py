# Get the embeddings of the tags

import os
import torch.utils.data
import model
import json
import numpy as np

dataset_folder = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_MCC'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tags_embeddings.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
tags = []
for line in open(tags_file):
    tags.append(line.replace('\n', ''))

tag_embeddings = {}
weights = model_test.module.output_layer.weight
for tag_idx, tag in enumerate(tags):
	tag_embeddings[tag] = np.array(weights[tag_idx,:].detach().cpu()).tolist()


print("Writing results")
json.dump(tag_embeddings, output_file)

print("DONE")