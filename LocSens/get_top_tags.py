# For each image-location pair get its score with each tag, and save the top10

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset_test_tagging
import time

dataset_folder = '../../../hd/datasets/YFCC100M/'
split = 'test.txt'
img_backbone_model = 'YFCC_MCC'

batch_size = 1
workers = 0
ImgSize = 224

model_name = 'LocSens_tagging.pth'
model_name = model_name.replace('.pth', '')

gpus = [1]
gpu = 1

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/images_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0': 'cuda:1', 'cuda:2': 'cuda:1', 'cuda:3': 'cuda:1'})

model_test = model.Model_Test_Tagging()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_tagging.YFCC_Dataset(dataset_folder, img_backbone_model, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

print("Loading textual model ...")
text_model_path = '../../../hd/datasets/YFCC100M/results/YFCC_MCC/tags_embeddings.json'
text_model = json.load(open(text_model_path))
print("Vocabulary size: " + str(len(text_model)))
print("Normalizing vocab")
for k, v in text_model.items():
    v = np.asarray(v, dtype=np.float32)
    text_model[k] = v / np.linalg.norm(v, 2)

print("Putting vocab in a tensor using ordered tag list")
tags_tensor = np.zeros((100000, 300), dtype=np.float32)
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
for i, line in enumerate(open(tags_file)):
    tag = line.replace('\n', '').lower()
    tags_tensor[i, :] = np.asarray(text_model[tag], dtype=np.float32)
tags_tensor = torch.autograd.Variable(torch.from_numpy(tags_tensor).cuda(gpu))
print("Tags tensor created")

print("Running model...")
results = {}
with torch.no_grad():
    model_test.eval()
    for i, (img_id, img, lat, lon) in enumerate(test_loader):
        st = time.time()
        img = torch.autograd.Variable(img)
        lat = torch.autograd.Variable(lat)
        lon = torch.autograd.Variable(lon)
        # Try to run it with BS of 100k (vocab size). Else create here mini-batches and stack results
        outputs = model_test(img, tags_tensor, lat, lon, gpu)
        outputs = outputs.squeeze(-1)
        top_values, top_tag_indices = outputs.topk(200)
        results[str(img_id)] = {}
        results[str(img_id)]['tags_indices'] = np.array(top_tag_indices.cpu()).tolist()
        results[str(img_id)]['tags_scores'] = np.array(top_values.cpu()).tolist()
        end = time.time()
        print(str(i) + ' / ' + str(len(test_loader)) + " Time per iter: " + str(end-st))

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")