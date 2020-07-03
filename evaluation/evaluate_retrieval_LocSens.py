# Evaluate LocSens Retrieval

# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K

# But this code does location-sensitive evaluation: A retrieval result is only correct if it's near the location query.
# We use an evalution procedure similar to the one used in IMG2GPS paper:
# Different granularities: (street level (1km), city (25km), region (200km), country (750km) and continent (2500km)).


import aux
import random
from shutil import copyfile
import os
import json
import numpy as np
import geopy.distance


granularities = [99999999,2500,750,200,25,1] # granularities in km
granularities_str = ['street level (1km)', 'city (25km)', 'region (200km)', 'country (750km)', 'continent (2500km)', 'not sensitive (inf)']

def check_location(lat1,lon1,lat2,lon2):
    coords_1 = (lat1,lon1)
    coords_2 = (lat2, lon2)
    try:
        distance_km = geopy.distance.vincenty(coords_1, coords_2).km
    except:
        print("Error computing distance with gropy. Values:")
        print(coords_1)
        print(coords_2)
        distance_km = 100000
    results = np.zeros(len(granularities))
    for i,gr in enumerate(granularities):
        if distance_km <= gr:
            results[i] = 1
        else:
            break
    return results

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'geoModel_retrieval_ProgressiveFusionGaussianSampling_lr0_00005_var_0_epoch_57_ValLossNotBest0.018_var_0.pth'
model_name = model_name.replace('.pth','')
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
top_img_per_tag_path = dataset + 'results/' + model_name + '/tagLoc_top_img.json'

precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results


print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
tags_list = []
for line in open(tags_file):
    tags_list.append(line.replace('\n', ''))
print("Vocabulary size: " + str(len(tags_list)))

print("Reading top img per class")
top_img_per_tag = json.load(open(top_img_per_tag_path))
print("Num query pairs: "+ str(len(top_img_per_tag.keys())))

print("Reading tags and locations of testing images ...")
test_images_tags, test_images_latitudes, test_images_longitudes = aux.read_tags_and_locations(test_split_path)
print("Num test images: " + str(len(test_images_tags)))

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


print("Starting per-tag evaluation")
precisions = np.zeros(len(granularities), dtype=np.float32)
ignored = 0
used = 0
for i, (pair_info, cur_tag_top_img) in enumerate(top_img_per_tag.items()):

    d = pair_info.split(',')
    # print(d)
    cur_tag = d[0]
    cur_lat = float(d[1]) # * 180 - 90 # + 90) / 180
    cur_lon = float(d[2]) # * 360 -  180 #  180) / 360

    if cur_tag not in tags_test_histogram_filtered:
        ignored+=1
        continue

    used+=1

    if i % 100 == 0 and used > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*precisions[0]/used))


    # Compute Precision at k
    correct = False
    precisions_tag = np.zeros(len(granularities), dtype=np.float32)
    for top_img_idx in cur_tag_top_img:
        if cur_tag in test_images_tags[int(top_img_idx)]:
            # The image has query tag. Now check its location!
            results = check_location(cur_lat, cur_lon, test_images_latitudes[int(top_img_idx)], test_images_longitudes[int(top_img_idx)])
            for r_i,r in enumerate(results):
                if r == 1:
                    precisions_tag[r_i] += 1

    precisions_tag /= precision_k

    if precisions_tag[3] > 0.2:
        correct = True

    precisions += precisions_tag

    # Save img
    if save_img and correct and random.randint(0, 100) < 5:
        print("Saving results for: " + tag)
        if not os.path.isdir(dataset + '/retrieval_results/' + model_name + '/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + model_name + '/' + tag + '/')

        for top_img_idx in cur_tag_top_img:
            copyfile('../../../datasets/YFCC100M/test_img/' + str(top_img_idx) + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + str(top_img_idx) + '.jpg')

precisions /= used
precisions *= 100

print("Used query pairs: " + str(used))
print("Ignored pairs: " + str(ignored))
print("Location Sensitive Precision at " + str(precision_k) + ":")

for i,p in enumerate(precisions):
    print(granularities_str[-i-1] + ': ' + str(p))

print(model_name)