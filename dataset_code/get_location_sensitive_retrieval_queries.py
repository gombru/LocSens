# Generates retrieval queries for LocSens

import os
import random

random.seed(0)

dataset_folder = '../../../datasets/YFCC100M/'
split = 'test.txt'

num_query_pairs = 500000 # 100000
print("Using num query paris: " + str(num_query_pairs))


if not os.path.exists(dataset_folder + 'geosensitive_queries/'):
    os.makedirs(dataset_folder + 'geosensitive_queries/')

output_file_path = dataset_folder + 'geosensitive_queries/' + 'queries.txt'
output_file = open(output_file_path, "w")


print("Generating query tag-location pairs")
print("Reading tags and locations ...")

results = {}

for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
    if i % 100000 == 0 and i != 0: print(i)
    if i == num_query_pairs:
        print("Stopping at: " + str(i))
        break
    data = line.split(';')
    tag = random.choice(data[1].split(','))

    lat = float(data[4])
    lon = float(data[5])

    query = tag + ',' + str(lat) + ',' + str(lon) + '\n'
    output_file.write(query)

print("DONE")