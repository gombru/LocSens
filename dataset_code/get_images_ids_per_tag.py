import json

anns_file = '../../../datasets/YFCC100M/splits/train_filtered.txt'
out_file = '../../../datasets/YFCC100M/splits/images_per_tag_train_filtered.json'

tags_img_ids = {}

print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
for line in open(tags_file):
    tags_img_ids[line.replace('\n', '')] = []

c = 0
for line in open(anns_file):
    if len(line) < 5: continue
    c+=1
    data = line.split(';')
    tags = data[1].split(',')

    for tag in tags:
        tags_img_ids[tag].append(int(data[0]))

    if c % 1000 == 0: print(c)


json.dump(tags_img_ids, open(out_file,'w'))

print("DONE")