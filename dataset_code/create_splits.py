import os
import random

# Create train, val and test splits, checking that img exists

ann_file = open("../../../ssd2/YFCC100M/anns/anns_geo_filtered.txt", "r")

train_file = open("../../../ssd2/YFCC100M/splits/train.txt","w")
val_file = open("../../../ssd2/YFCC100M/splits/val.txt","w")
test_file = open("../../../ssd2/YFCC100M/splits/test.txt","w")

test_samples = 500000
val_samples = 250000

samples = []

c=0
print("Reading anns")
for line in ann_file:
    c+=1
    if c%100000 == 0: print(c)
    # Check that image exists
    if os.path.isfile("/home/Imatge/hd/datasets/YFCC100M/img/" + line.split(';')[0] + ".jpg"):
    	samples.append(line)
    	# print("File found")
	# print("File not found: " + "/home/Imatge/hd/datasets/YFCC100M/img/" + line.split(';')[0] + ".jpg")
ann_file.close()


random.shuffle(samples)

print("Toral files " + str(len(samples)))

print("Writing splits")
for i,s in enumerate(samples):
    if i < test_samples: test_file.write(s)
    elif i < (test_samples+val_samples): val_file.write(s)
    else: train_file.write(s)

print("DONE")

