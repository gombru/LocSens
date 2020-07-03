# Gets images geolocation form GT

import json

print("Extracting images geolocations")
dataset_file = open("/home/Imatge/hd/datasets/YFCC100M/original_data/yfcc100m_dataset")

out_file = open("/home/Imatge/hd/datasets/YFCC100M/anns/img_geolocations.json", 'w')

selected = 0

out_dict = {}

c = 0
for line in dataset_file:
    c+=1
    if c%20000 == 0: print(c)
    # if c == 1000: break

    metadata = line.split('\t')

    id = int(metadata[1])

    if metadata[12] == "" or metadata[13] == "":
        continue

    lat = float(metadata[13])
    lon = float(metadata[12])

    # Image selected: Has geolocation info and tags
    selected+=1

    out_dict[id] = (lat,lon)

json.dump(out_dict, out_file)


print("Selected number of images: " + str(selected))
