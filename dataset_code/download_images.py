import urllib
import cStringIO
from joblib import Parallel, delayed
from PIL import Image
import os

# Downloads images from the created annotations file using Flickr static urls
# Resizes images so that their shorter size is 300px
# If an image already exist in the destination folder, it ignores it

def resize(im, minSize):
    w = im.size[0]
    h = im.size[1]
    if w < h:
        new_width = minSize
        new_height = int(minSize * (float(h) / w))
    if h <= w:
        new_height = minSize
        new_width = int(minSize * (float(w) / h))
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    return im

def download_save_image(id, url):
    try:
        if id in existing_ids:
            return
        img = Image.open(cStringIO.StringIO(urllib.urlopen(url).read()))
        img = resize(img, 300)
        image_path = dest_path + str(id) + '.jpg'
        img.save(image_path)
    except:
        print("Error with image")
    #print("Done")



dest_path = "/home/Imatge/hd/datasets/YFCC100M/img/"

# Read existing files
print("Reading existing files ids")
existing_ids = []
for filename in os.listdir(dest_path):
    id = int(filename.split('/')[-1].split('.')[0])
    existing_ids.append(id)

print("Num existing files: " + str(len(existing_ids)))

# Read anns
print("Reading anns")
ann_file = open("/home/Imatge/ssd2/YFCC100M/anns_gombru.txt")

to_download = {}
c=0
for line in ann_file:
    if c%1000000 == 0: print(c)
    # if c == 1000: break
    c+=1
    data = line.split(';')
    id = int(data[0])
    # if id in existing_ids:
    #     continue
    url = data[-1]
    to_download[id] = url

# print("Files to be downloaded: " + str(len(to_download)))

print("Downloading")
Parallel(n_jobs=64)(delayed(download_save_image)(id, url) for id, url in to_download.iteritems())
print("DONE")