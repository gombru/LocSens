import json
import pycountry_convert as pc

anns_file = '../../../datasets/YFCC100M/splits/train.txt'
out_file_tags = '../../../datasets/YFCC100M/anns/tags_count.json'
out_file_continents = '../../../datasets/YFCC100M/anns/continent_count.json'

tags_count = {}
average_tags_per_image = 0
images_per_continent = {}
images_per_continent['EU'] = 0
images_per_continent['NA'] = 0
images_per_continent['SA'] = 0
images_per_continent['AS'] = 0
images_per_continent['AF'] = 0
images_per_continent['OC'] = 0

c = 0
for line in open(anns_file):
    if len(line) < 5: continue
    c+=1
    data = line.split(';')
    tags = data[1].split(',')

    for tag in tags:
        if tag in tags_count:
            tags_count[tag] += 1
        else:
            tags_count[tag] = 1

    average_tags_per_image += len(tags)

    country = data[2].replace('+',' ')
    # country[0] = country[0].upper()

    if len(country) > 2:
        try:
            country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            images_per_continent[continent_name] += 1
        except:
            # print("Country code for " + country + " not found.")
            continue

    if c % 100000 == 0:
        # break
        print(c)

print("Toral images: " + str(c))
print("Avg tags per image: " + str(float(average_tags_per_image) / c))

json.dump(tags_count, open(out_file_tags,'w'))

json.dump(images_per_continent, open(out_file_continents,'w'))