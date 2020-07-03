import json



places_file = open("../../../hd/datasets/YFCC100M/original_data/yfcc100m_places")
out_file = open("../../../datasets/YFCC100M/places.txt",'w')

places = {}
c=0
for line in places_file:
    c+=1
    if c%100000 == 0: print(c)
    # if c == 100000: break
    metadata = line.split(',')
    if len(metadata) < 2:
        continue  # No geolocation info
    country = ""
    town = ""
    countries = []
    towns = []
    id = int(metadata[0].split('\t')[0])
    for field in metadata:
        if "Country" in field or "Town" in field or "District" in field or "State" in field:
        	name = field.split(':')[1]
        	name = name.lower()
        	name = name.replace('+','').replace(',','').replace('.','').replace('-','')
        	if name not in places:
        		places[name] = 1
        	else: 
        		places[name]+=1

s=0
for k,v in places.items():
	if v>5:
		out_file.write(k + '\n')
		s+=1

print("Total places: {}, Saved places: {}".format(len(places), s))
