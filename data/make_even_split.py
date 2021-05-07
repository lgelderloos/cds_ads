import sys
import json
import random
import os

# load the inclded data
with open("included.json", "r") as f:
    included = json.load(f)

# first, process ADS

ads_wavfiles = included["ADS"]
ids = set([wav[:-4] for wav in ads_wavfiles])
valsize = 1000
testsize = 1000

print("Size of ADS: {} items".format(len(ids)))
test = set(random.sample(ids, testsize))
rest = ids - test
val = set(random.sample(rest, valsize))
train = rest - val
trainsize_ads = len(train)
splitdict = {"test": list(test),
           "val": list(val),
           "train": list(train)}

for split in splitdict:
    print(split, ": ", len(splitdict[split]), " items")
with open("NewmanRatner/ADS_splits.json", "w") as f:
    json.dump(splitdict, f)


# then make same nr of utterances splits for CDS, putting rest in restsplit

cds_wavfiles = included["CDS"]
ids = set([wav[:-4] for wav in cds_wavfiles])

print("Size of CDS: {} items".format(len(ids)))

test = set(random.sample(ids, testsize))
rest = ids - test
val = set(random.sample(rest, valsize))
rest = rest - val
train = set(random.sample(rest, trainsize_ads))
rest = rest - train

splitdict = {"test": list(test),
           "val": list(val),
             "train": list(train),
             "rest": list(rest)}

for split in splitdict:
    print(split, ": ", len(splitdict[split]), " items")

with open("NewmanRatner/CDS_splits.json", "w") as f:
    json.dump(splitdict, f)
