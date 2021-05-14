import torch
import platalea.score as S
import platalea.rank_eval as E
import platalea.dataset as D
import numpy as np
import os
import json

def cross_test(dataset, variant, seed):
    scores = {}
    path = "{}/models/".format(variant.lower())
    models = os.listdir(path)
    for model in models:
        if model.split(".")[1] == seed:
            epoch = model.split(".")[2]# parse epoch from filename
            net = torch.load(path + model)
            net.cuda()
            net.eval()
            scores[epoch] = S.score(net, dataset.dataset)
    return scores

def multi_cross_test(dataset, variants, seeds):
    scores = {}
    # test different models on this dataset
    for variant in variants:
        print("Testing {}".format(variant), flush = "True")
        scores[variant] = {}
        for seed in seeds:
            scores[variant][seed] = cross_test(dataset, variant, seed)
            print("Done testing seed {}".format(seed), flush = True)
    return scores

variants = ["ADS", "CDS"]
seeds = ["123", "234", "345"]

root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/NewmanRatner/"

# for each of the valsets
for variant in variants:
    print("Testing on {}".format(variant), flush = True)
    # make a data loader
    data = D.NewmanRatner_loader(split='val', register=variant, root=root, batch_size=16, shuffle=False)
    scores = multi_cross_test(data, variants, seeds)
    json.dump(scores, open("crossval_results/{}.json".format(variant),"w"))
