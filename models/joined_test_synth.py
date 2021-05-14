import torch
import platalea.score as S
import platalea.rank_eval as E
import platalea.dataset as D
import numpy as np
import os
import json

def cross_test(dataset, variant, seed):
    # find best performing epoch on validation data, seleted based on validation set R@1
    val_results = json.load(open("crossval_results/"+variant+".json"))
    r1 = 0.0
    best_epoch = 0
    for epoch in val_results[variant][seed]:
        if val_results[variant][seed][epoch]["recall"]["1"] > r1 :
            r1 = val_results[variant][seed][epoch]["recall"]["1"]
            best_epoch = epoch
    print("best epoch for {}: {}".format(variant, best_epoch))
    print("r@1 at val: {}".format(r1))
    path = "{}/models/".format(variant.lower())
    best_model = "net.{}.{}.pt".format(seed, best_epoch)
    net = torch.load(path + best_model)
    net.cuda()
    net.eval()
    scores = S.score(net, dataset.dataset)
    print("Test scores:")
    print(scores)
    return scores

def multi_cross_test(dataset, variants, seeds):
    scores = {}
    # test different models on this dataset
    for variant in variants:
        print("Testing {}".format(variant), flush = "True")
        scores[variant] = {}
        for seed in seeds:
            scores[variant][seed] = cross_test(dataset, variant, seed)
    return scores

variants = ["ADS_synth", "CDS_synth"]
seeds = ['123', '234', '345']

root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/NewmanRatner/"

data = D.NewmanRatner_loader(split='test', register="joined_synth", root=root, batch_size=16, shuffle=False)
scores = multi_cross_test(data, variants, seeds)
json.dump(scores, open("joinedtest_results/syntheticspeech.json","w"))
