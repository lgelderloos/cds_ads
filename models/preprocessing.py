import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import json
import logging
from scipy.io.wavfile import read
import os
import numpy

def prepare_features(dataset, semantic_features):
    if (dataset == "ADS" or dataset == "CDS") and semantic_features == "sbert":
        newmanratner_features("sbert", dataset)
    else:
        raise ValueError("Unknown combination of dataset and features ({}, {})".format(dataset, semantic_features))

def newmanratner_features(bertversion, dataset):
    # prepare natural speech
    config = dict(audio=dict(dir='../data/NewmanRatner/audio/',
                             dataset=dataset, type='mfcc', delta=True, alpha=0.97,
                             n_filters=40, window_size=0.025, frame_shift=0.010),
                  bert=dict(dir="../data/NewmanRatner/", version=bertversion))
    newman_audio_features(config["audio"])
    # prepare synthetic speech
    config = dict(audio=dict(dir='../data/NewmanRatner/synthetic_speech/',
                             dataset=dataset, type='mfcc', delta=True, alpha=0.97,
                             n_filters=40, window_size=0.025, frame_shift=0.010),
                  bert=dict(dir="../data/NewmanRatner/", version=bertversion))
    newman_audio_features(config["audio"])
    # prepare semantic embeddings
    bert_features(dataset, config["bert"])

def newman_audio_features(config):
    directory = config['dir'] + "{}/".format(config['dataset'])
    files = os.listdir(directory)
    paths = [ directory + file for file in files ]
    files, features = audio_features(paths, config)
    torch.save(dict(features=features, filenames=files), config['dir'] + '{}_mfcc_features.pt'.format(config['dataset']))

def fix_wav(path):
    import wave
    logging.warning("Trying to fix {}".format(path))
    #fix wav file. In the flickr dataset there is one wav file with an incorrect
    #number of frames indicated in the header, causing it to be unreadable by pythons
    #wav read function. This opens the file with the wave package, extracts the correct
    #number of frames and saves a copy of the file with a correct header

    file = wave.open(path, 'r')
    # derive the correct number of frames from the file
    frames = file.readframes(file.getnframes())
    # get all other header parameters
    params = file.getparams()
    file.close()
    # now save the file with a new header containing the correct number of frames
    out_file = wave.open(path + '.fix', 'w')
    out_file.setparams(params)
    out_file.writeframes(frames)
    out_file.close()
    return path + '.fix'

def clean_sentence(sentence):
    clean_sentence = []
    for word in sentence:
        word = word.lower()
        # clean out underscores in fixed expressions
        if "_" in word:
            clean_sentence.extend(word.split("_"))
        else:
            clean_sentence.append(word)
    # return as string rather than as wordlist
    return " ".join(clean_sentence)

def bert_features(dataset, config):
    if config["version"] == "sbert":
        from sentence_transformers import SentenceTransformer
        bertmodel = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        with open(config["dir"]+"{}.json".format(dataset), "r") as f:
            data = json.load(f)
        ids = list(data.keys())
        plain_sents = []
        for item in ids:
            sentence = clean_sentence(data[item]['sentence'])
            plain_sents.append(sentence)

        print("encoding BERT sentences")

        sentembs = bertmodel.encode(plain_sents)
        torch.save(dict(features=sentembs, filenames=ids), config['dir'] + '{}_bert_features.pt'.format(dataset))
    else:
        raise NotImplementedError("unknown bert version: {}".format(config["version"]))

def audio_features(paths, config):
    # Adapted from https://github.com/gchrupala/speech2image/blob/master/preprocessing/audio_features.py#L45
    from platalea.audio.features import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
    if config['type'] != 'mfcc':
        raise NotImplementedError()

    output = []
    files = []
    n = 0
    t = len(paths)

    # only extracting mfccs for files for which you don't have them yet
    #existing_mfccs = torch.load(config['dir'] + '{}_mfcc_features.pt'.format(config['dataset']))
    #files = existing_mfccs['filenames']
    #output = existing_mfccs['features']
    ####
    for cap in paths:
        n += 1
        logging.info("Processing {}".format(cap))
        if cap.split("/")[-1] in files:
            print("in existing data", cap)
        else:
            try:
                input_data = read(cap)
            except ValueError:
                # try to repair the file
                path = fix_wav(cap)
                input_data = read(path)
            try:
                # sampling frequency
                fs = input_data[0]
                # get window and frameshift size in samples
                window_size = int(fs*config['window_size'])
                frame_shift = int(fs*config['frame_shift'])

                [frames, energy] = raw_frames(input_data, frame_shift, window_size)
                freq_spectrum = get_freqspectrum(frames, config['alpha'], fs, window_size)
                fbanks = get_fbanks(freq_spectrum, config['n_filters'], fs)
                features = get_mfcc(fbanks)

                #  add the frame energy
                features = numpy.concatenate([energy[:,None], features], 1)
                # optionally add the deltas and double deltas
                if config['delta']:
                    single_delta = delta(features, 2)
                    double_delta = delta(single_delta, 2)
                    features = numpy.concatenate([features, single_delta, double_delta], 1)
                output.append(torch.tensor(features))
                files.append(cap.split("/")[-1])
                print("{}/{} succesfully processed {}".format(t, n, cap), flush=True)
            except:
                print("{}/{} warning: processing audio failed for {}".format(t, n, cap), flush=True)
    return files, output
