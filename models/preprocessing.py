import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import json
import logging
from scipy.io.wavfile import read
import os
import numpy

def prepare_features(dataset, semantic_features):
    if dataset == "flickr8k" and semantic_features == "visual":
        flickr8k_features()
    elif dataset == "flickr8k" and semantic_features == "sbert":
        #this is temporary, TODO
        raise NotImplementedError()
    elif (dataset == "ADS" or dataset == "CDS") and semantic_features == "sbert":
        newmanratner_features("sbert", dataset)
    else:
        raise ValueError("Unknown combination of dataset and features ({}, {})".format(dataset, semantic_features))

def newmanratner_features(bertversion, dataset):
    # prepare natural speech
    config = dict(audio=dict(dir='/roaming/u1270964/cds/data/NewmanRatner/audio/',
                             dataset=dataset, type='mfcc', delta=True, alpha=0.97,
                             n_filters=40, window_size=0.025, frame_shift=0.010),
                  bert=dict(dir="/roaming/u1270964/cds/data/NewmanRatner/", version=bertversion))
    newman_audio_features(config["audio"])
    # prepare synthetic speech
    config = dict(audio=dict(dir='/roaming/u1270964/cds/data/NewmanRatner/synthetic_speech/',
                             dataset=dataset, type='mfcc', delta=True, alpha=0.97,
                             n_filters=40, window_size=0.025, frame_shift=0.010),
                  bert=dict(dir="/roaming/u1270964/cds/data/NewmanRatner/", version=bertversion))
    newman_audio_features(config["audio"])
    # prepare semantic embeddings
    bert_features(dataset, config["bert"])

def newman_audio_features(config):
    directory = config['dir'] + "{}/".format(config['dataset'])
    files = os.listdir(directory)
    paths = [ directory + file for file in files ]
    files, features = audio_features(paths, config)
    torch.save(dict(features=features, filenames=files), config['dir'] + 'test_{}_mfcc_features.pt'.format(config['dataset']))

def flickr8k_features():
    config = dict(audio=dict(dir='/roaming/gchrupal/datasets/flickr8k/', type='mfcc', delta=True, alpha=0.97, n_filters=40, window_size=0.025, frame_shift=0.010),
                  image=dict(dir='/roaming/gchrupal/datasets/flickr8k/', model='resnet'))
    flickr8k_audio_features(config['audio'])
    flickr8k_image_features(config['image'])

def flickr8k_audio_features(config):
    directory = config['dir'] + 'flickr_audio/wavs/'
    files = [ line.split()[0] for line in open(config['dir'] + 'wav2capt.txt') ]
    paths = [ directory + file for file in files ]
    features = audio_features(paths, config)
    torch.save(dict(features=features, filenames=files), config['dir'] + 'mfcc_features.pt')


def flickr8k_image_features(config):
    directory = config['dir'] + 'Flickr8k_Dataset/Flicker8k_Dataset/'
    data = json.load(open(config['dir'] + 'dataset.json'))
    files =  [ image['filename'] for image in data['images'] ]
    paths = [ directory + file for file in files ]

    features = image_features(paths, config).cpu()
    torch.save(dict(features=features, filenames=files), config['dir'] + 'resnet_features.pt')


def image_features(paths, config):
    if config['model'] == 'resnet':
        model = models.resnet152(pretrained = True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif config['model'] == 'vgg19':
        model = models.vgg19_bn(pretrained = True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    if torch.cuda.is_available():
        model.cuda()
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    device = list(model.parameters())[0].device
    def one(path):
        logging.info("Extracting features from {}".format(path))
        im = PIL.Image.open(path)
        return prep_tencrop(im, model, device)

    return torch.stack([one(path) for path in paths])



def prep_tencrop(im, model, device):
    # Adapted from: https://github.com/gchrupala/speech2image/blob/master/preprocessing/visual_features.py#L60

    # some functions such as taking the ten crop (four corners, center and horizontal flip) normalise and resize.
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406],
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)

    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    im = im.to(device)
    # there are some grayscale images in mscoco that the vgg and resnet networks
    # wont take
    if not im.size()[1] == 3:
        im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])
    activations = model(im)
    return activations.mean(0).squeeze()

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
    # return as strng rather than as wordlist
    return "_".join(clean_sentence)

def bert_features(dataset, config):
    #dict(version=bertversion, feattype="sentemb"))
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
        torch.save(dict(features=sentembs, filenames=ids), config['dir'] + '{}/bert_features.pt'.format(dataset))
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


    # this is temporary: only extracting mfccs for files for hwich you didnt have them yet!
    existing_mfccs = torch.load(config['dir'] + '{}_mfcc_features.pt'.format(config['dataset']))
    files = existing_mfccs['filenames']
    output = existing_mfccs['features']
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
                    single_delta= delta (features, 2)
                    double_delta= delta(single_delta, 2)
                    features= numpy.concatenate([features, single_delta, double_delta], 1)
                output.append(torch.tensor(features))
                files.append(cap.split("/")[-1])
                print("{}/{} succesfully processed {}".format(t, n, cap), flush=True)
            except:
                print("{}/{} warning: processing audio failed for {}".format(t, n, cap), flush=True)
    return files, output
