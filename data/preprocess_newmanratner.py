from bs4 import BeautifulSoup
import os
import json
from pydub import AudioSegment

def corpus_parser(path, folder, file):
    with open(path+folder+"/"+file) as f:
        soup = BeautifulSoup(f, "xml")
    data_parsed = {}
    n = 1
    sentence = [] # not every utterance has timing info; in those cases all
    #sentences without timing info are really in the stretch specified by the
    #first utterance that does.

    speaker = [] # list of speaker on a word by word basis
    for utterance in soup.find_all("u"):
        who = utterance["who"]
        item = {}
        item["file"] = file[:-4]
        item["folder"] = folder
        for word in utterance.find_all("w"):
            # remove children (this has stem, morphological tags, etc)
            for c in word.find_all():
                c.decompose()
            speaker.append(who)
            sentence.append(word.get_text()) # check this for morphological weirdness tags
        if utterance.find("media"):
            item["end"] = float(utterance.find("media")["end"])
            item["start"] = float(utterance.find("media")["start"])
            identifier = folder + "_" + item["file"] + "_" + str(n)
            item["sentence"] = sentence
            item["speaker"] = set(speaker)
            # only include item in data if it only contains speech by MOT
            if item["speaker"] == {"MOT"}:
                # only include if segment is at least 50 ms
                # take out the speaker annotation, it's unnecessary
                if (float(item["end"]) -float(item["start"])) >= 0.05:
                    item.pop("speaker")
                    data_parsed[identifier] = item
            # flush sentence & speaker for collecting next sentence
            sentence = []
            speaker = []
            n += 1
    return data_parsed

def store_wavs(data, path, split, folder, file):
    audio = open_audiofile(path+folder, file)
    for utterance in data:
        if "start" in data[utterance]:
            segment = audio[int(data[utterance]["start"]*1000):int((data[utterance]["end"]*1000)+1)]
            segment.export("/roaming/u1270964/cds/data/NewmanRatner/audio/{}/{}.wav".format(split, str(utterance)),
                        format="wav")
        else:
            print("no wav stored:", utterance, data[utterance])

def open_audiofile(dir, filename):
    if filename[-4:] == ".wav":
        audio = AudioSegment.from_wav(dir+"/"+filename)
    elif filename[-4:] == ".mp3":
        audio = AudioSegment.from_mp3(dir+"/"+filename)
    else:
        raise ValueError("Unknown filetype {}".format(filename))
    return audio

# walk through all the files you need.
# this is a list of folders that exists both in CDS and ADS
folders = ["07", "10", "11", "18", "24"]

########################################################
# ADS
ADS_path = "/roaming/u1270964/NewmanRatner/transcripts/Interviews/"
ADS_audio_path = "/roaming/u1270964/NewmanRatner/audio/Interviews/"
ADS_data = {}

for folder in folders:
    files = os.listdir(ADS_path+folder)
   # read the associated XML
    for file in files:
        data = corpus_parser(ADS_path, folder, file)
        audiofile = file[:-4]+".mp3"
        store_wavs(data, ADS_audio_path, "ADS", folder, audiofile)

        # add to data dctionary
        ADS_data.update(data)

with open("/roaming/u1270964/cds/data/NewmanRatner/ADS.json", "w") as f:
    json.dump(ADS_data, f)

# CDS
CDS_path = "/roaming/u1270964/NewmanRatner/transcripts/"
CDS_audio_path = "/roaming/u1270964/NewmanRatner/audio/0wav/"
CDS_data = {}

for folder in folders:
    files = os.listdir(CDS_path+folder)
    # read the associated XML
    for file in files:
        data = corpus_parser(CDS_path, folder, file)
        audiofile = file[:-4]+".wav"
        try:
            store_wavs(data, CDS_audio_path, "CDS", folder, audiofile)

            #    add to data dctionary
            CDS_data.update(data)
        except:
            print("warning: no file {}".format(audiofile))

with open("/roaming/u1270964/cds/data/NewmanRatner/CDS.json", "w") as f:
    json.dump(CDS_data, f)
