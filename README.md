# CDS_ADS

This directory contains code associated with the ACL 2020 short paper ["Learning to understand child directed and adult directed speech"](https://www.aclweb.org/anthology/2020.acl-main.1/).


## Prerequisites

In order to run the code in this repository, make sure the following is installed:
PyTorch
pydub

For data preprocessing, the following are required in addition. See the section on data below.
BeautifulSoup (for processing the XML files in the original NewmanRatner dataset)
The Google cloud API for generating synthetic speech


## Data

### NewmanRatner
The data used is from the [NewmanRatner corpus](https://childes.talkbank.org/access/Eng-NA/NewmanRatner.html) which is available through Childes. Please download both the audio and transcripts and place them into data/NewmanRatner. Then run data/preprocess_newmanratner.py which will store each utterance by the mothers as a separate .wav, and produce ADS.json and CDS.json that link the wav-files to their transcription.

A splitting script is provided in the /data directory, make_even_split.py. It produces ADS_splits.json and CDS_splits.json. The splitting script includes those utterances in included.json. This excludes a small proportion of data for which speech synthesis failed. If you wish to recreate the experiments in the paper exactly, the splits we used are provided as ADS_splits.json and CDS_splits.json under /data/NewmanRatner.

### Synthetic speech

The synthetic speech was generated using the Google Cloud API. To replicate this process, you need to first obtain an API key. You can then use data/synthesize.py. Please note that there may have been changes to the Google Cloud TTS system since the data for the paper was generated in November 2019.

If you wish to use the exact synthetic speech used for the ACL paper, please contact Lieke Gelderloos.

### Semantic embeddings

The semantic embeddings used are [Sentence-BERT](https://www.aclweb.org/anthology/D19-1410/) embeddings made using the [Sentence-Transformers repository](https://github.com/UKPLab/sentence-transformers) (Reimers and Gurevych, 2019).

## Models

The model itself is based on [Merkx et al. (2019)](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/3067.html), and the code was based on (an earlier version of) the [Platalea repository](https://github.com/gchrupala/platalea) by Chrupała et al.

## Contact

If you have any questions regarding the paper or the code, please contact Lieke Gelderloos at l dot j dot gelderloos at tilburguniversity dot edu.
