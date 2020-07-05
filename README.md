# CDS_ADS

This directory contains code associated with the ACL 2020 short paper ["Learning to understand child directed and adult directed speech"](https://www.aclweb.org/anthology/2020.acl-main.1/).

July 5th 2020: Code is released but in need of cleaning; in particular regarding file paths. It is released now, so that interested ACL 2020 attendees can inspect it. A clean and more complete repository will be up by the end of this week.

## Data

The data used is from the [NewmanRatner corpus](https://childes.talkbank.org/access/Eng-NA/NewmanRatner.html) which is available through Childes.

## Semantic embeddings

The semantic embeddings used are [Sentence-BERT](https://www.aclweb.org/anthology/D19-1410/) embeddings made using the [Sentence-Transformers repository](https://github.com/UKPLab/sentence-transformers) (Reimers and Gurevych, 2019).

## Models

The model itself is based on [Merkx et al. (2019)](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/3067.html), and the code was based on (an earlier version of) the [Platalea repository](https://github.com/gchrupala/platalea) by Chrupała et al.
