# bleu2vec
Modified BLEU score. Uses word2vec similarity instead of one to one similarity.

# How to use
Usage is quite simple: 

## Project folder structure
wmt17-metrics-task/wmy17-submitted-data/txt/ and in txt folder there are references, sources and system-outputs. Then the script files must be in wmy17-metrics-task folder and data must be as specified folders under txt.

## This script calculates scores for segments. -en calculates scores for all languages that end with -en (lv-en, fi-en etc). Could be also specifically lv-en or en-lv.
./segment_scoring_script.py -en location/to/en-unigram-model location/to/en-bigram-model location/to/en-trigram-model 

## This script calculates scores for systems. en-fi calculates scores for en-fi language pair, could also be -en or -fi.
./system_scoring_script.py en-fi location/to/en-unigram-model location/to/en-bigram-model location/to/en-trigram-model 

# Requirements
NLTK
pandas
numpy
gensim
