#!/usr/bin/python3
import os
from time import time
from nltk import word_tokenize
from gensim.models import KeyedVectors

from bc4 import corpus_bleu
import sys


hypothesis_folder = os.path.join("wmt17-submitted-data", "txt", "system-outputs", "newstest2017")
reference_folder = os.path.join("wmt17-submitted-data", "txt", "references")
file_prefix = "newstest2017"
results = []
style_name = "bleu2vec_sep"
weights = (0.33, 0.34, 0.33)

lang = sys.argv[1]

tsart = int(time())

for translation_folder in os.listdir(hypothesis_folder):
    if lang in translation_folder:
        print(translation_folder)
        tref_file = "".join(translation_folder.split("-"))
        tref_file = "-".join([file_prefix, tref_file, "ref"])
        tref_file = ".".join([tref_file, translation_folder.split("-")[-1]])
        ref_file = os.path.join(reference_folder, tref_file)
        reference = []
        with open(ref_file) as f:
            for line in f:
                reference.append([word_tokenize(line.strip())])
        print("Read referece: " + translation_folder + ".")

        trans_hyp_folder = os.path.join(hypothesis_folder, translation_folder)
        hypothesis = []
        for file in os.listdir(trans_hyp_folder):
            tmp = []
            tmp.append(".".join(file.split(".")[1:-1]))
            with open(os.path.join(trans_hyp_folder, file)) as f:
                for line in f:
                    tmp.append(word_tokenize(line.strip()))
            hypothesis.append(tmp)
        print("Read hypothesis: " + translation_folder + ".")

        uni_model = KeyedVectors.load_word2vec_format(sys.argv[2], binary=True)
        bi_model = KeyedVectors.load_word2vec_format(sys.argv[3], binary=True)
        tri_model = KeyedVectors.load_word2vec_format(sys.argv[4], binary=True)
        print("Loaded models. Starting evaluating.")

        for hyp in hypothesis:
            candidates = hyp[1:]
            tss = int(time())
            score = corpus_bleu(reference, candidates, uni_model, bi_model, tri_model, weights=weights)
            tse = int(time())
            results.append([style_name, translation_folder, file_prefix, hyp[0],
                            round(score, 6), tss, tse, "non-ensemble", "https://github.com/TartuNLP/bleu2vec"])
            print(translation_folder + " " + str(round(tse-tss, 3)) + "s")


with open(".".join([style_name, lang, "sys", "score"]), "w+") as f:
    for res in results:
        line = "\t".join([str(_) for _ in res]) + "\n"
        f.write(line)


print(int(time()) - tsart)
