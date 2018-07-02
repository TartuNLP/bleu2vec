# -*- coding: utf-8 -*-
# Natural Language Toolkit: BLEU Score
#
# Copyright (C) 2001-2017 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# Contributors: Dmitrijs Milajevs, Liling Tan
# URL: <http://nltk.org/>
# Real url: http://www.nltk.org/_modules/nltk/translate/bleu_score.html
# For license information, see LICENSE.TXT

"""BLEU score implementation."""
from __future__ import division

import math
import fractions
import warnings
from collections import Counter
from itertools import chain
import numpy as np
import pandas as pd

from nltk.util import ngrams

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

from gensim.models import KeyedVectors


def sentence_bleu(references, hypothesis, uni_model, bi_model, tri_model, weights=(0.25, 0.25, 0.25, 0.25)):
    return corpus_bleu([references], [hypothesis], uni_model, bi_model, tri_model, weights)

# This method is modified by Andre Tättar.
def corpus_bleu(list_of_references, hypotheses, uni_model, bi_model, tri_model, weights=(0.25, 0.25, 0.25, 0.25)):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"
    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, uni_model, bi_model, tri_model, i)
            p_numerators[i] += p_i[0]
            p_denominators[i] += p_i[1]

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = []
    for i, _ in enumerate(weights, start=1):
        if p_denominators[i] == 0:
            p_n.append(0)
        else:
            p_n.append(p_numerators[i]/p_denominators[i])

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references,
                             hypothesis=hypothesis, hyp_len=hyp_len)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    return bp * math.exp(math.fsum(s))


def modified_precision(references, hypothesis, uni_model, bi_model, tri_model, n):
    # This file is made and modified by Andre Tättar
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()

    eps = 0.001

    if n < 4:
        if n == 2:
            hypothesis = ["_".join(p) for p in zip(hypothesis[:-1], hypothesis[1:])]
            references = [["_".join(p) for p in zip(references[0][:-1], references[0][1:])]]
        elif n == 3:
            hypothesis = ["_".join(p) for p in zip(hypothesis[:-2], hypothesis[1:-1], hypothesis[2:])]
            references = [["_".join(p) for p in zip(references[0][:-2], references[0][1:-1], references[0][2:])]]
        cntr_cand = Counter(hypothesis)
        cntr_hyp = Counter(references[0])
        min_ref_hyp = cntr_hyp - cntr_cand
        min_hyp_ref = cntr_cand - cntr_hyp
        min_all = cntr_hyp & cntr_cand
        up = sum(min_all.values())
        unused_reference = list(chain(*[v * [k] for k, v in min_hyp_ref.items()]))
        unused_hypotheses = list(chain(*[v * [k] for k, v in min_ref_hyp.items()]))

        M = np.zeros(shape=(len(unused_reference), len(unused_hypotheses)), dtype=float)

        for ix, w1 in enumerate(unused_reference):
            for jx, w2 in enumerate(unused_hypotheses):
                try:
                    if n == 1:
                        M[ix][jx] = uni_model.similarity(w1, w2)
                    elif n == 2:
                        M[ix][jx] = bi_model.similarity(w1, w2)
                    elif n == 3:
                        M[ix][jx] = tri_model.similarity(w1, w2)
                except KeyError:
                    M[ix][jx] = eps
        df = pd.DataFrame(data=M, index=unused_reference, columns=unused_hypotheses, dtype=float)

        while df.shape[0] > 0 and df.shape[1] > 0:
            w1, w2, vmax = None, None, -1
            for ix, ival in df.iterrows():
                for jx, jval in ival.iteritems():
                    if jval > vmax:
                        w1, w2, vmax = ix, jx, jval
            df = df.drop(w1)
            df = df.drop(w2, axis=1)
            up += vmax

        return up, len(hypothesis)
    else:
        # This part is actually not required at all, since we only use up to 3 words
        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0),
                                        reference_counts[ngram])

        # Assigns the intersection between hypothesis and references' counts.
        clipped_counts = {ngram: min(count, max_counts[ngram])
                          for ngram, count in counts.items()}

        numerator = sum(clipped_counts.values())
        # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
        # Usually this happens when the ngram order is > len(reference).
        denominator = max(1, sum(counts.values()))

    return numerator, denominator



def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: The length of the hypothesis.
    :type hypothesis: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len



def brevity_penalty(closest_ref_len, hyp_len):
    """
    Calculate brevity penalty.

    As the modified n-gram precision still has the problem from the short
    length sentence, brevity penalty is used to modify the overall BLEU
    score according to length.

    :param hyp_len: The length of the hypothesis for a single sentence OR the
    sum of all the hypotheses' lengths for a corpus
    :type hyp_len: int
    :param closest_ref_len: The length of the closest reference for a single
    hypothesis OR the sum of all the closest references for every hypotheses.
    :type closest_reference_len: int
    :return: BLEU's brevity penalty.
    :rtype: float
    """
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)



class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """
    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        This will initialize the parameters required for the various smoothing
        techniques, the default values are set to the numbers used in the
        experiments from Chen and Cherry (2014).

        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
    
    # This is modified to suit the new data - Andre Tättar
    def method0(self, p_n, *args, **kwargs):
        """ No smoothing. """
        p_n_new = []
        for i, p_i in enumerate(p_n):
            if p_i > 0:
                p_n_new.append(p_i)
            else:
                _msg = str("\nCorpus/Sentence contains 0 counts of {}-gram overlaps.\n"
                           "BLEU scores might be undesirable; "
                           "use SmoothingFunction().").format(i+1)
                warnings.warn(_msg)
                # If this order of n-gram returns 0 counts, the higher order
                # n-gram would also return 0, thus breaking the loop here.
                break
        return p_n_new
