#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Bleu score
    """
    reference_ngrams = {}
    sentence_ngrams = Counter(sentence)
    for i in range(len(references)):
        ref_len = len(references[i])

        # Update the maximum reference length
        max_ref_len = max(max_ref_len, ref_len)
        reference_ngrams[i] = Counter(references[i])

    match_list = []
    for i in range(len(reference_ngrams)):
        match_list.append(sum((sentence_ngrams & reference_ngrams[i]).values()))

    match = max(match_list)
    possible_ngrams = sum(reference_ngrams[match_list.index(match)].values())


    precision = match / possible_ngrams if possible_ngrams > 0 else 0

    bp = 1 if possible_ngrams <= max_ref_len else math.exp(1 - max_ref_len / possible_ngrams)
    precision = bp * math.pow(precision, 1)

    return precision
