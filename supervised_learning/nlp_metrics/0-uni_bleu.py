#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter
import string


def uni_bleu(references, sentence):
    """
    Bleu score
    """
    reference_ngrams = {}
    sentence_ngrams = Counter(sentence)
    for i in range(len(references)):
        reference_ngrams[i] = Counter(references[i])

    match_list = []
    for i in range(len(reference_ngrams)):
        match_list.append(sum((sentence_ngrams & reference_ngrams[i]).values()))

    match = max(match_list)
    possible_ngrams = sum(reference_ngrams[match_list.index(match)].values())

    if 
    precision = match / possible_ngrams if possible_ngrams > 0 else 0
    
    return precision
