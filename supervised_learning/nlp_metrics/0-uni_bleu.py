#!/usr/bin/env python3
"""
Unigram BLEU score
"""
import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.

    Args:
    references (list of lists): A list of reference translations, where each reference translation is a list of words.
    sentence (list): The model proposed sentence as a list of words.

    Returns:
    float: The unigram BLEU score.
    """
    # Calculate the length of the sentence
    sentence_len = len(sentence)

    # Initialize the maximum reference length
    max_ref_len = 0

    # Initialize the count of matching unigrams
    matches = 0

    # Iterate over the references
    for reference in references:
        # Calculate the length of the current reference
        ref_len = len(reference)

        # Update the maximum reference length
        max_ref_len = max(max_ref_len, ref_len)

        # Calculate the count of matching unigrams for the current reference
        ref_counter = Counter(reference)
        sent_counter = Counter(sentence)
        matches += sum(min(ref_counter[word], sent_counter[word]) for word in ref_counter)

    # Calculate the precision
    precision = matches / sentence_len if sentence_len > 0 else 0

    # Calculate the brevity penalty
    bp = 1 if sentence_len <= max_ref_len else math.exp(1 - max_ref_len / sentence_len)

    # Calculate the unigram BLEU score
    bleu = bp * math.pow(precision, 1)

    return bleu