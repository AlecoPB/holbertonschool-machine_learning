#!/usr/bin/env python3
"""
n-gram BLEU score
"""
import math
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    :param sentence: list containing the model proposed sentence
    :param n: size of the n-gram to use for evaluation
    :return: n-gram BLEU score
    """
    # Function to extract n-grams from a sentence
    def extract_ngrams(sentence, n):
        return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

    # Extract n-grams from the sentence
    sentence_ngrams = extract_ngrams(sentence, n)
    sentence_ngram_counts = Counter(sentence_ngrams)

    # Extract n-grams from the references
    reference_ngram_counts = Counter()
    for ref in references:
        reference_ngrams = extract_ngrams(ref, n)
        reference_ngram_counts.update(reference_ngrams)

    # Count the number of matching n-grams
    match_count = 0
    for ngram in sentence_ngram_counts:
        match_count += min(sentence_ngram_counts[ngram], reference_ngram_counts[ngram])

    # Calculate precision
    precision = match_count / len(sentence_ngrams)

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(ref_lengths, key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len))
    if len(sentence) > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / len(sentence))

    # Calculate n-gram BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
