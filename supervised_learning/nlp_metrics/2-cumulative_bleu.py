#!/usr/bin/env python3
"""
Cummulative BLEU score
"""
import math
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    :param sentence: list containing the model proposed sentence
    :param n: size of the largest n-gram to use for evaluation
    :return: cumulative n-gram BLEU score
    """
    def extract_ngrams(sentence, n):
        return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

    def ngram_bleu(references, sentence, n):
        sentence_ngrams = extract_ngrams(sentence, n)
        sentence_ngram_counts = Counter(sentence_ngrams)

        reference_ngram_counts = Counter()
        for ref in references:
            reference_ngrams = extract_ngrams(ref, n)
            reference_ngram_counts.update(reference_ngrams)

        match_count = 0
        for ngram in sentence_ngram_counts:
            match_count += min(sentence_ngram_counts[ngram],
                               reference_ngram_counts[ngram])

        precision = match_count / len(sentence_ngrams)

        ref_lengths = [len(ref) for ref in references]
        closest_ref_length = min(ref_lengths,
                                 key=lambda ref_len: (abs(ref_len - len(sentence)),
                                                      ref_len))
        if len(sentence) > closest_ref_length:
            brevity_penalty = 1
        else:
            brevity_penalty = math.exp(1 - closest_ref_length / len(sentence))

        bleu_score = brevity_penalty * precision

        return bleu_score

    cumulative_score = 0
    for i in range(1, n+1):
        cumulative_score += ngram_bleu(references, sentence, i)

    cumulative_score /= n

    return cumulative_score
