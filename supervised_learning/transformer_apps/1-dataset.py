#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers
import numpy as np


class Dataset:
    """
    Dataset class to load and prep a dataset for machine translation.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en =\
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Generate sub-word tokenizers
        """

        tokenizer_pt = transformers.AutoTokenizer.\
            from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.\
            from_pretrained('bert-base-uncased')

        def get_training_corpus_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        def get_training_corpus_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.\
            train_new_from_iterator(get_training_corpus_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.\
            train_new_from_iterator(get_training_corpus_en(), vocab_size=2**13)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Translates a sentence into tokens, including tokens for
        the start and end of the sentence.

        Args:
            pt: `tf.Tensor` containing the Portuguese sentence.
            en: `tf.Tensor` containing the corresponding English sentence.

        Returns: pt_tokens, en_tokens:
        """
        # Convert tf.Tensor to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Retrieve the vocabulary size from the tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Tokenize the sentences without special tokens
        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        # Add start and end tokens for the sentences
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens
