#!/usr/bin/env python3
"""
Dataset module for loading and preparing a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import transformers


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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.
        """
        tokenizer_pt = transformers.BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        # Get the original vocabulary
        vocab_pt = list(tokenizer_pt.vocab.keys())
        vocab_en = list(tokenizer_en.vocab.keys())

        # Define the desired vocabulary size
        vocab_size = 30000

        # Filter out less frequent tokens
        vocab_pt = vocab_pt[:vocab_size]
        vocab_en = vocab_en[:vocab_size]

        # Create a new tokenizer with the reduced vocabulary
        tokenizer_pt_reduced = transformers.BertTokenizer(tokenizer_pt.vocab, tokenizer_pt.special_tokens_map)
        tokenizer_en_reduced = transformers.BertTokenizer(tokenizer_en.vocab, tokenizer_en.special_tokens_map)

        # Update the tokenizers with the reduced vocabulary
        tokenizer_pt_reduced.add_tokens(vocab_pt)
        tokenizer_en_reduced.add_tokens(vocab_en)

        return tokenizer_pt_reduced, tokenizer_en_reduced
