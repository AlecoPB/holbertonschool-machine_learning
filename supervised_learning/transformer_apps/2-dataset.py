#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers
import numpy as np
import tensorflow as tf


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
        Encodes a translation into tokens.

        Parameters:
        - pt: tf.Tensor, the Portuguese sentence.
        - en: tf.Tensor, the English sentence.

        Returns:
        - pt_tokens: np.ndarray containing the Portuguese tokens.
        - en_tokens: np.ndarray containing the English tokens.
        """
        # Convert tensors to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Tokenize the sentences
        pt_tokens = self.tokenizer_pt.encode(pt_sentence)
        en_tokens = self.tokenizer_en.encode(en_sentence)

        # Get vocab size for adding special tokens
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        # Add start and end tokens
        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        # Convert to numpy arrays
        pt_tokens = np.array(pt_tokens, dtype=np.int32)
        en_tokens = np.array(en_tokens, dtype=np.int32)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.

        Parameters:
        - pt: tf.Tensor containing the Portuguese sentence.
        - en: tf.Tensor containing the English sentence.

        Returns:
        - pt_tokens: tf.Tensor containing the tokenized Portuguese sentence.
        - en_tokens: tf.Tensor containing the tokenized English sentence.
        """
        # Apply encode using tf.py_function to make it TensorFlow-friendly
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int32, tf.int32])

        # Set shape of the tensors after tokenization
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
