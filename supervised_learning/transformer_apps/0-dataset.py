#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class
    """
    def __init__(self):
        """Initializes the Dataset class by loading the training and validation datasets."""
        # Load the TED translation dataset from TensorFlow Datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # Tokenize the datasets to create the tokenizers for Portuguese and English
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained models from Hugging Face's Transformers.

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en).

        Returns:
            tokenizer_pt: Portuguese tokenizer (BERT-based).
            tokenizer_en: English tokenizer (BERT-based).
        """
        # Portuguese tokenizer using a pre-trained BERT model from Hugging Face
        tokenizer_pt = transformers.BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        
        # English tokenizer using a pre-trained BERT model from Hugging Face
        tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        # Return the tokenizers for both languages
        return tokenizer_pt, tokenizer_en
