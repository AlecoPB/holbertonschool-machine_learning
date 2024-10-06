#!/usr/bin/env python3
"""
Transformer Applications project
By Ced, performing NLP tasks
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset():
    """
    Class Dataset that loads and preps a dataset for machine translation
    then tokenizes the data for the transformer model
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        self.bacth_size = batch_size
        self.max_len = max_len

        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train)
        )

        #  map est une fonction tres utile dans TensorFlow qui permet
        #  d'appliquer une transformation à chaque élément d'un dataset
        # a la maniere de map python
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def mask_maxlen(pt, en):
            """
            filter out examples that have more than max_len tokens
            idk why its working
            """
            return tf.logical_and(
                tf.size(pt) <= self.max_len,
                tf.size(en) <= self.max_len
            )

        self.data_train = self.data_train.filter(mask_maxlen)

        # Caches the elements in this dataset.
        # The first time the dataset is iterated over, its elements
        # will be cached either in the specified file or in memory.
        # Subsequent iterations will use the cached data.

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)

        # padded batch est une variante de batch qui permet de
        # de definir une taille fixe pour les sequences
        self.data_train = self.data_train.padded_batch(self.bacth_size)
        # AUTOTUNE ajuste automatiquement la taille du buffer
        # prefetch: précharge les données pour accélérer le traitement
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # moins d'operations pour le data_valid
        self.data_valid = self.data_valid.filter(mask_maxlen)
        self.data_valid = self.data_valid.padded_batch(self.bacth_size)

    def tokenize_dataset(self, data):
        """
        Instance method
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def get_training_corpus_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        def get_training_corpus_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        #  How to tokenize a sentence
        #  sentences = "i love you madly"
        #  tokens = tokenizer_en.tokenize(sentences)
        #  print(tokens)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            get_training_corpus_pt(), vocab_size=8192
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            get_training_corpus_en(), vocab_size=8192
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Instance method
        """
        if tf.is_tensor(pt):
            pt = pt.numpy().decode('utf-8')
        if tf.is_tensor(en):
            en = en.numpy().decode('utf-8')

        # nouveaux indexs pour les tokens CLS et SEP
        nouveau_cls_id = 8192
        nouveau_sep_id = 8193

        # Exemple de tokenization manuelle avec vos propres IDs
        pt_tokens = ([nouveau_cls_id] +
                     self.tokenizer_pt.encode(pt, add_special_tokens=False) +
                     [nouveau_sep_id])
        en_tokens = ([nouveau_cls_id] +
                     self.tokenizer_en.encode(en, add_special_tokens=False) +
                     [nouveau_sep_id])
        # print("encode en", self.tokenizer_en.encode(en))
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        associé avec la fonction map de TensorFlow
        permet de tokeniser le data_train et data_valid
        """

        encoder = tf.py_function(func=self.encode, inp=[pt, en],
                                 Tout=[tf.int64, tf.int64])

        # attention, tf.py_function ne renvoie pas un tensor mais un tuple
        # print("encoder type", type(encoder))

        # [None] indique "un vecteur 1D de longueur variable"
        pt_tensor = tf.ensure_shape(encoder[0], [None])
        en_tensor = tf.ensure_shape(encoder[1], [None])
        # ensure_shape reconstruit le tensor avec la shape donnée
        # print("pt_tensor type", type(pt_tensor))
        return pt_tensor, en_tensor
