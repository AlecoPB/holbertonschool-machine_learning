#!/usr/bin/env python3
"""
Answer questions from multiple references
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity


def question_answer(corpus_path):
    """
    Add multiple references to 2-qa.py
    """
    def read_documents(directory_path, extension='*.md'):
        """
        Read documents from a directory
        """
        file_paths = glob.glob(os.path.join(directory_path, extension))
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                yield f.read()

    # Load model from tf_hub
    qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the Universal Sentence Encoder from TensorFlow Hub
    encoder_model =\
        hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    exit = ["exit", "quit", "goodbye", "bye"]

    while True:
        # Read the question and ensure its not empty
        Q = input("Q: ")

        if Q == "":
            return None

        # Check exit cases
        if Q.lower() not in exit:

            # Load the corpus of reference documents
            documents = list(read_documents(corpus_path))

            # Encode the reference documents into vectors in one go
            doc_embeddings = encoder_model(documents).numpy()

            # Encode the search sentence into a vector
            query_embedding = encoder_model([Q]).numpy()

            # Compute cosine similarities between sentence and documents
            cosine_similarities = cosine_similarity(query_embedding,
                                                    doc_embeddings)[0]

            # Find the index of the most similar document
            most_similar_idx = np.argmax(cosine_similarities)

            # Return the most similar document
            most_sim_doc = documents[most_similar_idx]

            # Tokenize the input text
            inputs = tokenizer.encode_plus(Q,
                                           most_sim_doc,
                                           return_tensors='tf',
                                           return_attention_mask=True)

            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention_mask = inputs['attention_mask']

            # Pass inputs to the model
            outputs = qa_model([input_ids, attention_mask, token_type_ids])

            # Extract start and end logits
            start_logits, end_logits = outputs[0], outputs[1]

            # Get the input sequence length
            sequence_length = inputs["input_ids"].shape[1]

            # Get the most likely start and end token positions
            start_idx = tf.math.argmax(
                start_logits[0, 1:sequence_length-1]) + 1
            end_idx = tf.math.argmax(
                end_logits[0, 1:sequence_length-1]) + 1

            # Error handling
            if start_idx > end_idx:
                return None

            # Convert token inx back to the text
            tokens = input_ids[0].numpy()[start_idx:end_idx+1]
            answer = tokenizer.decode(tokens,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)

            # More error handling
            if not answer.strip():
                return None

            print(f"A: {answer}")

        else:
            print("A: Goodbye")
            break
