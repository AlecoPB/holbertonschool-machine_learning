#!/usr/bin/env python3
"""
Answer questions from multiple references
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(corpus_path):
    """
    Add multiple references to 2-qa.py
    """
    # Load model from tf_hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    exit = ["exit", "quit", "goodbye", "bye"]

    while True:
        # Initialize a list to store possible answers
        PAnswer = []

        # Read the question and ensure its not empty
        Q = input("Q: ")
        if Q == "":
            return None

        # Check exit cases
        if Q.lower() not in exit:
                # Tokenize the input text
                inputs = tokenizer.encode_plus(Q,
                                               reference,
                                               return_tensors='tf',
                                               return_attention_mask=True)

                input_ids = inputs['input_ids']
                token_type_ids = inputs['token_type_ids']
                attention_mask = inputs['attention_mask']

                # Pass inputs to the model
                outputs = model([input_ids, attention_mask, token_type_ids])

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
