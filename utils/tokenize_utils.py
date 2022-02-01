import transformers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch 
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

def get_input_ids(train_sentences):
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []

  # For every sentence...
  for sent in train_sentences:
      # `encode` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      encoded_sent = tokenizer.encode(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )
      
      # Add the encoded sentence to the list.
      input_ids.append(encoded_sent)
  return input_ids

# We'll borrow the `pad_sequences` utility function to do this.
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 128

def tf_pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post"):
  print('\nPadding/truncating all sentences to %d values...' % maxlen)
  print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
  input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", 
                          value=0, truncating="post", padding="post")
  return input_ids

def get_maxlen(input_ids):
  x = max([len(sen) for sen in input_ids])
  return x

def get_masks(input_ids):
  # Create attention masks
  train_masks = []
  # For each sentence...
  for sent in input_ids:
      # Create the attention mask.
      #   - If a token ID is 0, then it's padding, set the mask to 0.
      #   - If a token ID is > 0, then it's a real token, set the mask to 1.
      att_mask = [int(token_id > 0) for token_id in sent]
      
      # Store the attention mask for this sentence.
      train_masks.append(att_mask)
  return train_masks