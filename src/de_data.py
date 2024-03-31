import pathlib
import os
import re
import string
from unicodedata import normalize
import pandas as pd
from datetime import datetime

import time
import random
random.seed(42)
from random import randint
from pathlib import Path
import pathlib
import scipy
import numpy as np
np.random.seed(123)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from .autonotebook import tqdm as notebook_tqdm
# https://github.com/cheeyeo/neural-machine-translation/blob/master/data/deu.txt

### dataset mentioned in the Cristina book - https://github.com/Rishav09/Neural-Machine-Translation-System/blob/master/english-german-both.pkl
#### This also has the same file - https://github.com/Rishav09/Neural-Machine-Translation-System/blob/master/deu.txt


class DataGen:
    def __init__(self, fpath="D:\\MyDev\\attention\deu.txt"):
        self.fpath = fpath
        self.doc = self.load_doc()


    def load_doc(self, fpath=None):
      """
      Opens up a file with utf8 encoding
      """
      if not fpath:
          fpath = self.fpath
      with open(fpath, mode='rt', encoding='utf-8') as f:
        text = f.read()
        return text

    def clean_pairs(self, lines):
      cleaned = list()
      re_punc = re.compile('[%s]' % re.escape(string.punctuation))
      re_print = re.compile('[^%s]' % re.escape(string.printable))
      for pair in lines:
        clean_pair = list()
        for line in pair:
          line = normalize('NFD', line).encode('ascii', 'ignore')
          line = line.decode('UTF-8')
          line = line.split()
          line = [w.lower() for w in line]
          line = [re_punc.sub('', w) for w in line]
          line = [re_print.sub('', w) for w in line]
          line = [w for w in line if w.isalpha()]
          clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
      return np.array(cleaned)

    def clean_lines(self, lines):
      cleaned = list()
      re_print = re.compile('[^%s]' % re.escape(string.printable))
      # translation table for removing punctuation
      table = str.maketrans('', '', string.punctuation)
      for line in lines:
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = line.split()
        line = [w.lower() for w in line]
        # Remove punctuation
        line = [w.translate(table) for w in line]

        # Remove single characters left from aprostrophe removal etc...
        line = [w for w in line if len(w) > 1]

        # Remove non-printable characters
        line = [re_print.sub('', w) for w in line]
        line = [w for w in line if w.isalpha()]
        cleaned.append(' '.join(line))
      return cleaned

    def save_clean_lines(self, sentences, filename):
      pickle.dump(sentences, open(filename, 'wb'))
      print('[INFO] Saved clean lines: {}'.format(filename))


    def save_tokenizer(self, tokenizer, filename):
      pickle.dump(tokenizer, open(filename, 'wb'))
      print('[INFO] Saved tokenizer: {}'.format(filename))


    def load_tokenizer(self, filename):
      with open(filename, 'rb') as f:
        return pickle.load(f)


    def to_pairs(self, doc):
      lines = doc.strip().split('\n')
      pairs = [line.split('\t') for line in lines]
      return pairs


    def to_sentences(self, doc):
      """
      Turns a binary doc file object into a list
      """
      return doc.strip().split('\n')


    def sentence_length(self, sentences):
      """
      Return max lengths for list of sentences
      """
      lengths = [len(s.split()) for s in sentences]
      return max(lengths)

    def load_saved_lines(self, filename):
      with open(filename, 'rb') as f:
        return pickle.load(f)

    def load_clean_lines(self, filename):
      doc = load_saved_lines(filename)
      lines = list()
      for line in doc:
        new_line = 'startseq ' + line + ' endseq'
        lines.append(new_line)
      return lines

    def add_delimiters_to_lines(self, lines):
      new_lines = list()
      for line in lines:
        new_line = 'sos ' + line + ' eos'
        new_lines.append(new_line)
      return new_lines

    def to_vocab(self, lines):
      vocab = Counter()
      for line in lines:
        tokens = line.split()
        vocab.update(tokens)
      return vocab

    def trim_vocab(self, vocab, min_occurences):
      tokens = [k for k, c in vocab.items() if c >= min_occurences]
      return set(tokens)

    def update_dataset(self, lines, vocab):
      new_lines = list()
      for line in lines:
        new_tokens = list()

        for token in line.split():
          if token in vocab:
            new_tokens.append(token)
          else:
            new_tokens.append('unk')
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)

      return new_lines

    def get_cleaned_delimited_data_as_array(self, n_sentences=40000, from_end=False, print_samples=10):
        pairs = self.to_pairs(self.doc)
        print(len(pairs))
        print(pairs[1])
        cleaned = self.clean_pairs(pairs)
        print('[INFO] Total dataset size: ', cleaned.shape)
        for i in range(print_samples):
          print('[{}] => [{}]'.format(cleaned[i, 0], cleaned[i, 1]))
        if from_end:
            dataset = cleaned[-n_sentences:, :]
        else:
            dataset = cleaned[:n_sentences, :]
        dataset[:, 1] = self.add_delimiters_to_lines(dataset[:, 1])
        np.random.shuffle(dataset)
        print('[INFO] Total dataset size: {:d}'.format(len(dataset)))
        print(dataset[-1])
        print(dataset.shape)
        lang1 = dataset[:, 0]
        lang2 = dataset[:, 1]
        print(lang2.shape)
        print(lang1[-1])
        return dataset, lang1, lang2


    def gen_data(self, lang1, lang2):
      # lng2_with_sos_eos = ['<start> ' + s + ' <end>' for s in lang2]
      lng2_with_sos_eos = lang2
      lang1_tokenizer = Tokenizer()
      lang1_tokenizer.fit_on_texts(lang1)
      lang2_tokenizer = Tokenizer()
      lang2_tokenizer.fit_on_texts(lng2_with_sos_eos)
      lang1_sequences = lang1_tokenizer.texts_to_sequences(lang1)
      lang2_sequences = lang2_tokenizer.texts_to_sequences(lng2_with_sos_eos)      
      lang1_padded = pad_sequences(lang1_sequences, padding='post')
      ### remove eos for the input data before padding 
      lang2_seq_without_eos = [sequence[:-1] for sequence in lang2_sequences]
      decoder_input_data = pad_sequences(lang2_seq_without_eos, padding='post') 
      ### remove sos from the label   
      lang2_padded = pad_sequences(lang2_sequences, padding='post') 
      decoder_output_data = lang2_padded[:, 1:]  # All but the first token
      # Shuffle the data
      indices = np.arange(len(lang1_padded))
      np.random.shuffle(indices)
      lang1_padded = lang1_padded[indices]
      decoder_input_data = decoder_input_data[indices]
      decoder_output_data = decoder_output_data[indices]
      # Example output
      print('encoder input:\n', lang1_padded[:1])  # Encoded and padded English sentences
      print('num for sos=', lang2_tokenizer.texts_to_sequences([['sos']])[0][0])
      print('num for eos=', lang2_tokenizer.texts_to_sequences([['eos']])[0][0])
      print('decoder input:\n',decoder_input_data[:1])  # Encoded and padded target sentences (as decoder input)
      print('decoder label:\n', decoder_output_data[:1])
      # Vocabulary sizes
      vocab_size_lang1 = len(lang1_tokenizer.word_index) + 1
      vocab_size_lang2 = len(lang2_tokenizer.word_index) + 1
      return lang1_padded, decoder_input_data, decoder_output_data, lang1_tokenizer, lang2_tokenizer
      # return lang1_one_hot, decoder_input_data_one_hot, decoder_target_data_one_hot, lang1_tokenizer, lang2_tokenizer
    def get_source_target_data_n_tokenizer(self, n_sentences=40000, from_end=False):
        dataset, lang1, lang2 = self.get_cleaned_delimited_data_as_array(
                                n_sentences=n_sentences, from_end=from_end)
        X1, X2, y, lang1_tokenizer, lang2_tokenizer = self.gen_data(lang1, lang2)
        en_vocab_size = len(lang1_tokenizer.word_index) + 1
        de_vocab_size = len(lang2_tokenizer.word_index) + 1
        en_seq_len = X1.shape[1]
        de_seq_len = X2.shape[1]
        print(f'en_seq_len: {en_seq_len}, de_seq_len: {de_seq_len}, en_vocab_size: {en_vocab_size}, de_vocab_size: {de_vocab_size}')
        return X1, X2, y, lang1_tokenizer, lang2_tokenizer, en_vocab_size, de_vocab_size, en_seq_len, de_seq_len

