"""Dataset reader and process"""

import os
import random
import html
import string
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial


class Dataset():
  """Dataset class to read images and sentences from base (raw files)"""

  def __init__(self, source, name):
    self.source = source
    self.name = name
    self.dataset = None
    self.partitions = ['train', 'valid', 'char_test',  'word_test']

  def read_partitions(self):
    """Read images and sentences from dataset"""

    dataset = getattr(self, f"_{self.name}")()

    if not self.dataset:
      self.dataset = dict()

      for y in self.partitions:
        self.dataset[y] = {'dt': [], 'gt': []}

    for y in self.partitions:
      self.dataset[y]['dt'] += dataset[y]['dt']
      self.dataset[y]['gt'] += dataset[y]['gt']

  def preprocess_partitions(self, input_size, no_aug):
    """Preprocess images and sentences from partitions"""

    for y in self.partitions:
      arange = range(len(self.dataset[y]['gt']))

      for i in reversed(arange):
        text = pp.text_standardize(self.dataset[y]['gt'][i])

        # if not self.check_text(text):
        #   self.dataset[y]['gt'].pop(i)
        #   self.dataset[y]['dt'].pop(i)
        #   continue

        self.dataset[y]['gt'][i] = text.encode()

      results = []
      with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        print(f"Partition: {y}")
        for result in tqdm(pool.imap(partial(pp.preprocess, input_size=input_size, no_aug=no_aug), self.dataset[y]['dt']),
                   total=len(self.dataset[y]['dt'])):
          results.append(result)
        pool.close()
        pool.join()

      self.dataset[y]['dt'] = results

  def _dataset(self):
    """Dataset reader"""
    words = dict()
    dataset = dict()
    for i in self.partitions:
      dataset[i] = {"dt" : [], "gt" : []}

    
    for i in string.ascii_uppercase:
      for img in range(301):
        padded_img = str(img).zfill(5)
        img_path = os.path.join(self.source, i, f"hsf_0_{padded_img}.png")
        if (img < 10): #char_test
          dataset[self.partitions[2]]['gt'].append(i.lower())
          dataset[self.partitions[2]]['dt'].append(img_path)
        elif ((img - 10) / (301 - 10) * 100 <= 10): #valid
          dataset[self.partitions[1]]['gt'].append(i.lower())
          dataset[self.partitions[1]]['dt'].append(img_path)
        else : #train
          dataset[self.partitions[0]]['gt'].append(i.lower())
          dataset[self.partitions[0]]['dt'].append(img_path)

    word = open(os.path.join(self.source, "words.txt")).read().splitlines()
    for w in word: #word test
      split = w.split()
      img_path = os.path.join(self.source, 'Words', split[0])
      c = pp.seg_char(img_path)
      for i in c:
        dataset[self.partitions[3]]['gt'].append(split[1].lower()) 
        dataset[self.partitions[3]]['dt'].append(i)
    return dataset

  @staticmethod
  def check_text(text):
    """Make sure text has more characters instead of punctuation marks"""

    strip_punc = text.strip(string.punctuation).strip()
    no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

    if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
      return False

    punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

    return len(no_punc) > 2 and punc_percent <= 0.1
