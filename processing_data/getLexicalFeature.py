import numpy as np
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(1, os.getenv('PATH_ROOT'))
from domain_feature.lexical_feature import *


def getLexicalInputNN(url):
    lexical = LexicalURLFeature(url)
    length = lexical.get_length_to_feed_model()
    entropy = lexical.get_entropy()
    percent_number = lexical.get_percentage_digits()
    number_special_char = lexical.get_count_special_characters()
    lexical_array = np.array(
        [length, entropy, percent_number, number_special_char])
    type_url = one_hot_encode(lexical.get_type_url())
    inputNN_array = np.concatenate((lexical_array, type_url))
    return inputNN_array


def one_hot_encode(string):
    categories = ['bao_chi', 'chinh_tri',
                  'co_bac', 'khieu_dam', 'tpmt', 'con_lai']

    # Initialize an array of zeros with length equal to the number of categories
    encoded = np.zeros(len(categories), dtype=int)

    # Check if the string matches any category
    if string in categories:
        # Find the index of the matched category
        index = categories.index(string)
        # Set the corresponding element to 1
        encoded[index] = 1
    return encoded
