import math
import re
from urllib.parse import urlparse
from socket import gethostbyname
import pandas as pd


def levenshtein_distance_string(str1, str2):
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    m = len(str1)
    n = len(str2)

    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(curr_row[j - 1],
                                      prev_row[j], prev_row[j - 1])
        prev_row = curr_row.copy()
    return curr_row[n]


def get_substrings_of_length(s, length):
    return [s[start:start+length] for start in range(len(s) - length + 1)]


class LexicalURLFeature:
    def __init__(self, url):
        self.description = 'blah'
        self.url = url
        self.domain = self.extract_domain()
        self.url_parse = urlparse(self.url)
        self.dict_word = self.load_dict_word()
        
    def load_dict_word(self):
        file_path = "main/featureURL/dict_check_type.xlsx"
        try:
            df = pd.read_excel(file_path)
            return dict(zip(df['word'].tolist(), df['type'].tolist()))
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return {}

    def extract_domain(self):
        return self.url.split('.')[0]

    def get_entropy(self):
        probs = [self.domain.count(c) / len(self.domain)
                 for c in set(self.domain)]
        entropy = -sum(p * math.log(p) / math.log(2.0) for p in probs)
        return round(entropy, 3)

    def get_length_to_feed_model(self):
        return len(self.domain)

    def get_length_to_display(self):
        return len(self.url)

    def get_percentage_digits(self):
        num_digits = sum(1 for char in self.domain if char.isdigit())
        total_chars = len(self.domain)
        if total_chars == 0:
            return 0
        return round((num_digits / total_chars) * 100, 3)

    def get_count_special_characters(self):
        regex = r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|<,/<>?]'
        special_characters = re.findall(regex, self.domain)
        return len(special_characters)

    def get_type_url(self):
        # if it have the blank like: unews vn -> get unews
        if " " in self.domain:
            self.domain = self.domain.split(" ")[0]
        # check the domain have the keyword in dict
        for word in self.dict_word.keys():
            if str(word) in self.domain:
                print(f"1 {self.domain} have {str(word)} and type {self.dict_word.get(str(word))}")
                return self.dict_word.get(str(word), "con_lai")
        # check the domain have the keyword with leven distances
        for word in self.dict_word.keys():
            for sub_word in get_substrings_of_length(self.domain, len(str(word))):
                if levenshtein_distance_string(sub_word, word) == 1:
                    print(f"2 {self.domain} have {str(sub_word)} and type {self.dict_word.get(str(word))}")
                    return self.dict_word.get(str(word), "con_lai")
        print(f"3 {self.domain} is type con_lai")
        return "con_lai"
