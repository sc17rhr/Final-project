import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator


# --------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


def _find_n_grams(sentence):
    n_word = 1  ## length of n_grams
    w = sentence.split(' ')
    n_grams = []
    for i in range(len(w)):
        # if w[i]not in self.words[s]: self.words[s].append(w[i])
        for j in range(i + 1, np.min([i + 1 + n_word, len(w) + 1])):
            n_grams.append(' '.join(w[i:j]))
    return n_grams


def _get_words(sentences):
    n_grams = []
    n = _find_n_grams(sentences[id]['text'])
    for word in n:
        if word not in n_grams:
            n_grams.append(word)
    return n_grams


idf = {}
n_doc = 0.0
for scene in range(1, 1001):
    #print 'extracting feature from scene : ', scene
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_linguistic_features.p'
    sentences = _read_pickle(scene)
    for id in sentences:
        # if len(sentences.keys())>0:
        n_doc += 1
        words = _get_words(sentences)
        for word in words:
            if word not in idf:
                idf[word] = 1.0
            else:
                idf[word] += 1

sorted_x = sorted(idf.items(), key=operator.itemgetter(1))
print sorted_x

# print idf['lift']
# print idf['box']
print idf['row']
x = idf
FW = []
alpha_min = .2
alpha_max = np.log(n_doc / 23.0)
for word in idf:
    idf[word] = np.log(n_doc / idf[word])
    if idf[word] < alpha_min or idf[word] > alpha_max:
        FW.append(word)

pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
pickle.dump(FW, open(pkl_file, 'wb'))
