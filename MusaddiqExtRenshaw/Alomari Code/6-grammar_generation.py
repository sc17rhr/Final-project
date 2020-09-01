import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator


# --------------------------------------------------------------------------------------------------------#
def _read_sentences(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


def _read_tfidf_words():
    pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    tfidf = pickle.load(data)
    return tfidf


def _read_vf(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    vf, tree = pickle.load(data)
    return vf, tree


def _get_grammar_trees(S, tree):
    grammar_trees = {}
    count = 0
    try:
        if len(tree['py'].keys()) == 3:
            for i1 in range(1, len(S) - 1):
                for i2 in range(1, len(S) - i1):
                    grammar_trees[count] = [S[0:i1], S[i1:i2 + i1], S[i2 + i1:]]
                    print [S[0:i1], S[i1:i2 + i1], S[i2 + i1:]]
                    count += 1
        if len(tree['py'].keys()) == 2:
            for i1 in range(1, len(S)):
                grammar_trees[count] = [S[0:i1], S[i1:]]
                print [S[0:i1], S[i1:]]
                count += 1
    except KeyError:
        print("missing key")

    return grammar_trees


# pkl_file = '/home/'+getpass.getuser()+'/Datasets/Dukes_modified/learning/tags.p' data = open(pkl_file,
# 'rb') hypotheses_tags, VF_dict, LF_dict = pickle.load(data) this is why I can't have nice things total = 1 for word
# in hypotheses_tags: total*=len(hypotheses_tags[word].keys())+1 print '>>>>>>>>>>>>>>>>>>',total In this dataset,
# we have 114 unique words, were each word has between 2 and 4 potential visual tags. If we compute the combinations
# of these words tags it's is 1.7 quattuordecillion 1.7*10^45, which gives the reader an idea of how massive the
# search space is, and that it can't be the case where we keep track of all combinations. Therefore we take a
# different approach, were we do the learning incrementally. The system analyse each sentence separately, and record

tfidf_words = _read_tfidf_words()

for scene in range(1, 1001):
    print 'generating grammar from scene : ', scene
    VF, Tree = _read_vf(scene)
    sentences = _read_sentences(scene)
    grammar_trees = {}
    for id in sentences:
        S = sentences[id]['text'].split(' ')
        for word in tfidf_words:
            S = filter(lambda a: a != word, S)
        grammar_trees[id] = _get_grammar_trees(S, Tree)
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_grammar.p'
    pickle.dump(grammar_trees, open(pkl_file, 'wb'))
