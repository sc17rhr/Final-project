import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator
import itertools
from copy import deepcopy
from nltk.tree import *
from nltk.tree import ParentedTree
from learn_pcfg import *


# ---------------------------------------------------------------------------#
def _read_stop_wrods():
    pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    stop = pickle.load(data)
    return stop


# ---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '../Datasets/Dataset1/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]


def _read_sentences(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


# ---------------------------------------------------------------------------#
def _read_vf(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    vf, tree = pickle.load(data)
    return vf, tree


# ---------------------------------------------------------------------------#
def _read_semantic_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_semantic_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


# ---------------------------------------------------------------------------#
def _read_layout(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_layout.p'
    data = open(pkl_file, 'rb')
    layout = pickle.load(data)
    return layout


# ---------------------------------------------------------------------------#
def _read_grammar_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


def _read_passed_tags():
    pkl_file = '../Datasets/Dataset1/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching, Matching_VF, passed_scenes, passed_sentences = pickle.load(data)
    # print Matching,Matching_VF,passed_scenes,passed_ids
    return [Matching, Matching_VF, passed_scenes, passed_sentences]


def _is_yuk(sentence):
    yuk = 0
    yuk_words = [' and', ' closest', ' near', ' far', ' nearest', ' edge', ' corner', ' side', ' leftmost', 'rightmost',
                 '1', '2', '3', '4', '5', 'two', 'one', 'three', 'five', 'four', 'single', 'position', 'grid', 'right',
                 'lift', 'box', 'left', 'location', 'exactly', 'lower', 'that', 'next', 'lowest', 'opposite', ' it',
                 ' to ']
    for word in yuk_words:
        if word in sentence:
            yuk = 1
            break
    return yuk


def _change_tree(RCL, text):
    words = text.split(' ')
    # print words
    tokens = [str(i) for i in range(50)]
    tree = Tree.fromstring(RCL)
    # print tree
    to_be_removed = []
    to_be_changed = {}
    for p in (tree.treepositions()):
        p = list(p)
        if tree[p] in tokens:
            change = p[:-1]
            if change[-1] != 0:
                if p[:-1] not in to_be_removed:
                    to_be_removed.append(p[:-1])
                change[-1] -= 1
            change = tuple(change)
            if change not in to_be_changed:
                to_be_changed[change] = [p]
            else:
                to_be_changed[change].append(p)
    for item in to_be_changed:
        if len(to_be_changed[item]) == 1:
            token = to_be_changed[item][0]
            s = words[int(tree[token]) - 1]
        else:
            token0 = to_be_changed[item][0]
            token1 = to_be_changed[item][1]
            t0 = int(tree[token0]) - 1
            t1 = int(tree[token1]) - 1
            # s = ''
            for count, i in enumerate(range(t0, t1 + 1)):
                if not count:
                    s = words[i]
                elif words[i] != 'the':
                    s += ' ' + words[i]
            # print to_be_changed[item]
            # for count,token in enumerate(range(to_be_changed[item]):
            # if count == 0:
            #     s+=words[int(tree[token])-1]
            # else:
            #     s+=' '+words[int(tree[token])-1]
        # print 'change >>',tree[item],'>>',s
        tree[item] = s
    # print '>>>>>>>>>>>>',to_be_changed
    for item in to_be_removed:
        # print item
        # print 'remove >>',tree[item]
        del tree[item]
    # print tree
    # print '--------------------'
    return tree


# ---------------------------------------------------------------------------#
Matching, Matching_VF, passed_scenes, passed_sentences = _read_passed_tags()
sentences_to_test = {}
bad_trees = [14588, 23958, 10646, 25409, 25625, 14427, 23982, 16360, 22369, 23928, 16792, 18058, 25013, 9323, 26997,
             25565, 14412, 16159, 26955, 4028, 9207, 18582, 25100, 25058, 23428, 23985, 12027, 25653, 14624, 14423,
             25682, 12515, 13775, 4073, 10186, 13046, 25622, 26283, 23217, 12453, 23955, 23970, 23756, 23898, 14789,
             25477, 9418, 2541, 23738, 24170]
trees = []
for scene in range(1, 1001):  # 920!!!
    print '###', scene
    sentences = _read_sentences(scene)
    for id in sentences:
        if id not in bad_trees:
            if not _is_yuk(sentences[id]['text']):
                S = sentences[id]['text']
                sentences_to_test[id] = sentences[id]
                # if id in passed_sentences:
                print '>>>>', sentences[id]['RCL']
                tree = _change_tree(sentences[id]['RCL'], sentences[id]['text'])
                print tree
                print '----'
                trees.append(tree)
                pkl_file = '../Datasets/Dataset1/RCL-trees/' + str(id) + '_tree.p'
                pickle.dump(tree, open(pkl_file, 'wb'))

grammar = learn_trees(trees)
print '##############################################################'
print grammar
print '##############################################################'
S = S.replace('the ', '')
# print prob_parse(grammar,S)


pkl_file = '../Datasets/Dataset1/experiment/sentences.p'
pickle.dump(sentences_to_test, open(pkl_file, 'wb'))

file1 = '../Datasets/Dataset1/experiment/sentences.txt'
F = open(file1, 'w')
for id in sentences_to_test:
    F.write(sentences_to_test[id]['text'] + '\n')
F.close()

Data = read_data()
tokens = [str(i) for i in range(50)]
f = open('../Datasets/Dataset1/experiment/tags.txt', 'w')
for id in sentences_to_test:
    tree = ParentedTree.fromstring(sentences_to_test[id]['RCL'])
    sentence = []
    for p in tree.pos():
        if str(p[0]) not in tokens:
            # print (p[0],p[1].split(':')[0])
            sentence.append(p[1].split(':')[0])
    S = ' '.join(sentence)
    f.write(S + '\n')
f.close()
