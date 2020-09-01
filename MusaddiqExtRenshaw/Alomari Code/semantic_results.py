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

#---------------------------------------------------------------------------#
def _read_stop_wrods():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    stop = pickle.load(data)
    return stop

#---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]

def _read_sentences(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences

def _read_passed_tags():
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching,Matching_VF,passed_scenes,passed_sentences = pickle.load(data)
    # print Matching,Matching_VF,passed_scenes,passed_ids
    return [Matching,Matching_VF,passed_scenes,passed_sentences]

#---------------------------------------------------------------------------#
hypotheses_tags, VF_dict, LF_dict = _read_tags()
Matching,Matching_VF,passed_scenes,passed_sentences = _read_passed_tags()
Matching['the'] = 0
Keys =  Matching.keys()
not_ok = []
for tag in LF_dict.keys():
    if tag not in Keys:
        not_ok.append(tag)
print passed_sentences
pass_count = 0
fail_count = 0
fail = {}
for scene in range(1,1001):
    sent = _read_sentences(scene)
    for key in sent.keys():
        # if key not in passed_sentences: continue
        if 'and' in sent[key]['text']: continue
        if 'nearest' in sent[key]['text']: continue
        if 'closest' in sent[key]['text']: continue
        if 'one' in sent[key]['text']: continue
        if 'between' in sent[key]['text']: continue
        if 'corner' in sent[key]['text']: continue
        for word in sent[key]['text'].split(' '):
            if word in Keys:
                pass_count+=1
            else:
                if word not in fail:
                    fail[word]=0
                fail[word]+=1
                fail_count+=1

print 'pass =', pass_count
print 'fail =', fail_count

# print fail
print sorted(fail.items(), key=operator.itemgetter(1))
