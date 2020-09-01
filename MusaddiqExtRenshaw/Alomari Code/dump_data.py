import numpy as np
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
#--------------------------------------------------------------------------------------------------------#
def _fix_sentences(S):
    # S = self.Data['commands'][self.scene]
    # for i in S:
        S = S.replace("    ", " ")
        S = S.replace("   ", " ")
        S = S.replace("  ", " ")
        S = S.replace("  ", " ")
        S = S.replace("  ", " ")
        S = S.replace("  ", " ")
        S = S.replace(".", "")
        S = S.replace(",", "")
        S = S.replace("'", "")
        S = S.replace("-", " ")
        S = S.replace("/", " ")
        S = S.replace("!", "")
        S = S.replace("(", "")
        S = S.replace(")", "")
        S = S.replace("?", "")
        A = S.split(' ')
        while '' in A:         A.remove('')
        S = ' '.join(A)
        return S.lower()

def traverse(t):
    try:
        t.label()
    except AttributeError:
        return
    else:

        if t.height() == 2:   #child nodes
            print t.parent()
            return

        for child in t:
            traverse(child)


Data = read_data()
tokens = [str(i) for i in range(50)]
f = open('/home/omari/Dropbox/Thesis/Evaluation/tags.txt', 'w')
for id in Data['RCL']:
    # print id
    # S = _fix_sentences(Data['commands_id'][id])
    # f.write(S+'\n')
    # print Data['RCL'][id]
    # tree = Tree.fromstring(Data['RCL'][id])
    # tree.pretty_print()
    tree = ParentedTree.fromstring(Data['RCL'][id])
    # print tree
    # leaves = tree.leaves()
    sentence = []
    for p in tree.pos():
        if str(p[0]) not in tokens:
            # print (p[0],p[1].split(':')[0])
            sentence.append(p[1].split(':')[0])
    S = ' '.join(sentence)
    f.write(S+'\n')
    # traverse(tree)
    # print '----------'
    # break
f.close()
