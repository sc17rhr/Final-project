from nltk import *
from nltk.corpus import treebank
from nltk.treetransforms import *
import dynamic_pcfg
from nltk import grammar, parse

"""
This file contains some useful utilities for probabilistic parsing in Lab 2.
"""

# Three parse trees that you'll use for the first question.
# three_trees = [bracket_parse(t) for t in
# ['''(S (NP John) (VP (V1 said) (SBAR (COMP that)
# (S (NP Sally) (VP (VP (V2 snored)) (ADVP loudly))))))''',
# '''(S (NP Sally) (VP (V1 declared) (SBAR (COMP that)
# (S (NP Bill) (VP (VP (V2 ran)) (ADVP quickly))))))''',
# '''(S (NP Fred) (VP (V1 pronounced) (SBAR (COMP that)
# (S (NP Jeff) (VP (VP (V2 swam)) (ADVP elegantly))))))''']]
#
# print three_trees

sentences = [
'''The luxury auto maker last year sold 1,000 cars in the U.S.''',
'''Bell Industries Inc. increased its quarterly to 10 cents from seven cents
a share .''',
'''The dispute shows clearly the global power of Japan 's financial titans .''',
'''Was this why some of the audience departed before or during the second
half ?'''
]

def learn_treebank(files=None, markov_order=None):
    """
    Learn a PCFG from the Penn Treebank, and return it.

    By default, this learns from NLTK's 10% sample of the Penn Treebank.
    You can give the filename of a Treebank file; 'wsj-02-21.mrg' will
    learn from the entire training section of Treebank.
    """
    if files is None: bank = treebank.parsed_sents()
    else: bank = treebank.parsed_sents(files)
    return learn_trees(bank, collapse=True, markov_order=markov_order)

def learn_trees(trees, collapse=True, markov_order=None):
    """
    Given a list of parsed sentences, return the maximum likelihood PCFG
    for those sentences.

    If 'collapse' is True, it will collapse the trees before learning the
    grammar so that there are no unary productions.

    This will reduce productions of length more than 2 using Chomsky normal
    form.  You can Markov-smooth the results by setting markov_order to a
    number such as 2.
    """
    productions = []
    for tree in trees:
      if collapse: tree.collapse_unary(collapsePOS=False)
      if markov_order:
        tree.chomsky_normal_form(horzMarkov=markov_order)
      else:
        tree.chomsky_normal_form()
      productions += tree.productions()

    grammar1 = grammar.induce_pcfg(Nonterminal('event:'), productions)
    return grammar1

def prob_parse(grammar, sentence, n=1):
    """
    Return the n most likely parses (default 1) for a sentence, given a PCFG.

    If n=1, this will use Viterbi (A*) parsing for efficiency.

    If the grammar was trained on trees in Chomsky normal form, this function
    will un-Chomsky the trees before outputting them.
    """
    words = sentence.split()
    if n == 1:
        parses = [dynamic_pcfg.best_parse_with_n_grams(grammar, sentence, trace=5)]
    # else:
        # parser = InsideChartParser(grammar, trace=15)
        # parses = parser.nbest_parse(words, n)
    for parse in parses: un_chomsky_normal_form(parse)
    return parses

def change_start(grammar, newstart):
    """
    Return a new grammar with a different start symbol. This can be useful
    for parsing a single noun phrase instead of a complete sentence, for
    example.
    """
    if isinstance(newstart, basestring): newstart=Nonterminal(newstart)
    return Grammar(newstart, grammar.productions())
