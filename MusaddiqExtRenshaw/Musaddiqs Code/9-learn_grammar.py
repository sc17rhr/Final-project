import pickle
import numpy as np
from learn_pcfg import *


# ---------------------------------------------------------------------------#
def _read_stop_wrods():
    pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    stop = pickle.load(data)
    return stop


# ---------------------------------------------------------------------------#
def _read_RCL_tree(id):
    pkl_file = '../Datasets/Dataset1/RCL-trees/' + str(
        id) + '_tree.p'
    data = open(pkl_file, 'rb')
    RCL_tree = pickle.load(data)
    return RCL_tree


# ---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '../Datasets/Dataset1/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]


# ---------------------------------------------------------------------------#
def _read_sentences(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


# ---------------------------------------------------------------------------#
def _read_vf(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    vf, tree = pickle.load(data)
    return vf, tree


# ---------------------------------------------------------------------------#
def _read_semantic_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_semantic_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


# ---------------------------------------------------------------------------#
def _read_layout(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_layout.p'
    data = open(pkl_file, 'rb')
    layout = pickle.load(data)
    return layout


# ---------------------------------------------------------------------------#
def _read_grammar_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


# ---------------------------------------------------------------------------#
def _read_passed_tags():
    pkl_file = '../Datasets/Dataset1/matching/Passed_tags.p'
    data = open(pkl_file, 'rb')
    Matching, Matching_VF, passed_scenes, passed_sentences = pickle.load(data)
    # print Matching,Matching_VF,passed_scenes,passed_ids
    return [Matching, Matching_VF, passed_scenes, passed_sentences]


# ---------------------------------------------------------------------------#
def _is_yuk(sentence):
    yuk = 0
    yuk_words = [' and', ' closest', ' near', ' far', ' nearest', ' edge', ' corner', ' side', ' leftmost', 'rightmost',
                 '1', '2', '3', '4', '5', 'two', 'one', 'three', 'five', 'four', 'single', 'position', 'grid',
                 'right', 'lift', 'box', 'left', 'location', 'exactly', 'lower', 'that', 'next', 'lowest', 'opposite',#-------------------------------------------------------------------------------
                 ' it', ' to ']
    for word in yuk_words:
        if word in sentence:
            yuk = 1
            break
    return yuk


# ---------------------------------------------------------------------------#
def _get_GT():
    GT_total_links = 0
    GT_dict = {}
    GT_dict["actions"] = ["move", "pick", "pick up", "drop", "put down", "place", "take", "shift", "put", "remove",
                          "stack"]
    GT_dict["colours"] = ["black", "green", "blue", "yellow", "pink", "purple", "grey", "gray", "red", "magenta",
                          "white", "cyan", "turquoise", "sky"]
    GT_dict["shapes"] = ["sphere", "can", "ball", "cylinder", "prism", "cube", "pyramid", "orb", "block", "square",
                         "tower", "cylinders", "tetrahedron", "cans", "blocks", "cubes"]
    GT_dict["directions"] = ["on", "on top of", "over", "onto", "above", "front", "right", "back", "left"]
    GT_dict["locations"] = ["top left", "top right", "center", "bottom right", "bottom left"]
    GT_dict["aggregates"] = ["row", "column","stack"]
#    GT_dict["superlatives"] = ["leftmost", "rightmost"]
    return GT_dict


# ---------------------------------------------------------------------------#
def _read_tree(id):
    pkl_file = '../Datasets/Dataset1/matching/' + str(
        id) + '.p'
    data = open(pkl_file, 'rb')
    results = pickle.load(data)
    return results


def _create_simple_entity(categries, words):
    sub_trees = []
    for id in categries:
        for cat, word in zip(categries[id], words):
            sub_trees.append(Tree(cat.split('_')[0] + ':', [word]))
    return sub_trees
    # print words


def _get_entity(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    entity = results['entity']
    # print entity
    if len(entity[0]) == 1:
        Ent = _create_simple_entity(entity[1], grammar[struct['E']])
    if len(entity[0]) == 3:
        # print 'FIXXXXXX MEEEEEE !!!!! help help'
        Ent = ['fix me']
    return Ent


def _get_action(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    action = (' ').join(grammar[struct['A']])
    return [action]


# def _get_relation(results):
#     struct = results['tree_structure']
#     grammar = results['grammar']
#     action = (' ').join(grammar[struct['A']])
#     return [action]

def _get_destination(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    destination = results['destination']
    words = grammar[struct['D']]
    count = 0
    rel_words = []
    for r in destination[2]:
        for rel in destination[2][r]:
            rel_words.append(words[count])
            count += 1
    R = Tree('relation:', [(' ').join(rel_words)])
    E = Tree('entity:', _create_simple_entity(destination[1], words[count:]))
    dest = Tree('spatial-relation:', [R, E])
    return [dest]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    lists = []
    for i in range(n):
        list1 = np.arange(i * l / n + 1, (i + 1) * l / n + 1)
        lists.append(list1)
    return lists
    #
    # for i in range(0, len(l), n):
    #     yield l[i:i + n]


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/treebank/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes


def _evaluate(four_folds, test, data, grammar, fw):
    number_of_sentences = 0
    number_of_correctly_parsed = 0
    for c, data in enumerate(four_folds):
        if c == test:
            for scene in data:
                sentences = _read_sentences(scene)
                for id in sentences:
                    number_of_sentences += 1
                    s = sentences[id]['text']
                    s = s.split(" ")
                    for word in fw:  # remove function words
                        while word in s:
                            s.remove(word)
                    s = (" ").join(s)
                    RCL = sentences[id]['RCL']
                    # print s
                    parse = prob_parse(grammar, s, 1)
                    # print parse[0][0]
                    # print parse[0][1]
                    number_of_correctly_parsed += parse[0][1]
                    # print '-----------------------------'
                    # if parse != [None]:
                    # break
                # break

    print "Fold number ", test, "correctly parsed ", number_of_correctly_parsed, "out of ", number_of_sentences, "with a ratio ", number_of_correctly_parsed / number_of_sentences, "with ", len(
        grammar.productions()), "grammar productions"


# ---------------------------------------------------------------------------#
dir_save = "../Datasets/Dataset1/"
Matching, Matching_VF, passed_scenes, passed_sentences = _read_passed_tags()
# sentences_to_test = {}
bad_trees = [14588, 23958, 10646, 25409, 25625, 14427, 23982, 16360, 22369, 23928, 16792, 18058, 25013, 9323, 26997,
             25565, 14412, 16159, 26955, 4028, 9207, 18582, 25100, 25058, 23428, 23985, 12027, 25653, 14624, 14423,
             25682, 12515, 13775, 4073, 10186, 13046, 25622, 26283, 23217, 12453, 23955, 23970, 23756, 23898, 14789,
             25477, 9418, 2541, 23738, 24170]

scenes = _read_dataset()
four_folds = chunks(len(scenes), 4)
words_in_tree = []
for scene in scenes:
    sentences = _read_sentences(scene)
    for id in sentences:
        if id not in bad_trees:
            if not _is_yuk(sentences[id]['text']):
                if id in passed_sentences:
                    RCL_tree = _read_RCL_tree(id)
                    results = _read_tree(id)
                    struct = results['tree_structure']
                    A = Tree('action:', _get_action(results))
                    E = Tree('entity:', _get_entity(results))
                    if len(struct) == 2:
                        tree = Tree('event:', [A, E])
                    if len(struct) == 3:
                        D = Tree('destination:', _get_destination(results))
                        tree = Tree('event:', [A, E, D])
                    if tree == RCL_tree:
                        s = sentences[id]['text']
                        words_in_sent = s.split(" ")
                        leaves = tree.leaves()
                        for leaf in leaves:
                            if leaf not in words_in_tree:
                                words_in_tree.append(leaf)
print words_in_tree

#
# ######################################## Grammar
for test in range(4):
    counter = 0
    counter2 = 0
    counter_results = 0
    matched_trees = []
    RCL_trees = []
    scenes = []
    # ok_words = []
    fw = []
    for c, data in enumerate(four_folds):
        if c != test:
            for scene in data:
                try:
                    sentences = _read_sentences(scene)
                except IOError:
                    print "not found"
                for id in sentences:
                    if scene in [8, 14, 121]:
                        sentences[id]['text']
                        print scene, ">>>>>>>>>>>>>>>", sentences
                    if id not in bad_trees:
                        if not _is_yuk(sentences[id]['text']):
                            try:
                                RCL_trees.append(_read_RCL_tree(id))
                            except IOError:
                                print("file not found")
                            # sentences_to_test[id] = sentences[id]
                            if scene not in scenes:
                                scenes.append(scene)
                            counter += 1
                            if id in passed_sentences:
                                try:
                                    RCL_tree = _read_RCL_tree(id)
                                except IOError:
                                    print("file not found")
                                # RCL_trees.append(RCL_tree)
                                # print RCL_tree
                                # print 'sentence:',sentences[id]['text']
                                results = _read_tree(id)
                                struct = results['tree_structure']
                                # print struct
                                A = Tree('action:', _get_action(results))
                                E = Tree('entity:', _get_entity(results))
                                if len(struct) == 2:
                                    tree = Tree('event:', [A, E])
                                if len(struct) == 3:
                                    D = Tree('destination:', _get_destination(results))
                                    tree = Tree('event:', [A, E, D])
                                if tree == RCL_tree:
                                    # print RCL_tree
                                    # print tree.leaves()
                                    # print '****',tree
                                    counter_results += 1
                                    matched_trees.append(tree)
                                    s = sentences[id]['text']
                                    words_in_sent = s.split(" ")
                                    leaves = tree.leaves()
                                    words_in_tree = []
                                    for leaf in leaves:
                                        for l in leaf.split(" "):
                                            words_in_tree.append(l)
                                    # print leaves
                                    for word in words_in_sent:
                                        if word not in words_in_tree:
                                            if word not in fw:
                                                fw.append(word)
                                    pass
                                else:
                                    # print tree
                                    # print RCL_tree
                                    counter2 += 1
# #     # print fw
# #     # print counter
# #     # print counter_results
# #     # print counter2
    s = Tree('pick up', ['pick', 'up'])
    matched_trees.append(s)
# #
# #     # ################################################################
# #     # # Evaluates my system
# #     # ################################################################
    grammar = learn_trees(matched_trees)
    _evaluate(four_folds, test, data, grammar, fw)
        #print grammar
# #     #
# #     # # ################################################################
# #     # # # Evaluates supervised system
# #     # # ################################################################
    grammar = learn_trees(RCL_trees)
    _evaluate(four_folds, test, data, grammar, fw)
#     print grammar
