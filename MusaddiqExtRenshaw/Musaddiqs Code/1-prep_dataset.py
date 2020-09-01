#  this files handles the data preparation. This is code from alomari that is used
#  to fix the sentences in cases that have wrong spellings
#  the file is also used to generate and save some other files.
# it works with the robot_functions files which works with the xml_function files
from robot_functions import *
import re

R = Robot()

save = 0
save = 'save'  # saving image

accepted_scenes = []


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("/mnt/c/Users/The Rench/Final_Project/Datasets/Dataset1/treebank/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    #print accepted_scenes
    return accepted_scenes

#_read_dataset()
#print accepted_scenes

for scene in _read_dataset():
    #print '4'
    R.scene = scene
    R._initilize_values()
    R._fix_sentences()
    R._change_data()
    R._initialize_scene()  # place the robot and objects in the initial scene position without saving or motion
    R._print_scentenses()  # print the sentences on terminal and remove the SPAM sentence
    R._save_motion()

print len(R.all_words)
