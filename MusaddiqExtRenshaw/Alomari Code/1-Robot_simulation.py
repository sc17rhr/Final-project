# from visual import *
import numpy as np
from robot_functions import *

R = Robot()

save = 0
save = 'save'  # saving image

# scenes = [1,2,3,4,10,11,12,205,206,901,902]
for scene in range(1, 1001):
    # for scene in scenes:

    R.scene = scene
    R._initilize_values()
    R._fix_sentences()
    R._change_data()
    # print 'simulating scene number :',scene
    R._initialize_scene()  # place the robot and objects in the initial scene position without saving or motion
    R._print_scentenses()  # print the sentences on terminal and remove the SPAM sentence
    R._save_motion()                       # save the motion into a text file
    # R._clear_scene()                       # remove the objects from the scene once it's done
#print R.Data['commands']
print len(R.all_words)
# print R.total_num_objects/1000.0
