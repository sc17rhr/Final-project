import cv2
import numpy as np

colours = '/home/omari/Datasets/Dukes_modified/results/colours/'
shapes = '/home/omari/Datasets/Dukes_modified/results/shapes/'
directions = '/home/omari/Datasets/Dukes_modified/results/directions/'
locations = '/home/omari/Datasets/Dukes_modified/results/locations/'
aggregates = '/home/omari/Datasets/Dukes_modified/results/aggregates/'
image_dir = "/home/omari/Dropbox/Thesis/writing/Chapter7/Chapter7Figs/PNG/Dukes-visual-concepts.jpg"

color_im = ['13_cluster', '14_cluster']
shape_im = ['0_cluster', '1_cluster']
direction_im = ['7_cluster', '3_cluster']
location_im = ['1_cluster', '7_cluster']
distance_im = ['0_cluster', '2_cluster']
all_images = [color_im, shape_im, direction_im, location_im]  # , color_im, location_im, distance_im]
all_dir = [colours, shapes, directions, locations]  # , colors, locations, distances]#, actions, objects]
text = ["colour", "shape", "direction", "location", "aggregate"]  # , "colour", "location", "distance"]#, "action", "object"]

im_len = 60
th = 40
font = cv2.FONT_HERSHEY_SIMPLEX

image = np.zeros((im_len * 5 * 2 + th * 2, im_len * 5 * 4 + th * 3, 3), dtype=np.uint8) + 255

for c1, i in enumerate(all_images):
    dir_ = all_dir[c1]
    t = text[c1]
    for c2, j in enumerate(i):
        im = cv2.imread(dir_ + j + ".png")
        if t == "direction":
            im = im[60:-50, 100:-100, :]
        print dir_ + j + ".png"
        im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_AREA)
        image[c2 * (im_len * 5) + c2 * th:(c2 + 1) * (im_len * 5) + c2 * th,
        c1 * (im_len * 5) + c1 * th:(c1 + 1) * (im_len * 5) + c1 * th, :] = im
        cv2.putText(image, t + '_' + str(c2),
                    (100 + c1 * (im_len * 5) + c1 * th, (c2 + 1) * (im_len * 5) + (c2) * th + th * 1 / 2), font, .8,
                    (0, 0, 0), 2)
        # print t
cv2.imwrite(image_dir, image)
cv2.imshow("test", image)
cv2.waitKey(3000)
