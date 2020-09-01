import cv2
import numpy as np

count = 0
for i in range(1,260):
    img_final = np.zeros((10,10,3),dtype=np.uint8)
    for j in ["0002","0009","0016","0023"]:
        if i < 10:
            dir1 = "/home/omari/Datasets/Dukes_modified/images/"+str(i)+"/scene_000"+str(i)+"_frame_"+j+".png"
        elif i < 100:
            dir1 = "/home/omari/Datasets/Dukes_modified/images/"+str(i)+"/scene_00"+str(i)+"_frame_"+j+".png"
        elif i < 1000:
            dir1 = "/home/omari/Datasets/Dukes_modified/images/"+str(i)+"/scene_0"+str(i)+"_frame_"+j+".png"
        elif i < 10000:
            dir1 = "/home/omari/Datasets/Dukes_modified/images/"+str(i)+"/scene_"+str(i)+"_frame_"+j+".png"
        print dir1
        img = cv2.imread(dir1)
        # print img
        try:
            img = img[100:-150,100:-300,:]
            if j == "0002":
                img_final = img
            else:
                img_final = np.concatenate((img_final, img), axis=1)
        except Exception as e:
            # cv2.imwrite("/home/omari/Datasets/robot_modified/all_images/robot_"+str(count)+".png",img_final)
            # count+=1
            print "wrong scene"

    cv2.imwrite("/home/omari/Datasets/Dukes_modified/results/robot_"+str(i)+".jpg",img_final)
    # count+=1
    # cv2.imshow("img",img_final)
    # cv2.waitKey(200)
