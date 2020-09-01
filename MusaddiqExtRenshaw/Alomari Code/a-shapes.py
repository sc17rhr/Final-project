import cv2
import numpy as np
import glob
from operator import add
import pickle
from sklearn import mixture
# import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
from sklearn import svm

class shapes():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Dukes_modified/"
        self.dir_save = "/home/omari/Datasets/Dukes_modified/results/shapes/"
        self.dir_scale = "/home/omari/Datasets/scalibility/Dukes/"
        self.th = 10
        self.sp = 2
        self.X = []     # fpfh vales
        self.gX = []    # gfpfh values
        self.eX = []    # esf vales
        self.GT = []
        self.shapes = {}
        self.images = []
        self.im_len = 60
        self.shapes_per_video = {}

    def _extract_object_images(self):
        f_x, f_y = 1212.9-700, 1129.0-700
        c_x, c_y = 187.3-700, 439.6-700
        for video in range(1,205):
            print 'processing video: ',video
            dir1 = self.dir+str(video)+"/features/shapes/"
            img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
            height, width = img.shape[:2]
            unique_objects = sorted(glob.glob(dir1+"fpfh*.pcd"))
            for obj in range(len(unique_objects)):
                print obj
                dir2 = self.dir+str(video)+"/clusters/cloud_cluster_"+str(obj)
                tracks = sorted(glob.glob(dir2+".pcd"))
                for f1 in tracks:
                    X = []
                    Y = []
                    f = open(f1,"r")
                    for count,line in enumerate(f):
                        line = line.split("\n")[0]
                        if count == 6:      # get width
                            num = int(line.split(" ")[1])
                        if count > 10:
                            xyz = map(float,line.split(" "))
                            x,y,z = xyz[:-1]
                            x_2d = int((x/z)*f_x + c_x)
                            y_2d = int((y/z)*f_y + c_y)
                            if x_2d < 0:
                                x_2d += width
                            if y_2d < 0:
                                y_2d += height
                            X.append(x_2d)
                            Y.append(y_2d)
                    x1,x2 = np.min(X)-20,np.max(X)+20
                    y1,y2 = np.min(Y)-20,np.max(Y)+20
                    a = y2-y1
                    b = x2-x1
                    xo = (x2+x1)/2.0
                    yo = (y2+y1)/2.0
                    A = np.max([a,b])
                    x1,x2 = int(xo-A/2), int(xo+A/2)
                    y1,y2 = int(yo-A/2), int(yo+A/2)
                    cv2.imwrite(self.dir+str(video)+"/clusters/obj_"+str(obj)+".png",img[y1:y2,x1:x2,:])

    def _read_shapes(self):
        for video in range(1,205):
            self.shapes_per_video[video] = []
            dir1 = self.dir+str(video)+"/features/shapes/"
            dir2 = self.dir+str(video)+"/ground_truth/"
            files = sorted(glob.glob(dir1+"fpfh*.pcd"))
            gfiles = sorted(glob.glob(dir1+"gfpfh_*.pcd"))
            # efiles = sorted(glob.glob(dir1+"esf*.pcd"))
            # efiles = gfiles
            ground = sorted(glob.glob(dir2+"GT_obj*.txt"))
            # print ground
            for f1,f2,f3 in zip(files,ground,gfiles):
                num=1
                fpfh = 0

                f = open(f1,"r")
                for count,line in enumerate(f):
                    line = line.split("\n")[0]
                    if count == 6:      # get width
                        num = int(line.split(" ")[1])
                    if count == 11:
                        fpfh = map(float,line.split(" "))
                    if count > 11:
                        a = map(float,line.split(" "))
                        fpfh = map(add, fpfh, a)
                fpfh = [x / num for x in fpfh]
                if self.X == []:
                    self.X = fpfh
                else:
                    self.X = np.vstack((self.X,fpfh))
                f.close()

                f = open(f3,"r")
                for count,line in enumerate(f):
                    line = line.split("\n")[0]
                    if count == 11:      # get width
                        a = map(float,line.split(" "))
                        gfpfh = a
                if self.gX == []:
                    self.gX = [gfpfh]
                else:
                    self.gX = np.vstack((self.gX,gfpfh))
                f.close()


                X = list(fpfh)+list(gfpfh)
                if self.shapes_per_video[video] == []:
                    self.shapes_per_video[video] = X
                    self.shapes_per_video[video] = np.vstack((self.shapes_per_video[video],X))
                else:
                    self.shapes_per_video[video] = np.vstack((self.shapes_per_video[video],X))

                # f = open(f4,"r")
                # for count,line in enumerate(f):
                #     line = line.split("\n")[0]
                #     if count == 11:      # get width
                #         a = map(float,line.split(" "))
                #         esf = a
                # if self.eX == []:
                #     self.eX = [esf]
                # else:
                #     self.eX = np.vstack((self.eX,esf))
                # f.close()

                f = open(f2,"r")
                for line in f:
                    self.GT.append(line)
                    if line not in self.shapes:
                        self.shapes[line] = [fpfh]
                    else:
                        self.shapes[line].append(fpfh)
                f.close()
        pickle.dump( [self.shapes, self.GT, self.X, self.gX, self.eX], open("/home/omari/Datasets/Baxter_Dataset_final/features/shapes/shapes.p", "wb" ) )

    def _read_pickle_shapes(self):
        self.shapes,self.GT,self.X, self.gX, self.eX = pickle.load(open(self.dir_save+"shapes.p", "rb" ) )
        #
        # for shape in self.shapes:
        #     print shape,len(self.shapes[shape])
        # for i in range(len(self.GT)):
        #     if self.GT[i] in ["egg","carrot","banana","dolphin","apple","cup","octopus"]:
        #         self.GT[i] = "ball"

    def _read_shapes_images(self):
        # for video in range(1,205):
        dir1 = self.dir+"annotations/objects/"
        files = sorted(glob.glob(dir1+"obj*.jpg"))
        # print files
        for f1 in files:
            img = cv2.imread(f1)
            self.images.append(cv2.resize(img, (self.im_len,self.im_len), interpolation = cv2.INTER_AREA))

    def _plot_fpfh_values(self):
        th = self.th+15
        sp = self.sp+3
        for obj in self.shapes:
            L = len(self.shapes[obj])
            # L = 1
            img = np.zeros((th*L + sp*(L-1)+2*sp,   th*len(self.shapes[obj][0])+sp*(len(self.shapes[obj][0])-1)+2*sp,   3),dtype=np.uint8)
            for c1,i in enumerate(self.shapes[obj]):
                # print np.max(i)/100.0*255
                # if c1 == 30:
                    for c2,val in enumerate(i):
                        # c1 = 0
                        img[sp+c1*(th+sp):sp+c1*sp+(c1+1)*th,   sp+c2*(th+sp):sp+(c2+1)*th+c2*sp,  :] = int(val/100.0*255)+5
                # break
            cv2.imshow("img",img)
            cv2.imwrite(self.dir_save+"raw_features_"+obj+".png",img)

        for T in range(50):
            V = []
            count = 0
            for a1,a2 in zip(self.Y_,self.X):
                if a1 == T:
                    count+=1
                    if V == []:
                        V = a2
                    else:
                        V = map(add, V, a2)
            if count != 0:
                V = [x / float(count) for x in V]
                img = np.zeros((th*6+sp*5,  th*6+sp*5,  3),dtype=np.uint8)
                for c,val in enumerate(V):
                    a = np.mod(c,6)
                    b = int(c/6)
                    val = int(val/100.0*255)
                    img[a*(th+sp):(a+1)*th+a*sp,   b*(th+sp):(b+1)*th+b*sp,   :] = val
                cv2.imshow("img",img)
                cv2.imwrite(self.dir_save+"feature_"+str(T)+".png",img)

    def _cluster_objs(self):
        final_clf = 0
        best_v = 0
        X = []
        for i,j in zip(self.X,self.gX):
            if X == []:
                X = list(i)+list(j)
            else:
                X = np.vstack((X,list(i)+list(j)))
        for i in range(5):
            print '#####',i
            ## 18 components did well!! 0.45
            n_components_range = range(25, 30)
            cv_types = ['spherical', 'tied', 'diag', 'full']
            lowest_bic = np.infty
            for cv_type in cv_types:
                for n_components in n_components_range:
                    gmm = mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
                    gmm.fit(X)
                    # bic = gmm.bic(X)
                    # if bic < lowest_bic:
                    #     lowest_bic = bic
                    #     best_gmm = gmm
                    Y_ = gmm.predict(X)
                    v_meas = v_measure_score(self.GT, Y_)
                    if v_meas > best_v:
                        best_v = v_meas
                        final_clf = gmm
                        print best_v
            # clf = best_gmm
            # Y_ = clf.predict(X)
            # v_meas = v_measure_score(self.GT, Y_)
            # if v_meas > best_v:
            #     best_v = v_meas
            #     final_clf = clf
            #     print best_v
        # print best_gmm
        self.best_v = best_v
        Y_ = final_clf.predict(X)
        # self.predictions = {}
        # for obj in self.GT:
        #     self.predictions[obj] = np.zeros(len(final_clf.means_))
        #     for i in self.faces[obj]:
        #         self.predictions[obj][final_clf.predict([i])[0]]+=1
        #
        # self.cost_matrix = np.zeros((len(self.faces),len(final_clf.means_)))
        #
        # for count,obj in enumerate(self.predictions):
        #     self.predictions[obj]/=np.sum(self.predictions[obj])
        #     self.cost_matrix[count] = self.predictions[obj]
        #

        pickle.dump( [final_clf,self.best_v], open( self.dir_save+'shapes_clusters.p', "wb" ) )

    def _read_clusters(self):
        # final_clf,self.best_v = pickle.load(open( self.dir_save+'shapes_clusters.p', "rb" ) )
        # X = []
        # for i,j in zip(self.X,self.gX):
        #     if X == []:
        #         X = list(i)+list(j)
        #     else:
        #         X = np.vstack((X,list(i)+list(j)))
        # self.X = X
        # self.Y_ = final_clf.predict(self.X)
        self.Y_ = [0,0,0,0,0,0,0,0,0,0,0,0,
                   1,1,1,1,1,1,1,1,1,1,1,1,
                   2,2,2,2,2,2,2,2,2,2,2,2,
                   3,3,3,3,3,3,3,3,3,3,3,3]

        # ## get the clusters in each video
        # self.Y_per_video = {}
        # for i in self.shapes_per_video:
        #     Y_ = []
        #     X = final_clf.predict(self.shapes_per_video[i])
        #     for x in X:
        #         if x not in Y_:
        #             Y_.append(x)
        #     self.Y_per_video[i] = Y_
        # pickle.dump( [len(final_clf.means_),self.Y_per_video] , open( self.dir_save+'clusters_per_video.p', "wb" ) )
        #
        # unique_clusters = []
        # video = []
        # for i in self.Y_per_video:
        #     for j in self.Y_per_video[i]:
        #         if j not in unique_clusters:
        #             unique_clusters.append(j)
        #     video.append(len(unique_clusters))
        # pickle.dump( video , open( self.dir_scale+'shapes_per_video.p', "wb" ) )
        # print video

    def _pretty_plot(self):

        self.cluster_images = {}
        for img,val in zip(self.images,self.Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            self.cluster_images[val].append(img)

        for val in self.cluster_images:
            #self.cluster_images[val] = sorted(self.cluster_images[val])
            if len(self.cluster_images[val])>12:
                selected = []
                count = 0
                for i in range(0,len(self.cluster_images[val]),len(self.cluster_images[val])/12):
                    if count < 12:
                        selected.append(self.cluster_images[val][i])
                        count+=1
                self.cluster_images[val] = selected

        rows = 0
        maxi = 0
        for p in self.cluster_images:
            if len(self.cluster_images[p])>maxi:
                maxi = len(self.cluster_images[p])

        maxi = int(np.sqrt(maxi))
        for p in self.cluster_images:
            count = 0
            image = np.zeros((self.im_len*maxi,self.im_len*maxi,3),dtype=np.uint8)+255
            image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            for i in range(maxi):
                for j in range(maxi):
                    if count < len(self.cluster_images[p]):
                        image[i*self.im_len:(i+1)*self.im_len,j*self.im_len:(j+1)*self.im_len,:] = self.cluster_images[p][count]
                        image_avg += self.cluster_images[p][count]/(len(self.cluster_images[p])+1)
                    image[i*self.im_len:i*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[(i+1)*self.im_len:(i+1)*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  j*self.im_len:j*self.im_len+1,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  (j+1)*self.im_len:(j+1)*self.im_len+1,  :] = 0
                    count+=1
            # cv2.imshow('test',image)
            # cv2.waitKey(2000)
            cv2.imwrite(self.dir_save+"cluster_"+str(p)+'.png',image)
            # cv2.imwrite(self.dir_save+"cluster_"+str(p)+'_avg.jpg',image_avg)

        # self.cluster_images = {}
        # for img,val in zip(self.images,self.Y_):
        #     if val not in self.cluster_images:
        #         self.cluster_images[val] = []
        #     self.cluster_images[val].append(img)

        image_cluster_total = np.zeros((self.im_len*5*7,self.im_len*5*5,3),dtype=np.uint8)+255
        paper_img = np.zeros((self.im_len*5*2,self.im_len*5*4,3),dtype=np.uint8)+255
        # print len(self.cluster_images)
        # print iii
        count3 = 0
        for count2,p in enumerate(self.cluster_images):
            MAX_NUMBER_OF_IMAGES_SHOWN = 14
            maxi = np.min([len(self.cluster_images[p]),MAX_NUMBER_OF_IMAGES_SHOWN])
            # image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            image_cluster = np.zeros((self.im_len*5,self.im_len*5,3),dtype=np.uint8)+255
            # print maxi
            for count,img in enumerate(self.cluster_images[p]):
                img[0:2,:,:]=0
                img[-2:,:,:]=0
                img[:,0:2,:]=0
                img[:,-2:,:]=0
                # image_avg += img/(len(self.cluster_images[p])+1)
                ang = count/float(maxi)*2*np.pi
                xc = int(1.95*self.im_len*np.cos(ang))
                yc = int(1.95*self.im_len*np.sin(ang))
                # print xc,yc
                C = int(2.5*self.im_len)
                x1 = int(xc-self.im_len/2.0+2.5*self.im_len)
                x2 = x1+self.im_len
                y1 = int(yc-self.im_len/2.0+2.5*self.im_len)
                y2 = y1+self.im_len
                cv2.line(image_cluster,(int(y1+y2)/2,int(x1+x2)/2),(C,C),(20,20,20),2)
                # print x1,x2,y1,y2
                image_cluster[x1:x2,y1:y2,:] = img
            # image_avg = cv2.resize(image_avg, (int(self.im_len*1.4),int(self.im_len*1.4)), interpolation = cv2.INTER_AREA)
            x1 = int((2.5-.7)*self.im_len)
            x2 = int(x1+1.4*self.im_len)
            image_avg = cv2.imread(self.dir_save+"feature_"+str(p)+".png")
            image_avg = cv2.resize(image_avg, (int(self.im_len*1.4),int(self.im_len*1.4)), interpolation = cv2.INTER_AREA)
            image_cluster[x1:x2,x1:x2,:] = image_avg
            if count2<35:
                i1x = np.mod(count2,7)*self.im_len*5
                i2x = (np.mod(count2,7)+1)*self.im_len*5
                i1y = int(count2/7)*self.im_len*5
                i2y = (int(count2/7)+1)*self.im_len*5
                image_cluster_total[i1x:i2x,i1y:i2y,:] = image_cluster
                cv2.imwrite(self.dir_save+'all_clusters.jpg',image_cluster_total)

            if p in [0,7,19,26]:
                i1x = np.mod(count3,4)*self.im_len*5
                i2x = (np.mod(count3,4)+1)*self.im_len*5
                count3+=1
                i1y = 0
                i2y = self.im_len*5
                paper_img[i1y:i2y,i1x:i2x,:] = image_cluster
                cv2.imwrite(self.dir_save+'shapes_clusters_ex.jpg',paper_img)

            # cv2.imwrite(self.dir_save+str(p)+'_cluster.jpg',image_cluster)
            cv2.imwrite(self.dir_save+'cluster_images/'+str(p)+'_cluster.jpg',image_cluster)

    def _print_results(self):
        #print v_measure_score(self.GT, self.Y_)
        true_labels = self.GT
        pred_labels = self.Y_
        print "\n supervised number of clusters:", len(set(true_labels))
        print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))
        print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
        print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
        print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
        print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))

    def _SVM(self):
        maxi = 0
        mean = 0
        for i in range(50):
            clf = svm.SVC(kernel='linear')
            l1 = int(-.25*len(self.X))
            l2 = int(.75*len(self.X))
            clf.fit(self.X[:l1], self.GT[:l1])
            A = clf.predict(self.X[l2:])
            mean += metrics.v_measure_score(self.GT[l2:], A)
        mean/=50
        print("supervised V-measure: %0.2f" % mean)

def main():
    S = shapes()
    # S._extract_object_images()
    S._read_shapes()
    # S._read_pickle_shapes()
    S._read_shapes_images()
    # # S._cluster_objs()
    S._read_clusters()
    # S._plot_fpfh_values()
    S._pretty_plot()
    # S._SVM()
    # S._print_results()

if __name__=="__main__":
    main()
