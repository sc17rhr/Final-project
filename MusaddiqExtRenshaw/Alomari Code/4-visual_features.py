import math as m
import pickle

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import itertools
from sklearn import metrics
from sklearn import mixture
from sklearn import svm


# --------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_layout.p'
    data = open(pkl_file, 'rb')
    positions = pickle.load(data)
    return positions


def _get_actions(positions):
    actions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x_O = positions[mov_obj]['x']
        y_O = positions[mov_obj]['y']
        z_O = positions[mov_obj]['z']

        x_R = positions['gripper']['x']
        y_R = positions['gripper']['y']
        z_R = positions['gripper']['z']

        # check if it's a pick up
        if x_O[1] == x_R[1] and y_O[1] == y_R[1] and z_O[1] == z_R[1]:
            actions = ['approach,grasp,lift']
        elif x_O[0] == x_R[0] and y_O[0] == y_R[0] and z_O[0] == z_R[0]:
            actions = ['discard']  ## lower ?!?!?!?
        elif x_O[0] != x_O[1] or y_O[0] != y_O[1] or z_O[0] != z_O[1]:
            actions = ['approach,grasp,lift', 'discard', 'approach,grasp,lift,move,discard,depart']
    else:
        actions = []  # 'nothing'
    return actions


def _get_trees(actions, positions):
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break

    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        z = positions[mov_obj]['z']

    tree = {}
    if actions == ['approach,grasp,lift']:
        tree['NLTK'] = "(V (Action " + actions[0] + ") (Entity id_" + str(mov_obj) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
    elif actions == ['discard']:
        tree['NLTK'] = "(V (Action " + actions[0] + ") (Entity id_" + str(mov_obj) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
        # tree['py']['D'] = [x[1],y[1],z[1]]
    elif actions == ['approach,grasp,lift', 'discard', 'approach,grasp,lift,move,discard,depart']:
        tree['NLTK'] = "(V (Action " + actions[2] + ") (Entity id_" + str(mov_obj) + ") (Destination " + str(
            x[1]) + "," + str(y[1]) + "," + str(z[1]) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[2]
        tree['py']['E'] = mov_obj
        tree['py']['D'] = [x[1], y[1], z[1]]
    elif actions == ['nothing']:
        tree['NLTK'] = "(V (Action " + actions[0] + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
    return tree


def _get_locations(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        if x[0] < 3 and y[0] < 3:
            locations.append([0, 0])
        if x[0] < 3 and y[0] > 4:
            locations.append([0, 7])
        if x[0] > 4 and y[0] < 3:
            locations.append([7, 0])
        if x[0] > 4 and y[0] > 4:
            locations.append([7, 7])
        if x[0] > 1 and x[0] < 5 and y[0] > 1 and y[0] < 5:
            locations.append([3.5, 3.5])

        if x[1] < 3 and y[1] < 3:
            locations.append([0, 0])
        if x[1] < 3 and y[1] > 4:
            locations.append([0, 7])
        if x[1] > 4 and y[1] < 3:
            locations.append([7, 0])
        if x[1] > 4 and y[1] > 4:
            locations.append([7, 7])
        if x[1] > 1 and x[1] < 5 and y[1] > 1 and y[1] < 5:
            locations.append([3.5, 3.5])
    return locations


def _get_locations2(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            x = positions[obj]['x'][1]
            y = positions[obj]['y'][1]

            if [x, y] not in locations:
                locations.append([x, y])
    # print locations

    return locations


def _get_colors(positions):
    colors = []
    for obj in positions:
        if obj != 'gripper':
            color = positions[obj]['F_HSV']
            for c in color.split('-'):
                if c not in colors:
                    colors.append(c)
    return colors


def _get_shapes(positions):
    shapes = []
    for obj in positions:
        if obj != 'gripper':
            shape = positions[obj]['F_SHAPE']  # type: object
            for s in shape.split('-'):
                if s not in shapes:
                    shapes.append(s)

    # groups = {}
    # for obj in positions:
    #     if obj != 'gripper':
    #         x=positions[obj]['x'][0]
    #         y=positions[obj]['y'][0]
    #         if positions[obj]['F_SHAPE'] in ['cube','cylinder']:
    #             if (x,y) not in groups:
    #                 groups[(x,y)]=1
    #             else:
    #                 groups[(x,y)]+=1
    # for i in groups:
    #     if groups[i]>1:
    #         shapes.append('tower')
    #         break
    return shapes


def _get_distances(positions):
    distances = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        # z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                d = [np.abs(x1[0] - x2[0]), np.abs(x1[1] - x2[1]), np.abs(y1[0] - y2[0]), np.abs(y1[1] - y2[1])]
                for i in d:
                    if i not in distances:
                        distances.append(i)
    return distances


def cart2sph(x, y, z):
    num = 90
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq)) * 180 / np.pi  # theta
    elev = int(elev / num) * num
    az = m.atan2(y, x) * 180 / np.pi  # phi
    az = int(az / num) * num
    return int(elev), int(az)


def _func_directions(dx, dy, dz):
    dx = float(dx)
    dy = float(dy)
    dz = float(dz)
    max = np.max(np.abs([dx, dy, dz]))
    if np.abs(dx) / max < .5:
        dx = 0
    else:
        dx = np.sign(dx)

    if np.abs(dy) / max < .5:
        dy = 0
    else:
        dy = np.sign(dy)

    if np.abs(dz) / max < .5:
        dz = 0
    else:
        dz = np.sign(dz)
    return dx, dy, dz


def _get_directions(positions):
    # http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    directions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                z2 = positions[obj]['z']
                # d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                d = _func_directions(x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0])
                if d not in directions:
                    directions.append(d)
                # d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                d = _func_directions(x1[1] - x2[1], y1[1] - y2[1], z1[1] - z2[1])
                if d not in directions:
                    directions.append(d)
    return directions


def _get_directions2(positions):
    directions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    for obj1 in positions:
        if obj1 != mov_obj:
            continue
        x1 = positions[obj1]['x']
        y1 = positions[obj1]['y']
        z1 = positions[obj1]['z']
        for obj2 in positions:
            if obj2 != 'gripper' and obj2 != obj1:
                x2 = positions[obj2]['x']
                y2 = positions[obj2]['y']
                z2 = positions[obj2]['z']
                # d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                d = [x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0]]
                dx = float(d[0])
                dy = float(d[1])
                dz = float(d[2])
                max = np.max(np.abs([dx, dy, dz]))
                if max != 0:
                    d = [d[0] / max, d[1] / max, d[2] / max]
                # print "---",d
                # d = _func_directions(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                # if d not in directions:
                directions.append(d)
                # # d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                # d = _func_directions(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                # if d not in directions:
                #     directions.append(d)
    return directions


def _get_temporal(v):
    temporal = []
    if len(v) > 1:
        temporal = ['meets']
    return temporal


def _cluster_data(X, GT, name, n):
    print name
    for i in range(5):
        print '#####', i
        n_components_range = range(5, n)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        lowest_bic = np.infty
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
                gmm.fit(X)
                Y_ = gmm.predict(X)
                ######################################
                bic = gmm.bic(X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    final_Y_ = Y_
                ######################################
    pickle.dump([final_Y_, best_gmm], open(
        '../Datasets/Dataset1/results/' + name + '_clusters.p',"wb"))

    _print_results(GT, final_Y_, best_gmm)


def _append_data(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        npr = np.random.normal(mean, sigma, 1)
        uqi = unique_.index(i)

        d = uqi + npr
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_, d))
        GT_.append(i)
    return X_, unique_, GT_


def _append_data2(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        d = i + np.random.multivariate_normal(mean, sigma, 1)[0]
        # X.append(d[0])
        # Y.append(d[1])
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_, d))
        GT_.append(unique_.index(i))
    return X_, unique_, GT_


def _append_data3(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        # print i
        du = _func_directions(i[0], i[1], i[2])
        if du not in unique_:
            unique_.append(du)
        # print i,len(i)
        # d = i # + np.random.multivariate_normal(mean, sigma, 1)[0]
        # X.append(d[0])
        # Y.append(d[1])
        if X_ == []:
            X_ = [i]
        else:
            X_ = np.vstack((X_, i))
        GT_.append(unique_.index(du))
    return X_, unique_, GT_


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    lists = []
    for i in range(n):
        list1 = np.arange(i * l / n + 1, (i + 1) * l / n + 1)
        lists.append(list1)
    return lists


def _print_results(GT, Y_, best_gmm):
    # print v_measure_score(GT, Y_)
    true_labels = GT
    pred_labels = Y_
    print "\n dataset unique labels:", len(set(true_labels))
    print "number of clusters:", len(best_gmm.means_)
    print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
    print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))
    print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
    print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
    print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))


def _pretty_plot_directions():
    final_Y_, best_gmm = pickle.load(
        open('../Datasets/Dataset1/directions_clusters.p',"rb"))
    print best_gmm.means_
    mpl.rcParams['legend.fontsize'] = 10

    for cluster in range(9):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        counter = 0
        ax.plot([0, 1], [0, 0], [0, 0], 'r', linewidth=3)
        ax.plot([0, 0], [0, 1], [0, 0], 'g', linewidth=3)
        ax.plot([0, 0], [0, 0], [0, 1], 'b', linewidth=3)
        for i, j in zip(X_directions, final_Y_):
            if counter == 110:
                break
            if j == cluster:  # 0:
                # print i
                counter += 1
                # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
                # r = z**2 + 1
                d = np.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
                # print i,d
                if d != 0:
                    x = [0, i[0] / d]
                    y = [0, i[1] / d]
                    z = [0, i[2] / d]
                    ax.plot(x, y, z, 'y')

                # if i[0]>0:
                #     if i[1]>0:
                #         X, Y = np.mgrid[0:i[0]:3j, 0:i[1]:3j]
                #     if i[1]<0:
                #         X, Y = np.mgrid[0:i[0]:3j, i[1]:0:3j]
                #     if i[1]==0:
                #         X, Y = np.mgrid[0:i[0]:3j, 0:0.1:3j]
                # if i[0]<0:
                #     if i[1]>0:
                #         X, Y = np.mgrid[i[0]:0:3j, 0:i[1]:3j]
                #     if i[1]<0:
                #         X, Y = np.mgrid[i[0]:0:3j, i[1]:0:3j]
                #     if i[1]==0:
                #         X, Y = np.mgrid[i[0]:0:3j, 0:0.1:3j]
                # if i[0]==0:
                #     X, Y = np.mgrid[0:.01:3j, 0:1:3j]
                # Z = X
                # # print i
                # # print X
                # Y2=Y
                # if i[0]>0:
                #     Y2*=X
                # if i[0]<0:
                #     Y2*=-X
                #
                # # print Z
                # # print '-------------'
                # # print ttttt
                #
                # cset = ax.contour(X,Y2,Z, zdir='Y', offset=-1)#, cmap=cm.coolwarm)
                # ax.plot(x, y, zdir='z', offset=-1)#, cmap=cm.coolwarm)
                # ax.legend()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.savefig(
            '../Datasets/Dataset1/directions/' + str(cluster) + '_cluster.png')
        # plt.show()


def _pretty_plot_locations():
    clusters = {}
    XY = X_locations * 180 / 9 + 10
    # print GT_locations
    Y_, best_gmm = pickle.load(
        open(
            '../Datasets/Dataset1/results/locations_clusters.p',"rb"))
    print XY
    for x, val in zip(XY, Y_):
        if val not in clusters:
            clusters[val] = np.zeros((200, 200, 3), dtype=np.uint8)
        a, b = x
        a = int(a)
        b = int(b)
        for i in range(10):
            clusters[val][a - i:a + i, b - i:b + i, :] += 1
        if np.max(clusters[val]) == 255:
            clusters[val] *= 244 / 255
    avg_images = {}
    for c in clusters:
        plt.matshow(clusters[c][:, :, 0])
        plt.axis("off")
        plt.savefig(
            '/media/psf/Home/Desktop/Leeds Uni/Semester 2/MSC Project/Datasets/Dukes Datasets/results/locations/' + str(
                c) + '_cluster.png')
        # avg_images[c] = cv2.imread(dir_save+'avg_'+str(c)+".png")


def _pretty_plot_colours():
    cluster_images = {}
    # print '-------------------------------------',len(Y_),len(X)
    for rgb, val in zip(X, Y_):
        if val not in cluster_images:
            cluster_images[val] = []
        rgb = [rgb[0] + rgb[1] + rgb[2], int(rgb[2]), int(rgb[1]), int(rgb[0])]
        cluster_images[val].append(rgb)

    for val in cluster_images:
        cluster_images[val] = sorted(cluster_images[val])
        if len(cluster_images[val]) > 20:
            selected = []
            count = 0
            for i in range(0, len(cluster_images[val]), len(cluster_images[val]) / 19):
                if count < 20:
                    selected.append(cluster_images[val][i])
                    count += 1
            cluster_images[val] = selected
    image_cluster_total = np.zeros((im_len * 5 * 7, im_len * 5 * 5, 3), dtype=np.uint8) + 255
    paper_img = np.zeros((im_len * 5, im_len * 5 * 3, 3), dtype=np.uint8) + 255
    count3 = 0
    for count2, p in enumerate(cluster_images):
        maxi = len(cluster_images[p])
        image_avg = np.zeros((im_len, im_len, 3), dtype=np.uint8)
        image_cluster = np.zeros((im_len * 5, im_len * 5, 3), dtype=np.uint8) + 255
        # print maxi
        for count, rgb in enumerate(cluster_images[p]):
            img = np.zeros((im_len, im_len, 3), dtype=np.uint8)
            img[:, :, 0] += rgb[3]
            img[:, :, 1] += rgb[2]
            img[:, :, 2] += rgb[1]
            image_avg += img / (len(cluster_images[p]) + 1)
            ang = count / float(maxi) * 2 * np.pi
            xc = int(1.95 * im_len * np.cos(ang))
            yc = int(1.95 * im_len * np.sin(ang))
            # print xc,yc
            C = int(2.5 * im_len)
            x1 = int(xc - im_len / 2.0 + 2.5 * im_len)
            x2 = x1 + im_len
            y1 = int(yc - im_len / 2.0 + 2.5 * im_len)
            y2 = y1 + im_len
            cv2.line(image_cluster, (int(y1 + y2) / 2, int(x1 + x2) / 2), (C, C), (20, 20, 20), 2)
            # print x1,x2,y1,y2
            image_cluster[x1:x2, y1:y2, :] = img
        image_avg = cv2.resize(image_avg, (int(im_len * 1.4), int(im_len * 1.4)), interpolation=cv2.INTER_AREA)
        x1 = int((2.5 - .7) * im_len)
        x2 = int(x1 + 1.4 * im_len)
        image_cluster[x1:x2, x1:x2, :] = image_avg
        if count2 < 35:
            i1x = np.mod(count2, 7) * im_len * 5
            i2x = (np.mod(count2, 7) + 1) * im_len * 5
            i1y = int(count2 / 7) * im_len * 5
            i2y = (int(count2 / 7) + 1) * im_len * 5
            image_cluster_total[i1x:i2x, i1y:i2y, :] = image_cluster
            cv2.imwrite(dir_save + 'all_clusters.jpg', image_cluster_total)

        cv2.imwrite(dir_save + str(p) + '_cluster.jpg', image_cluster)
        cv2.imwrite(dir_save + str(p) + '_cluster_avg.jpg', image_avg)


def _svm(x, y, x_test, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    y_pred = clf.predict(x_test)
    mean = metrics.v_measure_score(y_test, y_pred)
    # mean/=50
    # print '-------'
    # print("supervised V-measure: %0.2f" % mean)
    print


##########################################################################
# save values for furhter analysis
##########################################################################
for scene in range(1, 1001):
    print 'extracting feature from scene : ', scene
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_visual_features.p'
    VF = {}
    positions = _read_pickle(scene)
    VF['actions'] = _get_actions(positions)
    VF['locations'] = _get_locations(positions)
    VF['color'] = _get_colors(positions)
    VF['type'] = _get_shapes(positions)
    # VF['distances'] = _get_distances(positions)
    VF['relation'] = _get_directions(positions)
    # VF['temporal'] = _get_temporal(VF['actions'])
    trees = _get_trees(VF['actions'], positions)
    pickle.dump([VF, trees], open(pkl_file, 'wb'))

##########################################################################
# Clustering analysis
##########################################################################
four_folds = chunks(1000, 4)

for test in range(1):
    X_colours = []
    X_colours_t = []
    GT_colours = []
    GT_colours_t = []
    unique_colours = []

    X_shapes = []
    GT_shapes = []
    X_shapes_t = []
    GT_shapes_t = []
    unique_shapes = []

    X_locations = []
    GT_locations = []
    X_locations_t = []
    GT_locations_t = []
    unique_locations = []

    X_directions = []
    GT_directions = []
    X_directions_t = []
    GT_directions_t = []
    unique_directions = []

    for c, data in enumerate(four_folds):
        if c != test:
            for scene in data:
                # print scene
                pkl_file = '../Datasets/Dataset1/learning/' + str(
                    scene) + '_visual_features.p'
                positions = _read_pickle(scene)
                X_colours, unique_colours, GT_colours = _append_data(_get_colors(positions), X_colours, unique_colours,
                                                                     GT_colours, 0, .4)
                X_shapes, unique_shapes, GT_shapes = _append_data(_get_shapes(positions), X_shapes, unique_shapes,
                                                                  GT_shapes, 0, .4)
                X_locations, unique_locations, GT_locations = _append_data2(_get_locations2(positions), X_locations,
                                                                            unique_locations, GT_locations, [0, 0],
                                                                            [[.4, 0], [0, .4]])
                X_directions, unique_directions, GT_directions = _append_data3(_get_directions2(positions),
                                                                               X_directions, unique_directions,
                                                                               GT_directions, [0, 0], [[0, 0], [0, 0]])

        if c == test:
            for scene in data:
                # print scene
                pkl_file = '../Datasets/Dataset1/learning/' + str(
                    scene) + '_visual_features.p'
                positions = _read_pickle(scene)
                X_colours_t, unique_colours, GT_colours_t = _append_data(_get_colors(positions), X_colours_t,
                                                                         unique_colours, GT_colours_t, 0, .35)
                X_shapes_t, unique_shapes, GT_shapes_t = _append_data(_get_shapes(positions), X_shapes_t, unique_shapes,
                                                                      GT_shapes_t, 0, .3)
                X_locations_t, unique_locations, GT_locations_t = _append_data2(_get_locations2(positions),
                                                                                X_locations_t, unique_locations,
                                                                                GT_locations_t, [0, 0],
                                                                                [[.4, 0], [0, .4]])
                X_directions_t, unique_directions, GT_directions_t = _append_data3(_get_directions2(positions),
                                                                                   X_directions_t, unique_directions,
                                                                                   GT_directions_t, [0, 0],
                                                                                   [[0, 0], [0, 0]])
    # print X_colours_t
    # print unique_directions
    # print GT_colours_t
    _cluster_data(X_colours, GT_colours, "colours", 9)
    _svm(X_colours, GT_colours, X_colours_t, GT_colours_t)

    _cluster_data(X_shapes, GT_shapes, "shapes", 9)
    _svm(X_shapes, GT_shapes, X_shapes_t, GT_shapes_t)

    _cluster_data(X_locations, GT_locations, "locations", 9)
    _svm(X_locations, GT_locations, X_locations_t, GT_locations_t)

    _cluster_data(X_directions, GT_directions, "directions", 15)
    _svm(X_directions, GT_directions, X_directions_t, GT_directions_t)
    # print '-------------------'
    # _pretty_plot_directions()
    # _pretty_plot_locations()
# _pretty_plot_colours() ## not yet working
