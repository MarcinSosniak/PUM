import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import random

import sklearn.cluster as skc




def get_data_from_picture(path):
    img = Image.open(path,'r')
    arr=np.array(img)
    out=[]
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if not  (arr[x][y][0]==0 and arr[x][y][1]==0  and arr[x][y][2]==0 ):
                out.append((x,y))
    return np.array(out)




def get_centroid(cluster):
    sum_x = sum(map(lambda p : p[0],cluster))
    sum_y = sum(map(lambda p : p[1],cluster))
    len_cluster = len(cluster)
    return sum_x/len_cluster,sum_y/len_cluster


def dist(a,b):
    x_d=a[0]-b[0]
    y_d=a[1]-b[1]
    return math.sqrt(x_d**2 + y_d**2)

def dist2(a,b):
    x_d = a[0] - b[0]
    y_d = a[1] - b[1]
    return x_d ** 2 + y_d ** 2


def si(cluster,center=None,p=2):
    A = center if center is not None else get_centroid(cluster)
    T = len(cluster)
    wo_normalization = sum(map(lambda x: dist2(x,A),cluster)) if p ==2 else sum(map(lambda x: dist(x,A)**p,cluster))
    wo_root = wo_normalization/T
    return math.sqrt(wo_root) if p==2 else wo_root**(1/2)

def mij(cluster_i,cluster_j,center_i=None,center_j=None,p=2):
    Ai = center_i if center_i is not None else get_centroid(cluster_i)
    Aj = center_j if center_j is not None else get_centroid(cluster_j)
    if p==2:
        return dist(Ai,Aj)
    else:
        out = 0
        for k in range(len(Ai)):
            diff_p  = (Ai[k] - Aj[k])**p
            out += diff_p if diff_p>= 0 else -diff_p
        return out**(1/p)




def rij(cluster_i,cluster_j,p=2):
    centre_i = get_centroid(cluster_i)
    centre_j = get_centroid(cluster_j)
    Si = si(cluster_i,center=centre_i,p=p)
    Sj = si(cluster_j, center=centre_j, p=p)
    Mij = mij(cluster_i,cluster_j,center_i=centre_i,center_j=centre_j,p=p)
    return (Si+Sj)/Mij

def di(clusters,i,p):
    max_val = -1
    for j in range(len(clusters)):
        if j == i:
            continue
        Rij = rij(clusters[i],clusters[j],p=p)
        if Rij > max_val:
            max_val = Rij
    return max_val

def ref_val(clusters,p=2):
    clusters_len = len(clusters)
    return (sum(map(lambda i: di(clusters,i,p=p),range(clusters_len) )))/clusters_len


def get_clusters_from_kmeans(kmeans,k,data):
    clusters = [ [] for i in range(k)]
    labels= kmeans.labels_
    for i in range(len(data)):
        clusters[labels[i]].append(data[i])
    return clusters


def radnom_init(k,data):
    max_x = -1
    max_y = -1
    for elem in data:
        if elem[0] > max_x:
            max_x = elem[0]
        if elem[1] > max_y:
            max_y = elem[1]
    out = np.zeros((k,2))
    for i in range(k):
        out[i][0] = random.randint(0, max_x)
        out[i][1] = random.randint(0, max_y)
    return out


def random_part_init(k,data):
    clusters = [[] for i in range(k)]
    for elem in data:
        clusters[random.randint(0,k-1)].append(elem)
    out = np.zeros((k,2))
    for i in range(k):
        centr = get_centroid(clusters[i])
        out[i][0] = centr[0]
        out[i][1] = centr[1]
    return out



def k_means(k,n,type_name,data):
    if type_name.lower() == 'k-means++':
        return k_means_pp(k,n,data)
    elif type_name.lower() == 'forgy':
        return k_means_Forgy(k,n,data)
    elif type_name.lower() == 'random':
        return k_means_random(k,n,data)
    elif type_name.lower() == 'random_partition':
        return k_means_random_partition(k,n,data)
    raise TypeError


def k_means_random_partition(k, n, data):
    min_inertia = -1
    best_out = None
    for i in range(n):
        kmeans = skc.KMeans(n_clusters=k, init=random_part_init(k,data), n_init=1).fit(data)
        inertia = kmeans.inertia_
        if inertia < min_inertia or min_inertia == -1:
            min_inertia = inertia
            best_out = kmeans
    return best_out



def k_means_random(k,n,data):
    min_inertia = -1
    best_out = None
    for i in range(n):
        kmeans= skc.KMeans(n_clusters=k, init=radnom_init(k,data), n_init=1).fit(data)
        inertia= kmeans.inertia_
        if inertia < min_inertia or min_inertia==-1:
            min_inertia=inertia
            best_out=kmeans
    return best_out

def k_means_Forgy(k,n,data):
    return skc.KMeans(n_clusters=k, init='random', n_init=n).fit(data)

def k_means_pp(k,n,data):
    return skc.KMeans(n_clusters= k, init='k-means++',n_init=n).fit(data)


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def main_n():
    data = get_data_from_picture('in/base_set.png')
    types = ['k-means++', 'forgy', 'random', 'random_partition']
    n_list= [1,3,7,11,13]
    n_total_tries = 20
    scores={}
    for type_name in types:
        scores[type_name]= {}
        for i in n_list:
            scores[type_name][i] = []

    for k in range(n_total_tries):
        for type_name in types:
            print(type_name)
            for i in n_list:
                kmeans = k_means(9,i,type_name,data)
                scores[type_name][i].append(ref_val(get_clusters_from_kmeans(kmeans,9,data)))
                print('    n = ' + str(i) + '   score = ' + str(scores[type_name][i]))

    scores_lists = {}
    error_lists= {}

    for type_name in types:
        scores_lists[type_name] = []
        error_lists[type_name] = []
        for j in n_list:
            scores_lists[type_name].append( (np.mean(scores[type_name][j])))
            error_lists[type_name].append(np.std(scores[type_name][j]))



    x = np.array([ 2+(i*6) for i in range(len(n_list))])
    width = 1
    fig,ax = plt.subplots()

    rec_list_top = []
    rec_list_top.append(ax.bar(x - (width*1.5), scores_lists[types[0]], width, yerr=error_lists[types[0]], label=types[0]))
    rec_list_top.append(ax.bar(x - (width * 0.5), scores_lists[types[1]], width, yerr=error_lists[types[1]], label=types[1]))
    rec_list_top.append(ax.bar(x + (width * .5), scores_lists[types[2]], width, yerr=error_lists[types[2]], label=types[2]))
    rec_list_top.append(ax.bar(x + (width * 1.5), scores_lists[types[3]], width, yerr=error_lists[types[3]], label=types[3]))


    # for elem in rec_list:
    #     autolabel(rec_list,ax)

    ax.set_ylabel('Davies-Bouldin average scores')
    ax.set_title('Scores for base set k = 9  tries ={} '.format(n_total_tries))
    ax.set_xticks(x)
    ax.set_xticklabels(['n='+str(i) for i in n_list])
    ax.legend()

    fig.tight_layout()

    plt.show()


def main_k():
    data = get_data_from_picture('in/base_set.png')
    types = ['k-means++', 'forgy', 'random', 'random_partition']
    k_list= [2,3,4,5,6,7,8,9,10,13,15,17,20]
    n_total_tries = 2
    n_base = 10
    scores={}
    for type_name in types:
        scores[type_name]= {}
        for i in k_list:
            scores[type_name][i] = []

    for k in range(n_total_tries):
        for type_name in types:
            print(type_name)
            for i in k_list:
                kmeans = k_means(i,n_base,type_name,data)
                scores[type_name][i].append(ref_val(get_clusters_from_kmeans(kmeans,i,data)))
                print('    n = ' + str(i) + '   score = ' + str(scores[type_name][i]))

    scores_lists = {}
    error_lists= {}

    for type_name in types:
        scores_lists[type_name] = []
        error_lists[type_name] = []
        for j in k_list:
            scores_lists[type_name].append( (np.mean(scores[type_name][j])))
            error_lists[type_name].append(np.std(scores[type_name][j]))



    x = np.array([ 2+(i*6) for i in range(len(k_list))])
    width = 1
    fig,ax = plt.subplots()

    rec_list_top = []
    rec_list_top.append(ax.bar(x - (width*1.5), scores_lists[types[0]], width, yerr=error_lists[types[0]], label=types[0]))
    rec_list_top.append(ax.bar(x - (width * 0.5), scores_lists[types[1]], width, yerr=error_lists[types[1]], label=types[1]))
    rec_list_top.append(ax.bar(x + (width * .5), scores_lists[types[2]], width, yerr=error_lists[types[2]], label=types[2]))
    rec_list_top.append(ax.bar(x + (width * 1.5), scores_lists[types[3]], width, yerr=error_lists[types[3]], label=types[3]))


    # for elem in rec_list:
    #     autolabel(rec_list,ax)

    ax.set_ylabel('Davies-Bouldin average scores')
    ax.set_title('Scores for bad set n for each call = 10; total calls per method = {} '.format(n_total_tries))
    ax.set_xticks(x)
    ax.set_xticklabels(['k='+str(i) for i in k_list])
    ax.legend()

    fig.tight_layout()

    plt.show()

    # for key in scores.keys():
    #     print(key)
    #     for n,value in scores[key].items():
    #         print('    n = '+str(n)+'   score = '+str(value))



if __name__=='__main__':
    # main_n()
    main_k()


