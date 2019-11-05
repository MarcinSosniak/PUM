import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import random


class Test_res:
    def __init__(self):
        self.false_positive=0
        self.false_negative=0
        self.true_positive=0
        self.true_negative=0

    def accuracy(self):
        return (self.true_negative+self.true_positive) / (self.true_negative
                                                          +self.true_positive
                                                          +self.false_negative
                                                          +self.false_positive)

    def recall(self):
        return self.true_positive/(self.true_positive + self.false_negative)

    def precision(self):
        return self.true_positive/(self.true_positive + self.false_positive)

    def f_score(self):
        rec = self.recall()
        prec = self.precision()
        return (2*rec*prec)/(rec+prec)

    def conf_matrix(self):
        arr=np.zeros((2,2),dtype=np.uint64)
        arr[0][0]=self.true_positive
        arr[0][1]=self.false_positive
        arr[1][0]=self.false_negative
        arr[1][1]=self.true_negative
        return arr


class Mah_dist:
    def __init__(self,class_data):
        arr=np.zeros((2,len(class_data)))
        for i,elem in enumerate(class_data):
            arr[0][i]=elem[0]
            arr[1][i]=elem[1]
        self.C_1 = np.linalg.inv(np.cov(arr))

    def dist(self, coord1, coord2):
        X = np.zeros((2,))
        X[0] = coord1[0]
        X[1] = coord1[1]
        Y = np.zeros((2,))
        Y[0] = coord2[0]
        Y[1] = coord2[1]
        # print(self.C_1.shape)
        ret_sq = np.dot(np.dot(np.subtract(X, Y).T, self.C_1), np.subtract(X, Y))
        # print(ret_sq.shape)
        return math.sqrt(ret_sq)


def get_class_data(data,class_no):
    out=[]
    for elem in data:
        if elem[2]==class_no:
            out.append((elem[0],elem[1]))
    return out


def get_data_from_picture(path):
    img = Image.open(path,'r')
    arr=np.array(img)
    out=[]
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x][y][0]==255:
                out.append((float(x)+random.uniform(0,1), float(y)+random.uniform(0,1),0))
            elif arr[x][y][1]==255:
                out.append((float(x)+random.uniform(0,1), float(y)+random.uniform(0,1),1))
            elif arr[x][y][2] == 255:
                    out.append((float(x)+random.uniform(0,1), float(y)+random.uniform(0,1), 2))
    return out,arr.shape


def make_class(data,class_no,shape,k_fun, n=50000): # shape with colors so (x,y,3)
    out= np.zeros(shape,dtype=np.uint8)
    for elem in data:
        if elem[2]==class_no:
            out[int(elem[0])][int(elem[1])][1] = 255

    for i in range(n):
        x = random.randrange(0,shape[0])
        org_x=x
        x += random.uniform(0,1)
        y = random.randrange(0,shape[1])
        org_y=y
        y += random.uniform(0,1)
        if class_no== k_fun(data,(x,y)):
            try:
                out[int(x)][int(y)][0] = 255
            except:
                pass
    return out

def euk_dist(coord1,coord2):
    return math.sqrt(((coord1[0]-coord2[0])*(coord1[0]-coord2[0]))+((coord1[1]-coord2[1])*(coord1[1]-coord2[1])))

def smollest_id(list_in,key=lambda x : x):
    out_id = 0
    min_v = key(list_in[0])
    for i in range(len(list_in)):
        if key(list_in[i]) < min_v:
            min_v=key(list_in[i])
            out_id=i
    return min_v,out_id


def biggest_id(list_in,key=lambda x : x):
    out_id = 0
    max_v = key(list_in[0])
    for i in range(len(list_in)):
        if key(list_in[i]) > max_v:
            max_v=key(list_in[i])
            out_id=i
    return max_v,out_id



def k_fun_n1_euk(data,coord,dist_obj):
    x,y=coord
    min_dist = euk_dist(coord,(data[0][0],data[0][1]))
    best_class = data[0][2]
    for elem in data:
        dist= euk_dist(coord,(elem[0],elem[1]))
        if dist < min_dist:
            min_dist=dist
            best_class= elem[2]
    return best_class


def k_fun_n7_euk(data,coord,dist_obj):
    min_dist_and_class_list=[[-1,-1] for i in range(7)]
    for elem in data:
        dist= euk_dist(coord,(elem[0],elem[1]))
        max_v,id= biggest_id(min_dist_and_class_list,key= lambda x : x[0])
        if dist < max_v or max_v==-1:
            min_dist_and_class_list[id][0]=dist
            min_dist_and_class_list[id][1]=elem[2]
    classes=[0 for i in range(3)]
    for elem in min_dist_and_class_list:
        classes[elem[1]] += 1
    void,id =biggest_id(classes,key=lambda  x: x)
    return id


def k_fun_n7_euk_weight(data,coord,dist_obj):
    min_dist_and_class_list=[[-1,-1] for i in range(7)]
    for elem in data:
        dist= euk_dist(coord,(elem[0],elem[1]))
        max_v,id= biggest_id(min_dist_and_class_list,key= lambda x : x[0])
        if dist < max_v or max_v==-1:
            min_dist_and_class_list[id][0]=dist
            min_dist_and_class_list[id][1]=elem[2]
    classes=[0 for i in range(3)]
    max_v,id_v = smollest_id(min_dist_and_class_list,key= lambda x : x[0])
    for elem in min_dist_and_class_list:
        classes[elem[1]] += max_v/elem[0] # weight
    max_v,id_v = biggest_id(classes,key=lambda x: x)
    return id_v


def k_fun_n1_mah(data,coord,dist_obj):
    best_class= data[0][2]
    min_dist = dist_obj.dist((data[0][0],data[0][1]),coord)
    for elem in data:
        dist=  dist_obj.dist((elem[0],elem[1]),coord)
        if dist < min_dist:
            min_dist=dist
            best_class=elem[2]
    return best_class


def k_fun_n7_mah_weigh(data,coord,dist_obj):
    min_dist_and_class_list = [[-1, -1] for i in range(7)]
    for elem in data:
        dist = dist_obj.dist(coord, (elem[0], elem[1]))
        max_v, id = biggest_id(min_dist_and_class_list, key=lambda x: x[0])
        if dist < max_v or max_v==-1:
            min_dist_and_class_list[id][0] = dist
            min_dist_and_class_list[id][1] = elem[2]
    classes = [0 for i in range(3)]
    max_v, id_v = smollest_id(min_dist_and_class_list, key=lambda x: x[0])
    for elem in min_dist_and_class_list:
        classes[elem[1]] += max_v / elem[0]  # weight
    max_v, id_v = biggest_id(classes, key=lambda x: x)
    return id_v










def splilt_data(data):
    base = []
    test = []
    for elem in data:
        if random.uniform(0.,1.0) < 0.5:
            base.append(elem)
        else:
            test.append(elem)
    return base,test


def run_test(data,k_fun_list_euk,k_fun_list_mah):
    test_euk = [Test_res() for i in range(len(k_fun_list_euk))]
    base,test = splilt_data(data)
    for k_fun,test_obj in zip(k_fun_list_euk,test_euk):
        for cat in range(3):
            for elem in test:
                ret= k_fun(base, (elem[0],elem[1]),None)
                if ret==cat:
                    if ret==elem[2]:
                        test_obj.true_positive += 1
                    else:
                        test_obj.false_positive += 1
                else:  # ret!=cat
                    if ret==elem[2]:
                        test_obj.false_negative += 1
                    else:  # ret!=elem
                        test_obj.true_negative += 1
    test_mah= [Test_res() for i in range(len(k_fun_list_mah))]
    for k_fun,test_obj in zip(k_fun_list_mah,test_mah):
        for cat in range(3):
            dist_obj= Mah_dist(get_class_data(base,cat))
            for elem in test:
                ret= k_fun(base, (elem[0],elem[1]),dist_obj)
                if ret==cat:
                    if ret==elem[2]:
                        test_obj.true_positive += 1
                    else:
                        test_obj.false_positive += 1
                else:  # ret!=cat
                    if ret==elem[2]:
                        test_obj.false_negative += 1
                    else:  # ret!=elem
                        test_obj.true_negative += 1
    test_euk.extend(test_mah)
    return test_euk








def main_base_test():
    path_list= ['sets/set1.png','sets/set2.png','sets/set3.png']
    for path in path_list:
        print(path)
        max_v=0
        min_v=1
        data,shape= get_data_from_picture(path)
        test_res= run_test(data,[k_fun_n1_euk,k_fun_n7_euk,k_fun_n7_euk_weight],[k_fun_n1_mah,k_fun_n7_mah_weigh])
        for i,test in enumerate(test_res):
            print("  {}".format(i))
            print("    accuracy: {}".format(test.accuracy()))
            # print("  recall: {}".format(test.recall()))
            # print("  precision: {}".format(test.precision()))
            # print("  f_score: {}".format(test.f_score()))
            print(test.conf_matrix())
            acc = test.accuracy()
            if acc > max_v:
                max_v = acc
            if acc < min_v:
                min_v = acc
        print("  max diff = {}".format(max_v-min_v))


    pass


def main_print():
    data,shape = get_data_from_picture('sets/set2.png')
    out = make_class(data,0,shape,k_fun_n1_euk)
    plt.imshow(out)
    plt.show()
    out = make_class(data, 1, shape, k_fun_n1_euk)
    plt.imshow(out)
    plt.show()
    out = make_class(data, 2, shape, k_fun_n1_euk)
    plt.imshow(out)
    plt.show()
    pass


def full_test()


if __name__=="__main__":
    # main_print()
    main_base_test()
