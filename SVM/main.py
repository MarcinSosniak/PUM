
import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import random

from sklearn.svm import SVC



def get_data_from_picture(path):
    img = Image.open(path,'r')
    arr = np.array(img)
    class1 = []
    class2 = []
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x][y][0]>128:
                class1.append([x,y])
            elif arr[x][y][2] >128:
                class2.append([x,y])
    return class1, class2, arr.shape


def find_distance_to_nearest_differnt(arr,x,y,max_dist):
    min_dist2 = arr.shape[0]**2 +arr.shape[1]**2
    for xi in range(arr.shape[0]):
        for yi in range(arr.shape[1]):
            if not (arr[x][y][0]==arr[xi][yi][0] and arr[x][y][1]== arr[xi][yi][1] and arr[x][y][2] == arr[xi][yi][2]):
                dist2=(x-xi)**2 + (y-yi)**2
                if dist2< min_dist2:
                    min_dist2=dist2
    return math.sqrt(min_dist2)



    # for d in range(int(max_dist+1)):
    #     for s_x in range(x-d,x+d+1):
    #         if s_x < 0 or s_x >= arr.shape[0]:
    #             continue
    #         # y' = y +- sqrt(d-(x'-x)^2)
    #         delta = math.sqrt(d**2 -(s_x-x)**2)
    #         for s_y in [int(y+ delta),int(y- delta)]:
    #             if s_y < 0 or s_y >= arr.shape[1]:
    #                 continue
    #             if s_x==0 and s_y==40:
    #                 # print("br here")
    #                 pass
    #             if not (arr[x][y][0]
    #                     ==
    #                     arr[s_x][s_y][0] and
    #                     arr[x][y][1]==
    #                     arr[s_x][s_y][1] and
    #                     arr[x][y][2] ==
    #                     arr[s_x][s_y][2]):
    #                 out = math.sqrt((x-s_x)**2 + (x-s_y)**2)
    #                 return out
    #

# tax metric
def apply_range_filter(arr):
    max_dist = math.sqrt(arr.shape[0]**2+arr.shape[1]**2)
    out =  np.zeros(arr.shape,dtype=np.uint8)
    dist_matrix = np.zeros((arr.shape[0],arr.shape[1]),dtype=float)
    max_dist_2=0
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            dist = find_distance_to_nearest_differnt(arr,x,y,max_dist)
            dist_matrix[x][y]=dist
            if dist> max_dist_2:
                max_dist_2=dist
            # out[x][y][0] = int(arr[x][y][0] * (1 - (dist / max_dist)))
            # out[x][y][1] = int(arr[x][y][1] * (1 - (dist / max_dist)))
            # out[x][y][2] = int(arr[x][y][2] * (1 - (dist / max_dist)))

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            out[x][y][0] = int(arr[x][y][0] * (1 - (dist_matrix[x][y] / max_dist_2)))
            out[x][y][1] = int(arr[x][y][1] * (1 - (dist_matrix[x][y] / max_dist_2)))
            out[x][y][2] = int(arr[x][y][2] * (1 - (dist_matrix[x][y] / max_dist_2)))
    return out


def svm(classes,classes_nr,kernel,shape,c_v=1,gamma_v =None):
    svc_obj = None
    if gamma_v is not None:
        svc_obj=SVC(kernel=kernel,C=c_v,gamma=gamma_v)
    else:
        svc_obj = SVC(kernel=kernel, C=c_v)
    svc_obj.fit(classes, classes_nr)
    amount = len(classes)
    right = 0
    predicted = svc_obj.predict(classes)
    for pred, accurate in zip(predicted, classes_nr):
        if pred == accurate:
            right += 1
    accuracy = (right*100)/amount
    show_arr = np.zeros((shape[0],shape[1],3))
    to_predict = []
    for x in range(shape[0]) :
        for y in range(shape[1]):
            to_predict.append([x,y])
    predicted_classes= svc_obj.predict(to_predict)
    # print(predicted_classes)
    i=0
    for x in range(shape[0]) :
        for y in range(shape[1]):
            if predicted_classes[i] == 1:
                show_arr[x][y][0]=255
                show_arr[x][y][1]=0
                show_arr[x][y][2]=0
            elif predicted_classes[i] == 2:
                show_arr[x][y][0] = 0
                show_arr[x][y][1] = 0
                show_arr[x][y][2] = 255
            i+=1

    return accuracy,apply_range_filter(show_arr)


def map_char(c):
    if c=='.':
        return '_'
    if c==' ':
        return '__'
    return c


def fulltest():
    class1, class2, shape = get_data_from_picture("in/set.png")
    classes = list(class1)
    classes.extend(class2)
    classes_nr = [1 for elem in class1]
    classes_nr.extend([2 for elem in class2])

    c_list = [0.1,0.3,0.7,1,1.2,1.6,2.0,2.5,3.0,4.0,5.0]
    kernel_list = ['linear','poly','rbf']
    gamma_list = [0.1,0.2,0.35,0.5,0.65,0.8,1.1]


    retries = 3
    with open('out/stat.txt','w+') as f:
        for kernel in kernel_list:
            print(kernel)
            print(kernel, file=f)
            for gamma in gamma_list:
                print("  gamma: "+str(gamma))
                print("  gamma: "+str(gamma),file=f)
                for c in c_list:
                    print("    c: "+str(c))
                    print("    c: "+str(c),file=f)
                    print("      ",end='')
                    print("      ",end='',file=f)
                    for i in range(retries):
                        acc, show_arr =None,None
                        if not kernel=='linear':
                            acc,show_arr= svm(classes, classes_nr, kernel, shape,c_v=c,gamma_v=gamma)
                        else:
                            acc,show_arr= svm(classes, classes_nr, kernel, shape,c_v=c)
                        print(acc,end=' ')
                        print(acc,end=' ',file=f)
                        filename= "{} gamma {} c {} i {}".format(kernel,gamma,c,i)
                        filename = ''.join(list(map(map_char,filename)))
                        filename = 'out/' + filename
                        plt.imshow(show_arr)
                        plt.savefig(filename)
                    print('')
                    print('',file=f)





def main():
    class1,class2,shape = get_data_from_picture("in/set.png")
    classes = list(class1)
    classes.extend(class2)
    classes_nr = [1 for elem in class1]
    classes_nr.extend([2 for elem in class2])
    acc,show_arr = svm(classes,classes_nr,'linear',shape)
    print("accuracy = {}%".format(acc))
    print(show_arr)
    plt.imshow(show_arr)
    plt.show()

    # print(len(classes))
    # print(len(classes_nr))
    # svc_obj = SVC(kernel='linear')
    # svc_obj.fit(classes,classes_nr)
    # amount = len(classes)
    # right = 0
    # predicted = svc_obj.predict(classes)
    # for pred,accurate in zip(predicted,classes_nr):
    #     if pred==accurate:
    #         right+=1
    #
    # print()
    # print( "accuracy = {}%".format((right*100)/amount) )


    pass


if __name__=="__main__":
    # main()
    fulltest()
    pass
