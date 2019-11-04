import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import random






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

def k_fun_n1_euk(data,coord):
    x,y=coord
    min_dist = euk_dist(coord,(data[0][0],data[0][1]))
    best_class = data[0][2]
    for elem in data:
        dist= euk_dist(coord,(elem[0],elem[1]))
        if dist < min_dist:
            min_dist=dist
            best_class= elem[2]
    return best_class










def main():
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


if __name__=="__main__":
    main()
