import os
import cv2
from PIL import Image
import numpy as np
import pickle
import time

PD_DIR = "/data/2D/PD/"
CONTROL_DIR = "/data/2D/Control/"

ar = np.zeros((466, 176, 32, 30, 1))
count = 0
for i in range(2500):
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = 0
    time.sleep(0.2)
    if(os.path.isfile("len.txt")):
        reader = open("len.txt","r")
    else:
        continue
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = np.zeros((1))
    fir = True
    idx = 0
    for line in reader:
        arr = cv2.imread(line[:-1], cv2.IMREAD_GRAYSCALE)
        arr = cv2.resize(arr, (0,0), fx=0.125, fy=0.125)
        arr = np.expand_dims(arr, axis=-1)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1], 1))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    if(dArr.shape == (176, 32, 30, 1)):
        ar[count] = dArr
        count += 1
        print(count)
np.save("/data/PD.npy", ar)


count = 0
ar2 = np.zeros((148, 176, 32, 30, 1))
for i in range(675):
    os.system("find " + CONTROL_DIR+ " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + CONTROL_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = 0
    time.sleep(0.2)
    if(os.path.isfile("len.txt")):
        reader = open("len.txt","r")
    else:
        continue
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = np.zeros((1))
    fir = True
    idx = 0
    for line in reader:
        arr = cv2.imread(line[:-1], cv2.IMREAD_GRAYSCALE)
        arr = cv2.resize(arr, (0,0), fx=0.125, fy=0.125)
        arr = np.expand_dims(arr, axis=-1)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1], 1))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    if(dArr.shape == (176, 32, 30, 1)):
        ar2[count] = dArr
        count += 1
        print(count)
np.save("/data/Control.npy", ar2)
