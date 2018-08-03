from PIL import Image
import numpy as np
from scipy import misc
import os
import pickle
import cv2
import time

PD_DIR = '/data/data1/PD'

# PD GROUP
ar = np.zeros((466, 176, 256, 240, 1))
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
        arr = np.expand_dims(arr, axis=-1)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1], 1))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    if(dArr.shape == (176, 256, 240, 1)):
        ar[count] = dArr
        count += 1
np.save("/data2/3D/PD.npy", ar)

CONTROL_DIR = '/data/data1/Control'

# CONTROL GROUP
count = 0
ar = np.zeros((148, 176, 256, 240, 1))
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
        arr = np.expand_dims(arr, axis=-1)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1], 1))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    if(dArr.shape == (176, 256, 240, 1)):
        ar[count] = dArr
        count += 1
np.save("/data2/3D/Control.npy", ar)
