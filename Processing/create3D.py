from PIL import Image
import numpy as np
from scipy import misc
import os
import pickle
PD_DIR = '/data/data1/PD'

# PD GROUP
ar = []
count500 = 0
for i in range(2500):
    if i % 100 == 0:
        print(i)
    if i != 0 and i % 100 == 0:
        pickle.dump(ar, open("/data2/PD/PDP_" + str(count500) + ".pkl","wb"))
        count500 += 1
        ar = []
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = open("len.txt","r")
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = np.zeros((1))
    fir = True
    idx = 0
    for line in reader:
        temp = Image.open(line[:-1])
        arr = np.array(temp)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1]))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    ar.append(dArr)

pickle.dump(ar, open("/data2/PD/PDP.pkl", "wb"))

CONTROL_DIR = '/data/data1/Control'

# CONTROL GROUP
count500 = 0
ar = []
for i in range(675):
    if i != 0 and i % 100 == 0:
        pickle.dump(ar, open("/data2/Control/ControlP_" + str(count500) + ".pkl","wb"))
        count500 += 1
        ar = []

    os.system("find " + CONTROL_DIR+ " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + CONTRL_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = open("len.txt","r")
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = np.zeros((1))
    fir = True
    idx = 0
    for line in reader:
        temp = Image.open(line[:-1])
        arr = np.array(temp)
        if fir:
            fir = False
            dArr = np.zeros((l, arr.shape[0], arr.shape[1]))
        dArr[idx] = arr
        idx += 1
    os.system("rm temp.txt")
    os.system("rm len.txt")
    ar.append(dArr)

pickle.dump(ar, open("/data2/Control/ControlP.pkl", "wb"))
