from PIL import Image
import numpy as np
from scipy import misc
import os
import pickle
PD_DIR = '/data/data1/PD'
d = {}

# PD GROUP
for i in range(2500):
    if i % 100 == 0:
        print(i)
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + PD_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = open("len.txt","r")
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = 1
    fir = True
    idx = 0
    for line in reader:
        temp = Image.open(line[:-1])
        arr = np.array(temp)
        dArr = (arr.shape[0], arr.shape[1], l)
        break
    os.system("rm temp.txt")
    os.system("rm len.txt")
    
    if(dArr in d):
        d[dArr] += 1
    else:
        d[dArr] = 1
print(d)
CONTROL_DIR = '/data/data1/Control'
d = {}
# CONTROL GROUP
for i in range(675):
    os.system("find " + CONTROL_DIR+ " -name '"+ str(i) + "_*' > temp.txt")
    os.system("find " + CONTROL_DIR + " -name '"+ str(i) + "_*' | wc -l > len.txt")
    reader = open("len.txt","r")
    l = 0
    for line in reader:
        l = int(line[:-1])
    reader = open("temp.txt","r")
    dArr = 1
    fir = True
    idx = 0
    for line in reader:
        temp = Image.open(line[:-1])
        arr = np.array(temp)
        dArr = (arr.shape[0], arr.shape[1], l)
        break
    os.system("rm temp.txt")
    os.system("rm len.txt")
    if(dArr in d):
        d[dArr] += 1
    else:
        d[dArr] = 1
print(d)
