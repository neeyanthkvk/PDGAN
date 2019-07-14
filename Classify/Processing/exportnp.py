#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import glob
import time
import numpy as np
import scipy
from scipy import misc
import pydicom as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


data_PD_dir = "/data/Imaging/Raw/PD"
data_Control_dir = '/data/Imaging/Raw/Control'
output_PD_dir = "/data/Imaging/3D/PD"
output_Control_dir = "/data/Imaging/3D/Control"
types = ['t2', 't1', 'pd']
pref_shape = (32, 32, 32)

def reshape(arr):
    new_array = zoom(arr, (pref_shape[0]/arr.shape[0], pref_shape[1]/arr.shape[1], pref_shape[2]/arr.shape[2]))
    return new_array

def gen_numpy_arrays(indir, outdir):
    count_file = 0
    for path in glob.iglob(os.path.join(indir,"*/*/*/*")):
        _,_,_,_,_,pat,modal,date,id = tuple(path.split("/"))
        try:
            if(len(os.listdir(path)) < 5):
                continue
            datShape = None
            temp = {}
            for file in os.listdir(path):
                dat = pd.read_file(os.path.join(path, file))
                echo_number = dat[0x18,0x86].value
                if echo_number not in temp:
                    temp[echo_number] = {}
                temp[echo_number][int(file.split('_')[-3])] = dat.pixel_array
                datShape = dat.pixel_array.shape
            for echo in temp:
                np_arr = np.zeros((*datShape, len(temp[echo])))
                for i, (key, value) in enumerate(sorted(temp[echo].items())):
                    np_arr[:,:,i] = value
                new_arr = reshape(np_arr)
                np.save(os.path.join(outdir, str(count_file) + ".npy"), new_arr)
                count_file += 1
        except KeyError:
            print("ERROR: ", path)

