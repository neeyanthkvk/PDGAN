#!/usr/bin/env python
# coding: utf-8
import os
import glob
import numpy as np
import pydicom as pd
from scipy.ndimage import zoom


data_PD_dir = "/data/Imaging/Raw/PD"
data_Control_dir = '/data/Imaging/Raw/Control'
output_PD_dir = "/data/Imaging/3D/PD"
output_Control_dir = "/data/Imaging/3D/Control"
pref_shape = (128, 128, 128)


def reshape(arr):
    new_array = zoom(arr, (pref_shape[0]/arr.shape[0],
                           pref_shape[1]/arr.shape[1],
                           pref_shape[2]/arr.shape[2]))
    return new_array


def gen_numpy_arrays(indir, outdir):
    count_file = 0
    for path in glob.iglob(os.path.join(indir, "*/*/*/*")):
        _, _, _, _, _, pat, modal, date, id_ = tuple(path.split("/"))
        try:
            if len(os.listdir(path)) < 5:
                continue
            dat_shape = None
            echo_data = {}
            for file in os.listdir(path):
                dat = pd.read_file(os.path.join(path, file))
                echo_number = dat[0x18, 0x86].value
                if echo_number not in echo_data:
                    echo_data[echo_number] = {}
                echo_data[echo_number][int(file.split('_')[-3])] = dat.pixel_array
                dat_shape = dat.pixel_array.shape
            for echo in echo_data:
                np_arr = np.zeros((*dat_shape, len(echo_data[echo])))
                for i, (key, value) in enumerate(sorted(echo_data[echo].items())):
                    np_arr[:, :, i] = value
                new_arr = reshape(np_arr)
                np.save(os.path.join(outdir, str(count_file) + ".npy"), new_arr)
                count_file += 1
                if count_file % 10 == 0:
                    print("DEBUG: ", count_file)
        except KeyError:
            print("ERROR: ", path)


if __name__ == "__main__":
    gen_numpy_arrays(data_PD_dir, output_PD_dir)
    gen_numpy_arrays(data_Control_dir, output_Control_dir)
