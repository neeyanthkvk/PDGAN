import os
import random
c_dir = "/data/data1/Control/"
p_dir = "/data/data1/PD/"

c_train_dir = "/data/data1/Train/Control/"
c_val_dir = "/data/data1/Validation/Control"
c_test_dir = "/data/data1/Test/Control"

p_train_dir = "/data/data1/Train/PD"
p_val_dir = "/data/data1/Validation/PD"
p_test_dir = "/data/data1/Test/PD"

for pic in os.listdir(c_dir):
    randnum = random.randint(1,100)
    if(randnum <= 70):
        os.system("mv " + c_dir + pic + " " + c_train_dir)
    elif(randnum <= 85):
        os.system("mv " + c_dir + pic + " " + c_val_dir)
    else:
        os.system("mv " + c_dir + pic + " " + c_test_dir)
for pic in os.listdir(p_dir):
    randnum = random.randint(1,100)
    if(randnum <= 70):
        os.system("mv " + p_dir + pic + " " + p_train_dir)
    elif(randnum <= 85):
        os.system("mv " + p_dir + pic + " " + p_val_dir)
    else:
        os.system("mv " + p_dir + pic + " " + p_test_dir)
