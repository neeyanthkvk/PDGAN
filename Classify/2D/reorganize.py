import os
import random
c_dir = "/data2/Pictures/Control/"
p_dir = "/data2/Pictures/PD/"

c_train_dir = "/data2/2D/Train/Control/"
c_val_dir = "/data2/2D/Validation/Control"

p_train_dir = "/data2/2D/Train/PD"
p_val_dir = "/data2/2D/Validation/PD"

for pic in os.listdir(c_dir):
    randnum = random.randint(1,100)
    if(randnum <= 70):
        os.system("mv " + c_dir + pic + " " + c_train_dir)
    else:
        os.system("mv " + c_dir + pic + " " + c_val_dir)
for pic in os.listdir(p_dir):
    randnum = random.randint(1,100)
    if(randnum <= 70):
        os.system("mv " + p_dir + pic + " " + p_train_dir)
    else:
        os.system("mv " + p_dir + pic + " " + p_val_dir)
