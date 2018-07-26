import os

data_dir = "/data/PNG/"

for month in os.listdir(data_dir):
    month_dir = data_dir + month + "/"
    for pic in os.listdir(month_dir):
        arr = pic.split("_")
        x = int(arr[1].split(".")[0])
        if(x <= 10 or x >= 119):
            os.system("rm -f " + month_dir + pic)
        
