import os

data_dir = "/data/PD/"
finals_dir = "/data/data1/PD/"
count = 0
for monthname in os.listdir(data_dir):
    month_dir = data_dir + monthname + "/"
    for patient in os.listdir(month_dir):
        patient_dir = month_dir + patient + "/"
        for ftype in os.listdir(patient_dir):
            f_dir = patient_dir + ftype + "/"
            for date in os.listdir(f_dir):
                date_dir = f_dir + date + "/"
                for some in os.listdir(date_dir):
                    final_dir = date_dir + some + "/"
                    for mri in os.listdir(final_dir):
                        temp = mri.split("_")
                        sl = temp[len(temp)-3]
                        os.system("convert " + final_dir + mri + " " + finals_dir + str(count) + "_" + sl + ".png")
                        #os.system("temp=$(ls -1 " + final_dir + " | grep _" + str(num+1) + "_)")
                        #os.system("echo $temp")
                        #os.system("convert " + final_dir + "$temp " + final_dir + str(num) + ".png")
                    count += 1
