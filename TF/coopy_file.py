import os
import shutil
import os.path

sourse = '/home/cvlab/Desktop/Dataset/DIV2K/DIV2K_train_LR_bicubic/train2014 1960_hr'
target = '/home/cvlab/Desktop/Dataset/DIV2K/DIV2K_train_LR_bicubic/train2014 1960_lr'
for dirpath, dirnames, filenames in os.walk(sourse):
    for filename in filenames:
        if filename[-6:] == 'lr.jpg':
            # os.remove(sourse+'/'+filename)  刪除資料夾下所有ｌｒ檔案
            shutil.copy(sourse+'/'+filename, target)
