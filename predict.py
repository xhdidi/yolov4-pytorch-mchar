<<<<<<< HEAD
"""
作者：didi
日期：2022年05月06日
"""

'''
predict the test-dataset and generate a csv filr in required format
'''
import os
import csv
from yolo import YOLO
from PIL import Image


yolo = YOLO(model_path='logs/weights/FPN/Epoch_123.pth')

save_path = 'logs/detect_result/'
save_csv_file = save_path + 'result.csv'
with open(save_csv_file, 'a+', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(('file_name', 'file_code'))

file = 'VOC2007/JPEGImages/VOC2007_test/'
for idx, filename in enumerate(os.listdir(file)):
    img = file + filename
    print(f"Pic {idx}, {filename} detecting...")
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        yolo.detect_image(image, filename, save_path)

=======
"""
作者：didi
日期：2022年05月06日
"""

'''
predict the test-dataset and generate a csv filr in required format
'''
import os
import csv
from yolo import YOLO
from PIL import Image


yolo = YOLO(model_path='logs/weights/FPN/Epoch_123.pth')

save_path = 'logs/detect_result/'
save_csv_file = save_path + 'result.csv'
with open(save_csv_file, 'a+', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(('file_name', 'file_code'))

file = 'VOC2007/JPEGImages/VOC2007_test/'
for idx, filename in enumerate(os.listdir(file)):
    img = file + filename
    print(f"Pic {idx}, {filename} detecting...")
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        yolo.detect_image(image, filename, save_path)

>>>>>>> origin/main
