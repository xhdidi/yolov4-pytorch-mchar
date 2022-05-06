"""
作者：didi
日期：2022年05月06日
"""


import os
import random
import numpy as np
import codecs
import pandas as pd
import json
from glob import glob
import xml.etree.ElementTree as ET
import cv2
import shutil
from sklearn.model_selection import train_test_split
from IPython import embed


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def csv2voc(csv_file, set, saved_path):
    '''
    convert from csv to the style yolov4 need
    '''
    # read csv
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file, header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value

    # save xml
    for filename, label in total_csv_annotations.items():
        imgname = saved_path + f"JPEGImages/VOC2007_{set}/" + filename.split('/')[-1]
        height, width, channels = cv2.imread(imgname).shape

        xmlname = saved_path + f"Annotations/VOC2007_{set}/" + filename.split('/')[-1].replace(".png", ".xml")
        with open(xmlname, "w") as xml:
            xml.write('<annotation>\n')
            # xml.write('\t<folder>' + 'mchar' + '</folder>\n')
            # xml.write('\t<filename>' + filename + '</filename>\n')
            # xml.write('\t<source>\n')
            # xml.write('\t\t<database>mchar</database>\n')
            # xml.write('\t\t<annotation>mchar</annotation>\n')
            # xml.write('\t\t<image>flickr</image>\n')
            # xml.write('\t\t<flickrid>NULL</flickrid>\n')
            # xml.write('\t</source>\n')
            # xml.write('\t<owner>\n')
            # xml.write('\t\t<flickrid>NULL</flickrid>\n')
            # xml.write('\t\t<name>ChaojieZhu</name>\n')
            # xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            if isinstance(label, float):
                xml.write('</annotation>')
                continue
            for label_detail in label:
                labels = label_detail
                if int(labels[2]) <= int(labels[0]):
                    pass
                elif int(labels[3]) <= int(labels[1]):
                    pass
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + str(labels[4]) + '</name>\n')
                    # xml.write('\t\t<pose>Unspecified</pose>\n')
                    # xml.write('\t\t<truncated>1</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(labels[0]) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(labels[1]) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(labels[2]) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(labels[3]) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')


def voc2yolo4(set, classes, xmlfilepath, saveBasePath):
    # read xml
    temp_xml = os.listdir(os.join(xmlfilepath, f"VOC2007_{set}"))
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
    print(f"totally {len(total_xml)} data")

    # save txt file
    list_file = open(os.path.join(saveBasePath, f'{set}.txt'), 'w')

    for xml in total_xml:
        in_file = open(f'VOC2007/Annotations/VOC2007_{set}/{xml}', encoding='UTF-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        list_file.write(f'VOC2007/ImageSets/2007_{set}.txt')

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        list_file.write('\n')


if __name__ == '__main__':

    saved_path = 'VOC2007/'
    setting = ['train', 'val']
    classes_path = 'model_data/voc_classes.txt'
    xmlfilepath = 'VOC2007/Annotations'
    saveBasePath = 'VOC2007/ImageSets'

    classes = get_classes(classes_path)

    for set in setting:
        csv_file = f"VOC2007/mchar_{set}.csv"
        csv2voc(csv_file, set, saved_path)
        voc2yolo4(set, classes, xmlfilepath, saveBasePath)

