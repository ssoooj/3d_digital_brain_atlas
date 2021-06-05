import os
import sys
import json
import glob
import pathlib
import traceback
import multiprocessing
import scipy.io
import scipy.ndimage
import natsort
import itertools
import lap
import cv2
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

from collections import Counter
from itertools import islice
from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from utils import shading_correction

def region_info(dict_file, region):
    reg_dict = dict_file['Regions']  #all the region keys in the file

    for _ in reg_dict:
        if region in reg_dict:
            print("Region : ", region,",", reg_dict[region]['Abbreviation'])
            parent_id = int(reg_dict[region]['ParentID'])
            print("Parent : ", parent_id, ",", reg_dict[parent_id]['Abbreviation'])
            print("Color : ", reg_dict[region]['Color'])
            if reg_dict[region]['ROI'] == None:
                print("Slice : None")
            else:
                print("Slice number : ", reg_dict[region]['ROI']['SliceROIs'].keys())
                print("Total slice number : {} ".format(len(reg_dict[region]['ROI']['SliceROIs'])))
            # print("Total Slice Number : ", reg_dict[reg_key]['ROI']['SliceNumber']) #int
            cut_roi_yn = [i for i in reg_dict[region] if -1 in reg_dict[region]]
            print(f'Cut line line exists in this dict : {bool(cut_roi_yn)}, {len(cut_roi_yn)},\n{cut_roi_yn}')
                # print(reg_dict[reg_key]['ROI']['SliceROIs']) #dict
        else:
            print("The region doesn't exist")

def dict_info(dict_file, dict2, str2, outfile):
    dict2['Regions'][-1] = copy.deepcopy(dict_file['Regions'][-1])
    dict2['Regions'][-1]['ROI']['SliceROIs'] = {}
    for region in dict_file['Regions']:
        subregion_list = [1929, 1939, 1949, 1959, 1969, 1979, 4779, 4778, 4777, 4776, 4775, 4774, 4773, 3159, 3158, 3157, 3156, 3155, 315]
        if dict_file['Regions'][region]['ROI'] == None:
            continue
        for slice in dict_file['Regions'][region]['ROI']['SliceROIs'].keys():
            for shape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])):
                for subshape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                    line = dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Type']
                    isadd = dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['IsAdd']
                    print(f'region : {region} , slice : {slice} , shape : {shape} , subshape : {subshape} , type : {line} , isadd : {isadd}')
                    if line == 'Line':
                        if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                            dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                        dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]])
                    elif isadd and region in subregion_list:
                        dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Type'] = 'Line'
                        if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                            dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                        dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]])
                    else:
                        continue

    region_annotation.write_region_annotation_dict(dict2, str2)
    ra_dict2 = region_annotation.map_region_annotation_dict_slices(dict2, lambda s: s * 2)
    region_annotation.write_region_annotation_dict(ra_dict2, outfile)

def print_region(dict_file):
    for region in dict_file['Regions']:
        regionname = dict_file['Regions'][region]['Abbreviation']
        if dict_file['Regions'][region]['ROI'] == None:
            continue
        for slice in dict_file['Regions'][region]['ROI']['SliceROIs'].keys():
            # print (f'region : {regionname}, slice : {slice}')
            if regionname == 'Undefined':
                a = dict_file['Regions'][region]['ROI']['SliceROIs'][slice]
                b = len(dict_file['Regions'][region]['ROI']['SliceROIs'])
                if slice >= 0:
                    print(f'\nSlice num : {slice}')
                    for shape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])):
                        for subshape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                            c = dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Points']
                            if len(c) < 30:
                                # index = c[:]
                                # np.delete(c, index)
                                print(f'Coordinates : {c}')
                            else:
                                print("None")
    # region_annotation.write_region_annotation_dict(dict_file, str_file)

def map_regions(dict1, dict2, str2):
    for region in dict1['Regions']:
        if dict1['Regions'][region]['ROI'] == None:
            continue
        for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
            if slice not in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
                continue
            for shape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice])):
                for subshape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                    p1 = dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Points']
                    p2 = dict2['Regions'][-1]['ROI']['SliceROIs'][slice][shape][subshape]['Points']
                    if p1.any() == p2.any():
                        dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict1['Regions'][region]['ROI']['SliceROIs'][slice]])
                    else:
                        print("N/A")
        region_annotation.write_region_annotation_dict(dict2, str2)

def merge_regions(dict1, dict2, str3):
    # dict2['Regions'][-1] = copy.deepcopy(dict1['Regions'][-1])
    # dict2['Regions'][-1]['ROI']['SliceROIs'] = {}
    for region in dict1['Regions']:
        if dict1['Regions'][region]['ROI'] == None:
            continue
        for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
            for shape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice])):
                for subshape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                    line = dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Type']
                    if line == 'Line':
                        if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                            dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                        dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]])
            for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
                if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                    dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]])
    region_annotation.write_region_annotation_dict(dict2, str3)


def subregion_line_copy(dict1, dict2, str3):
    dict2['Regions'][-1] = copy.deepcopy(dict1['Regions'][-1])
    dict2['Regions'][-1]['ROI']['SliceROIs'] = {}

    for region in dict1['Regions']:
        bigregion_list = []
        if dict1['Regions'][region]['ROI'] == None:
            continue
        for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
            for shape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice])):
                for subshape in range(0, len(dict1['Regions'][region]['ROi']['SliceROIs'][slice][shape])):
                    line = dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]['Type']
                    if line == 'Line':
                        if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                            dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                    dict2['Regions'][-1]['ROI']['SliceROIs'][slice].append([dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]])
    region_annotation.write_region_annotation_dict(dict2, str3)

def copy_lines(dict1, dict2, str3):
    # slicelist4 = [95, 103, 111, 119, 125, 133, 139, 147]
    slicelist2 = [0, 6, 12, 18, 22, 26, 30, 34, 38, 40, 42, 44, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 88]
    for region in dict1['Regions']:
        if dict1['Regions'][region]['ROI'] == None:
            for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
                if slice in slicelist2:
                    if slice not in dict2['Regions'][-1]['ROI']['SliceROIs']:
                        dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = []
                    dict2['Regions'][-1]['ROI']['SliceROIs'][slice] = dict1['Regions'][region]['ROI']['SliceROIs'][slice]
    region_annotation.write_region_annotation_dict(dict2, str3)


def bigstructure_merge(dict1, dict2, str3):
    list1 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35,
             36, 37, 39, 41, 43, 45, 47, 48, 49, 51, 52, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164]

    list3 = [92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118,
             120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145,
             146, 148, 149, 150]

    both_list = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35,
             36, 37, 39, 41, 43, 45, 47, 48, 49, 51, 52, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116,
                 117, 118,120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 140, 141, 142,
                 143, 144, 145, 146, 148, 149, 150]

    # regionlist = [315, 1089, 1929, 1939, 1949, 1959, 1969, 1979, 477, 4779, 4778, 803, 4777, 4776, 1129, 549,
    #               1097, 4775, 313, 4774, 4773, 771, 354, 512, 1009, 9101, 9102, 9103]

    for region in dict1['Regions']:
        if dict1['Regions'][region]['ROI'] == None:
            continue
        for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
            if slice in both_list:
                if slice not in dict2['Regions'][region]['ROI']['SliceROIs']:
                    dict2['Regions'][region]['ROI']['SliceROIs'][slice] = []
                dict2['Regions'][region]['ROI']['SliceROIs'][slice] = dict1['Regions'][region]['ROI']['SliceROIs'][slice]
    region_annotation.write_region_annotation_dict(dict2, str3)


def del_residual_subshape(dict_file, str_file, reg_num): #should be fixed
    for region in dict_file['Regions']:
        if region == reg_num:
            for slice in dict_file['Regions'][region]['ROI']['SliceROIs'].keys():
                for shape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])):
                    for subshape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                        shape_num = len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])
                        if int(subshape) > int(shape_num):
                            del dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape][subshape]
                        else:
                            print(f'region : {region} , slice : {slice} , shape : {shape} , subshape : {subshape}')
    region_annotation.write_region_annotation_dict(dict_file, str_file)


def sub_to_shape(dict1, str1):
    for region in dict1['Regions']:
        if dict1['Regions'][region]['ROI'] == None:
            continue
        for slice in dict1['Regions'][region]['ROI']['SliceROIs'].keys():
            for shape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice])):
                sub_len = len(dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape])
                for subshape in range(0, len(dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                    if subshape >= 1:
                        if sub_len == 2:
                            mid_idx = sub_len // 2
                            first_half = dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][:mid_idx]
                            second_half = dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape][mid_idx:]
                            dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape] = first_half
                            dict1['Regions'][region]['ROI']['SliceROIs'][slice] += []
                            dict1['Regions'][region]['ROI']['SliceROIs'][slice].append(second_half)
                        else:
                           len_to_split = [1]
                           sub_input = iter(dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape])
                           sub_output = [list(islice(sub_input, elem)) for elem in len_to_split]
                           dict1['Regions'][region]['ROI']['SliceROIs'][slice][shape] = sub_output[0]
    region_annotation.write_region_annotation_dict(dict1, str1)


def convert_spline_into_line(dict_file, str_file, region):
    for slice in dict_file['Regions'][region]['ROI']['SliceROIs'].keys():
        for x in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])):
            for i in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][x])):
                line = dict_file['Regions'][region]['ROI']['SliceROIs'][slice][x][i]['Type']
                if line == 'Spline':
                    dict_file['Regions'][region]['ROI']['SliceROIs'][slice][x][i]['Type'] = 'Line'
    region_annotation.write_region_annotation_dict(dict_file, str_file)


def add_region_to_parent_region(dict1, dict2, str2, region, slice):
    parent_id = int(dict2['Regions'][region]['ParentID'])
    dict2['Regions'][parent_id]['ROI']['SliceROIs'][slice] += dict1['Regions'][region]['ROI']['SliceROIs'][slice]
    region_annotation.write_region_annotation_dict(dict2, str2)


def del_same_lines(dict_file, str_file, region):
    for slice in dict_file['Regions'][region]['ROI']['SliceROIs'].keys():
        for shape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice])):
            for subshape in range(0, len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape])):
                if len(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape]) >= 2:
                    len_to_split = [1]
                    sub_input = iter(dict_file['Regions'][region]['ROI']['SliceROIs'][slice][shape])
                    sub_output = [list(islice(sub_input, elem)) for elem in len_to_split]
                    if not sub_output[0]:
                        del sub_output
    region_annotation.write_region_annotation_dict(dict_file, str_file)


def region_line_to_undefined_line(dict_file, str_file, region, slice_list):
    for slice in slice_list:
        dict_file['Regions'][-1]['ROI']['SliceROIs'][slice] += dict_file['Regions'][region]['ROI']['SliceROIs'][slice]
        del dict_file['Regions'][region]['ROI']['SliceROIs'][slice]
    region_annotation.write_region_annotation_dict(dict_file, str_file)


def merge_to_the_left(dict_left, res_dict, mid_dict, res_str, region, slice, x):
    # x depends on the number of annotated region (in most cases, 1 <= x)
    l_coords = dict_left['Regions'][region]['ROI']['SliceROIs'][slice][x][0]['Points'][0][0]
    print("dict_left : ", l_coords)
    mid_coords = mid_dict['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'][0][0]
    print("mid : ", mid_coords)

    if mid_coords > l_coords:
        res_dict['Regions'][region]['ROI']['SliceROIs'][slice][x] = dict_left['Regions'][region]['ROI']['SliceROIs'][slice][x]
        region_annotation.write_region_annotation_dict(res_dict, res_str)
        print("Attached to the left")
    else:
        print("Left X")


def merge_to_the_right(dict_right, res_dict, mid_dict, res_str, region, slice, x):
    r_coords = dict_right['Regions'][region]['ROI']['SliceROIs'][slice][x][0]['Points'][0][0] # in most cases, x = 0
    print("dict_right : ", r_coords)
    mid_coords = mid_dict['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'][0][0]
    print("mid : ", mid_coords)

    if mid_coords < r_coords:
        res_dict['Regions'][region]['ROI']['SliceROIs'][slice][x] = dict_right['Regions'][region]['ROI']['SliceROIs'][slice][x]
        region_annotation.write_region_annotation_dict(res_dict, res_str)
        print("Attached to the right")
    else:
        print("Right X")

def read_image():
    img = region_annotation.read_region_annotation('white_pot_roi.reganno')
    print(img)
    height, width = len(dict_file['Regions'][]['ROI']['SliceROIs'][slice][shape][subshape]['Points']), len(img[1])
    mask = np.zeros(shape=(height, width), dtype=np.bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    plt.imshow(mask)
    plt.show()

if __name__ == "__main__":
    # path = '/Users/fenglab/Downloads/subregion_annotation'
    # str1 = os.path.join(path, f'sh_subregion_interpolation_final_20201220.reganno')
    # dict1 = region_annotation.read_region_annotation(str1)
    #
    # str_copy = os.path.join(path, f'bigsub_merged_jiwon_20201204_cutted_for_tagging.reganno')
    # dict_copy = region_annotation.read_region_annotation(str_copy)
    #
    # str_to = os.path.join(path, f'sh_subregion_interpolated_fix_cutline_20201218.reganno')
    # dict_to = region_annotation.read_region_annotation(str_to)
    #
    # str_2 = os.path.join(path, f'bigsub_merged_jiwon_20201204_fix_midline.reganno')
    # dict2 = region_annotation.read_region_annotation(str_2)
    #
    # print_region(dict1)
    # map_regions(dict_to, dict_copy, str_copy)
    # region_info(dict2, 315)
    read_image()
