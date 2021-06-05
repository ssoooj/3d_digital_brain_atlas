import os
import sys
import json
import glob
import copy
import math
import pathlib
import traceback
import multiprocessing
import scipy.io
import scipy.ndimage
import natsort
import itertools
import lap
import cv2
import time

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from utils import shading_correction


logger = setup_logger()

start = time.time()

def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e

def do_lemur_subregion_detection(folder):
    img_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    ref_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                   f'sh_subregion_interpolation_final_20201220.reganno')
    ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                               f'sh_subregion_interpolation_final_det_v1_20210104.reganno')
    if os.path.exists(ra_filename):
        logger.info(f'roi to ra {ra_filename} done')
    else:
        read_ratio = 16
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        img = ZImg(img_filename, region=ZImgRegion(), scene=0, ratio=read_ratio)
        logger.info(f'finish reading image from {img_filename}: {img}')
        img_data, _ = img_util.normalize_img_data(img.data[0], min_max_percentile=(2, 98))
        nchs, depth, height, width = img_data.shape

        ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        img_annotation_mask = np.zeros(shape=(depth, 1, height, width), dtype=np.bool)

        ctx_layers = [3159, 3158, 3157, 3156, 3155]
        hpf_regions = [1929, 1939, 1949, 1959, 1969, 1979]
        bg_regions = [4779, 4778, 4777, 4776, 4775, 4774, 4773]

        for region_id, slice_rois in region_to_masks.items():
            if region_id in ctx_layers:
                for img_slice, maskps in slice_rois.items():
                    for compact_mask, x_start, y_start, _ in maskps:
                         if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(height, width), dtype=np.bool)
                        mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                        img_annotation_mask[img_slice, 0, :, :] |= mask
            # for region_id, region_props in ra_dict['Regions'].items():
            #     region_props['ROI'] = None

        from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector
        lbd = get_lemur_bigregion_detector()

        for slice in range(depth):
            logger.info(f'slice {slice}')
            slice_mask = img_annotation_mask[slice, 0, :, :]
            do_left = do_right = True
            if slice_mask.sum() > 0:
                if slice_mask[:, 0:width // 2].sum() > slice_mask[:, width // 2:width].sum():
                    do_left = False
                else:
                    do_right = False

            slice_img_data = np.moveaxis(img_data[:, slice, :, :], 0, -1)

            if do_right:
                slice_img_data_right = slice_img_data.copy()
                slice_img_data_right[:, 0:int(width * 0.48), :] = 0
                detections = lbd.run_on_opencv_image(slice_img_data_right, tile_size=20000)
                for region_id, label_image in detections['id_to_label_image'].items():
                    if region_id in ctx_layers:
                        shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                        if len(shapes) > 0 and region_id in ctx_layers:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

            if do_left:
                slice_img_data_left = slice_img_data.copy()
                slice_img_data_left[:, int(width * 0.52):width, :] = 0
                detections = lbd.run_on_opencv_image(slice_img_data_left, tile_size=20000)
                for region_id, label_image in detections['id_to_label_image'].items():
                    if region_id in ctx_layers:
                        shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                        if len(shapes) > 0 and region_id in ctx_layers:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

        del ra_dict['Regions'][-1]
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
        region_annotation.write_region_annotation_dict(ra_dict, ra_filename)

        logger.info(f'det subregion {ra_filename} done')

if __name__ == '__main__':
    folder = os.path.join(io.fs3017_dir(), 'eeum', 'training', '2021', 'CTX_layers')

    do_lemur_subregion_detection(folder)
    print(f'Time taken : {(int(time.time() - start)//60)} minutes')
