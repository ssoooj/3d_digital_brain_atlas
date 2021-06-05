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

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger
from utils import shading_correction

logger = setup_logger()


def _callback(result):
    logger.info(f'finished {result}')


def _error_callback(e: BaseException):
    traceback.print_exception(type(e), e, e.__traceback__)
    raise e


def align_with_hj_transform_one_image(img_idx: int):
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')

    res_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.nim')
    if os.path.exists(res_filename):
        logger.info(f'image {img_idx} done')
    else:
        hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
        hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

        logger.info(img_idx)
        czi_file_idx = img_idx // 4
        czi_scene_idx = img_idx % 4
        czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_{czi_file_idx + 1:02}.czi')
        img_data, img_info = shading_correction.correct_shading(czi_filename, scene=czi_scene_idx,
                                                                inverse_channels=(0,))

        tfm = hj_transforms['tforms'][img_idx, 0].copy().astype(np.float64)
        if tfm[0, 0] < 0:
            tfm[2, 0] -= 2  # no idea why
        # img = ZImg(czi_filename, scene=czi_scene_idx)
        logger.info(img_data.shape)
        des_height = hj_transforms['refSize'][0, 0]
        des_width = hj_transforms['refSize'][0, 1]
        pad_img_data = np.ascontiguousarray(img_util.pad_img(np.swapaxes(img_data, -1, -2),
                                                             des_height=des_height,
                                                             des_width=des_width))
        logger.info(pad_img_data.shape)
        for ch in range(pad_img_data.shape[0]):
            pad_img_data[ch, 0, :, :] = cv2.warpAffine(pad_img_data[ch, 0, :, :], tfm.T[0:2, :],
                                                       dsize=(des_width, des_height),
                                                       flags=cv2.INTER_CUBIC,
                                                       borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=0)
        img2 = ZImg(pad_img_data, img_info)
        img2.save(res_filename)
        logger.info(f'image {img_idx} done')


def align_with_hj_transform_all_images(folder: str):
    hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
    hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

    with multiprocessing.Pool(8) as pool:
        pool.map_async(align_with_hj_transform_one_image, list(range(len(hj_transforms['tforms']))),
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    combined_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    if os.path.exists(combined_filename):
        logger.info(f'image {combined_filename} done')
    else:
        aligned_filenames = [os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.nim') for
                             img_idx in range(len(hj_transforms['tforms']))]
        # logger.info(';'.join(aligned_filenames))
        imgMerge = ZImgMerge()
        imgSubBlocks = []
        for idx, fn in enumerate(aligned_filenames):
            imgSubBlocks.append(ZImgTileSubBlock(ZImgSource(fn)))
            imgMerge.addImg(imgSubBlocks[-1], (0, 0, idx, 0, 0), pathlib.PurePath(fn).name)
        imgMerge.resolveLocations()
        imgMerge.save(combined_filename)
        logger.info(f'image {combined_filename} done')


def align_rois_with_hj_transform_one_image(img_idx: int):
    folder = os.path.join(io.fs3017_data_dir(), 'eeum', 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')

    res_filename = os.path.join(folder, 'hj_aligned_annotation', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.nimroi')
    res_filename2 = os.path.join(folder, 'hj_aligned_annotation', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.reganno')
    if os.path.exists(res_filename):
        logger.info(f'roi {img_idx} done')
    else:
        logger.info(img_idx)
        czi_file_idx = img_idx // 4
        czi_scene_idx = img_idx % 4
        czi_annotation_filename = glob.glob(os.path.join(folder, 'czi_annotation',
                                                         f'Hotsauce_{czi_file_idx + 1:02}_{czi_scene_idx + 1}of4*.nimroi'))
        assert len(czi_annotation_filename) <= 1, czi_annotation_filename
        if len(czi_annotation_filename) == 0:
            logger.info(f'no annotation for file {czi_file_idx + 1} scene {czi_scene_idx + 1}')
        else:
            hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
            hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

            czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_{czi_file_idx + 1:02}.czi')

            tfm = hj_transforms['tforms'][img_idx, 0].copy().astype(np.float64)
            if tfm[0, 0] < 0:
                tfm[2, 0] -= 2  # no idea why
            czi_img_info = ZImg.readImgInfos(czi_filename)[czi_scene_idx]
            czi_img_height = czi_img_info.height
            czi_img_width = czi_img_info.width
            logger.info(czi_img_info)
            des_height = hj_transforms['refSize'][0, 0]
            des_width = hj_transforms['refSize'][0, 1]
            print(des_width, des_height, czi_img_width, czi_img_height)

            def transform_fun(input_coords: np.ndarray):
                assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
                res = np.concatenate((input_coords,
                                      np.ones(shape=(input_coords.shape[0], 1), dtype=np.float64)),
                                     axis=1)
                # rotation back
                if img_idx != 12:
                    M1 = np.array([[0., 1., 0.], [-1, 0, 0], [0, 0, 1]])
                    res = M1 @ res.T
                else:
                    res = res.T
                # swap xy
                res[[0, 1]] = res[[1, 0]]
                # pad
                res[0, :] += int((des_width - czi_img_height) / 2.0)
                res[1, :] += int((des_height - czi_img_width) / 2.0)
                # tfm
                res = tfm.T @ res
                res = res.T[:, 0:2]
                assert res.ndim == 2 and res.shape[1] == 2, res.shape
                return res

            czi_annotation = nim_roi.read_roi(czi_annotation_filename[0])
            aligned_roi_dict = nim_roi.transform_roi_dict(czi_annotation, transform_fun)
            nim_roi.write_roi_dict(aligned_roi_dict, res_filename)

            czi_ra_filename = glob.glob(os.path.join(folder, 'czi_annotation',
                                                     f'Hotsauce_{czi_file_idx + 1:02}_{czi_scene_idx + 1}of4*.reganno'))
            assert len(czi_ra_filename) <= 1, czi_ra_filename
            if len(czi_ra_filename) > 0:
                czi_ra = region_annotation.read_region_annotation(czi_ra_filename[0])
                aligned_ra_dict = region_annotation.transform_region_annotation_dict(czi_ra, transform_fun)
                region_annotation.write_region_annotation_dict(aligned_ra_dict, res_filename2)

            logger.info(f'roi {img_idx} done')


def align_rois_with_hj_transform_all_images(folder: str):
    hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
    hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

    with multiprocessing.Pool(8) as pool:
        pool.map_async(align_rois_with_hj_transform_one_image, list(range(len(hj_transforms['tforms']))),
                       chunksize=1, callback=None, error_callback=_error_callback).wait()

    combined_filename = os.path.join(folder, 'hj_aligned_annotation_merge', f'Lemur-H_SMI99_VGluT2_NeuN_all.nimroi')
    combined_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge', f'Lemur-H_SMI99_VGluT2_NeuN_all.reganno')
    if os.path.exists(combined_filename):
        logger.info(f'roi {combined_filename} done')
    else:
        start_slice_and_roi_dict = []
        for img_idx in range(len(hj_transforms['tforms'])):
            fn = os.path.join(folder, 'hj_aligned_annotation', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.nimroi')
            if os.path.exists(fn):
                start_slice_and_roi_dict.append((img_idx, nim_roi.read_roi(fn)))
        merged_roi = nim_roi.merge_roi_dicts(start_slice_and_roi_dict)
        nim_roi.write_roi_dict(merged_roi, combined_filename)

        start_slice_and_ra_dict = []
        for img_idx in range(len(hj_transforms['tforms'])):
            fn = os.path.join(folder, 'hj_aligned_annotation', f'Lemur-H_SMI99_VGluT2_NeuN_{img_idx + 1}.reganno')
            if os.path.exists(fn):
                start_slice_and_ra_dict.append((img_idx, region_annotation.read_region_annotation(fn)))
        merged_ra = region_annotation.merge_region_annotation_dicts(start_slice_and_ra_dict)
        region_annotation.write_region_annotation_dict(merged_ra, combined_ra_filename)

        logger.info(f'roi {combined_filename} done')


def flip_rois_for_manual_tagging(folder: str):
    hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
    hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

    combined_filename = os.path.join(folder, 'hj_aligned_annotation_merge', f'Lemur-H_SMI99_VGluT2_NeuN_all.nimroi')
    combined_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge', f'Lemur-H_SMI99_VGluT2_NeuN_all.reganno')

    combined_flipped_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                             f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging.nimroi')
    combined_ra_flipped_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                                f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging.reganno')
    if os.path.exists(combined_flipped_filename):
        logger.info(f'flip roi {combined_flipped_filename} done')
    else:
        des_width = hj_transforms['refSize'][0, 1]

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res[:, 0] = des_width - res[:, 0]
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        slices_to_be_flipped = []
        for img_idx in range(len(hj_transforms['tforms'])):
            tfm = hj_transforms['tforms'][img_idx, 0]
            if tfm[0, 0] > 0:
                slices_to_be_flipped.append(img_idx)

        roi_dict = nim_roi.read_roi(combined_filename)
        roi_dict = nim_roi.transform_roi_dict(roi_dict, transform_fun, slices_to_be_flipped)
        nim_roi.write_roi_dict(roi_dict, combined_flipped_filename)

        ra_dict = region_annotation.read_region_annotation(combined_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun, slices_to_be_flipped)
        region_annotation.write_region_annotation_dict(ra_dict, combined_ra_flipped_filename)

        logger.info(f'flip roi {combined_flipped_filename} done')


def convert_rois_to_region_annotation_for_tagging(folder):
    hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
    hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

    combined_filename = os.path.join(folder, 'hj_aligned_annotation_merge', f'Lemur-H_SMI99_VGluT2_NeuN_all.nimroi')

    combined_ra_flipped_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                                f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging.reganno')
    if os.path.exists(combined_ra_flipped_filename):
        logger.info(f'roi to ra {combined_ra_flipped_filename} done')
    else:
        des_width = hj_transforms['refSize'][0, 1]

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res[:, 0] = des_width - res[:, 0]
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        slices_to_be_flipped = []
        for img_idx in range(len(hj_transforms['tforms'])):
            tfm = hj_transforms['tforms'][img_idx, 0]
            if tfm[0, 0] > 0:
                slices_to_be_flipped.append(img_idx)

        roi_dict = nim_roi.read_roi(combined_filename)
        ra_dict = region_annotation.convert_roi_dict_to_region_annotation_dict(roi_dict)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun, slices_to_be_flipped)

        region_annotation.write_region_annotation_dict(ra_dict, combined_ra_flipped_filename)

        logger.info(f'roi to ra {combined_ra_flipped_filename} done')


def do_tag_interpolation(folder: str):
    input_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                     f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging_sh.reganno')

    interpolated_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                            f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging_sh_interpolated.reganno')

    input_ra_filename = os.path.join(folder, 'blockface',
                                     f'shifted_Hotsauce_blockface-outline_grouped_fix.reganno')

    interpolated_ra_filename = os.path.join(folder, 'blockface',
                                            f'shifted_Hotsauce_blockface-outline_grouped_fix_interpolated.reganno')

    input_ra_filename = os.path.join(folder, 'subregion',
                                     f'sh_subregion_interpolation_final_20210216_cortical_half_cut.reganno')

    interpolated_ra_filename = os.path.join(folder, 'subregion',
                                            f'sh_subregion_interpolation_final_20210216_cortical_half_cut_interpolated.reganno')

    input_ra_filename = os.path.join(folder, 'subregion',
                                     f'04_scaled_deformed_annotation_bf_refine_cut_merge.reganno')

    interpolated_ra_filename = os.path.join(folder, 'subregion',
                                            f'04_scaled_deformed_annotation_bf_refine_cut_merge_interpolated.reganno')
    if os.path.exists(interpolated_ra_filename):
        logger.info(f'tag interpolation {interpolated_ra_filename} done')
    else:
        scale_down = 1.0 / 1  # otherwise the mask will be too big
        ra_dict = region_annotation.read_region_annotation(input_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {input_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {input_ra_filename}')
        region_to_annotated_slices = {}
        region_to_start_slice = {}
        region_to_end_slice = {}
        for region, slice_to_masks in region_to_masks.items():
            if region == -1:
                continue
            region_to_annotated_slices[region] = []
            for slice, masks in slice_to_masks.items():
                region_to_annotated_slices[region].append(slice)
            region_to_start_slice[region] = min(region_to_annotated_slices[region])
            region_to_end_slice[region] = max(region_to_annotated_slices[region])

        def get_overlap_between_masks(maskp, ref_maskp):
            mask, x, y, _ = maskp
            ref_mask, ref_x, ref_y, _ = ref_maskp
            des_width = max(mask.shape[1] + x, ref_mask.shape[1] + ref_x)
            des_height = max(mask.shape[0] + y, ref_mask.shape[0] + ref_y)
            # print(x, y, ref_x, ref_y, des_width, des_height, mask.shape, ref_mask.shape)
            # print(((y, des_height - mask.shape[0] - y),
            #        (x, des_width - mask.shape[1] - x)))
            pad_mask = np.pad(mask, ((y, des_height - mask.shape[0] - y),
                                     (x, des_width - mask.shape[1] - x)))
            # print(((ref_y, des_height - ref_mask.shape[0] - ref_y),
            #        (ref_x, des_width - ref_mask.shape[1] - ref_x)))
            pad_ref_mask = np.pad(ref_mask, ((ref_y, des_height - ref_mask.shape[0] - ref_y),
                                             (ref_x, des_width - ref_mask.shape[1] - ref_x)))
            mask_overlap = (pad_mask & pad_ref_mask).sum() * 1.0
            # print(mask.sum(), ref_mask.sum(), mask_overlap)
            return (mask_overlap / (pad_mask | pad_ref_mask).sum())
            # return max(mask_overlap / mask.sum(), mask_overlap / ref_mask.sum())

        def get_distance_between_mask_centroids(maskp, ref_maskp):
            mask, x, y, _ = maskp
            ref_mask, ref_x, ref_y, _ = ref_maskp
            mask_centroid = scipy.ndimage.measurements.center_of_mass(mask)
            mask_centroid = np.array([mask_centroid[1] + x, mask_centroid[0] + y])
            ref_mask_centroid = scipy.ndimage.measurements.center_of_mass(ref_mask)
            ref_mask_centroid = np.array([ref_mask_centroid[1] + ref_x, ref_mask_centroid[0] + ref_y])
            return np.linalg.norm(mask_centroid - ref_mask_centroid, ord=2)

        def caculate_pairwise_match_score(masks, all_ref_masks):
            # n1 is undefined regions in current slice
            # n2 is reference regions from last slice
            n1_n2_match_IOU_metric = np.full(shape=(len(masks), len(all_ref_masks)), fill_value=np.inf)
            n1_n2_match_dist_metric = np.full(shape=(len(masks), len(all_ref_masks)), fill_value=np.inf)
            IOU_metric_weight = 0.5
            dist_metric_weight = 0.5
            ref_region_list = []
            for n1_idx, maskp in enumerate(masks):
                assert maskp[0].sum() > 0
                for n2_idx, (ref_maskp, region) in enumerate(all_ref_masks):
                    if ref_maskp[0].sum() <= 0:
                        continue
                    ref_region_list.append(region)
                    ref_mask, ref_x, ref_y, _ = ref_maskp
                    n1_n2_match_IOU_metric[n1_idx, n2_idx] = -get_overlap_between_masks(maskp, ref_maskp)
                    n1_n2_match_dist_metric[n1_idx, n2_idx] = get_distance_between_mask_centroids(maskp, ref_maskp)
            n1_n2_match_IOU_metric = (n1_n2_match_IOU_metric - np.min(n1_n2_match_IOU_metric)) / np.ptp(
                n1_n2_match_IOU_metric)
            n1_n2_match_dist_metric = (n1_n2_match_dist_metric - np.min(n1_n2_match_dist_metric)) / np.ptp(
                n1_n2_match_dist_metric)
            n1_n2_match = n1_n2_match_IOU_metric * IOU_metric_weight + n1_n2_match_dist_metric * dist_metric_weight
            return n1_n2_match, ref_region_list

        def find_best_matched_region(maskp, region_to_ref_masks, annotated_regions_in_current_slice):
            assert len(region_to_ref_masks) > 0
            res = -1
            max_overlap = -1
            for region, ref_masks in region_to_ref_masks.items():
                if region in annotated_regions_in_current_slice:
                    continue
                else:
                    for ref_maskp in ref_masks:
                        overlap = get_overlap_between_masks(maskp, ref_maskp)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            res = region

            if res >= 0:
                return res

            # use centroid
            min_dist = 1e20
            for region, ref_masks in region_to_ref_masks.items():
                if region in annotated_regions_in_current_slice:
                    continue
                else:
                    for ref_maskp in ref_masks:
                        dist = get_distance_between_mask_centroids(maskp, ref_maskp)
                        if dist < min_dist:
                            min_dist = dist
                            res = region

            assert res >= 0, res
            return res

        region_to_ref_masks = {}
        for slice, masks in region_to_masks[-1].items():
            logger.info(slice)
            for region, annotated_slices in region_to_annotated_slices.items():
                for annotated_slice in annotated_slices:
                    if annotated_slice < slice:
                        region_to_ref_masks[region] = region_to_masks[region][annotated_slice]
                    else:
                        break
            annotated_regions_in_current_slice = [
                region for region, annotated_slices in region_to_annotated_slices.items() if slice in annotated_slices
            ]
            matched_region_to_masks = {}

            # 1: for each undefined region, find the best matching region with highest IOU, if not found, then find the
            #    best matching region with shortest centroid distance
            # 2: do a one-to-one match first, then for each undefined region that are not matched to any region (because
            #    this region appears more times than previous slice), use strategy 1
            matching_strategy = 2
            if matching_strategy == 1:
                for mask in masks:
                    assert mask[0].sum() > 0, slice
                    best_matched_region = find_best_matched_region(mask, region_to_ref_masks,
                                                                   annotated_regions_in_current_slice)
                    if slice not in ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs']:
                        ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs'][slice] = []
                    ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs'][slice].append(mask[-1])
                    if best_matched_region in matched_region_to_masks:
                        matched_region_to_masks[best_matched_region].append(mask)
                    else:
                        matched_region_to_masks[best_matched_region] = [mask]
            elif matching_strategy == 2:
                all_ref_masks = []
                for region, ref_masks in region_to_ref_masks.items():
                    if region in annotated_regions_in_current_slice:
                        continue
                    else:
                        for ref_mask in ref_masks:
                            all_ref_masks.append((ref_mask, region))
                n1_n2_match, ref_region_list = caculate_pairwise_match_score(masks, all_ref_masks)
                # Assignment. `x[i]` specifies the column to which row `i` is assigned.
                # Assignment. `y[j]` specifies the row to which column `j` is assigned.
                # print(n1_n2_match)
                cost, x, y = lap.lapjv(n1_n2_match, extend_cost=True, cost_limit=2.0, return_cost=True)
                assert len(x) == len(masks)
                for mask_idx, mask in enumerate(masks):
                    if x[mask_idx] >= 0:
                        best_matched_region = ref_region_list[x[mask_idx]]
                    else:
                        # use strategy 1
                        best_matched_region = find_best_matched_region(mask, region_to_ref_masks,
                                                                       annotated_regions_in_current_slice)
                    if slice not in ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs']:
                        ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs'][slice] = []
                    ra_dict['Regions'][best_matched_region]['ROI']['SliceROIs'][slice].append(mask[-1])
                    if best_matched_region in matched_region_to_masks:
                        matched_region_to_masks[best_matched_region].append(mask)
                    else:
                        matched_region_to_masks[best_matched_region] = [mask]

            # update reference
            for matched_region, masks in matched_region_to_masks.items():
                region_to_ref_masks[matched_region] = masks
            for region in annotated_regions_in_current_slice:
                if slice != region_to_end_slice[region]:
                    region_to_ref_masks[region] = region_to_masks[region][slice]
                else:
                    logger.info(f'region {region} end')
                    if region in region_to_ref_masks:
                        del region_to_ref_masks[region]

        del ra_dict['Regions'][-1]
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords / scale_down)
        region_annotation.write_region_annotation_dict(ra_dict, interpolated_ra_filename)

        logger.info(f'tag interpolation {interpolated_ra_filename} done')


def flip_region_annotation_back_after_finishing_tagging(folder):
    hj_transform_mat_file = os.path.join(folder, 'Lemur-H_NeuN_VGluT2_SMI99_tforms_1.mat')
    hj_transforms = scipy.io.loadmat(hj_transform_mat_file)

    # input_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
    #                                  f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging_sh_interpolated_sh.reganno')
    input_ra_filename = os.path.join(io.fs3017_dir(), 'eeum', 'sohyeon',
                                     'Lemur-H-SMI99_VGluT2_NeuN_final.reganno')
    interpolated_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                            f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final.reganno')
    if os.path.exists(interpolated_ra_filename):
        logger.info(f'roi to ra {interpolated_ra_filename} done')
    else:
        des_width = hj_transforms['refSize'][0, 1]

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res[:, 0] = des_width - res[:, 0]
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        slices_to_be_flipped = []
        for img_idx in range(len(hj_transforms['tforms'])):
            tfm = hj_transforms['tforms'][img_idx, 0]
            if tfm[0, 0] > 0:
                slices_to_be_flipped.append(img_idx)

        ra_dict = region_annotation.read_region_annotation(input_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun, slices_to_be_flipped)
        region_annotation.write_region_annotation_dict(ra_dict, interpolated_ra_filename)

        logger.info(f'flip back tagged {interpolated_ra_filename} done')


def do_lemur_bigregion_detection(folder):
    img_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    ref_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                   f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final.reganno')
    ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                               f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_det.reganno')
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
        for region_id, slice_rois in region_to_masks.items():
            for img_slice, maskps in slice_rois.items():
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
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
                    shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                    if len(shapes) > 0:
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
                    shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                    if len(shapes) > 0:
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

        logger.info(f'det big region {ra_filename} done')


def do_lemur_bigregion_detection_v2(folder):
    img_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    ref_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                   f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final.reganno')
    ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                               f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v3_det_1.reganno')
    ra_filename2 = os.path.join(folder, 'hj_aligned_annotation_merge',
                                f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v3_det_2.reganno')
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
        for region_id, region_props in ra_dict['Regions'].items():
            region_props['ROI'] = None

        from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector_v3
        lbd = get_lemur_bigregion_detector_v3()

        for slice in range(depth):
            logger.info(f'slice {slice}')

            slice_img_data = np.moveaxis(img_data[:, slice, :, :], 0, -1)

            detections = lbd.run_on_opencv_image(slice_img_data, tile_size=20000)
            for region_id, label_image in detections['id_to_label_image'].items():
                shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                if len(shapes) > 0:
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
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s * 2)
        region_annotation.write_region_annotation_dict(ra_dict2, ra_filename2)

        logger.info(f'det big region v3 {ra_filename} done')


def do_lemur_bigregion_detection_v2_czi(folder, prefix):
    from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector_2ch
    lbd = get_lemur_bigregion_detector_2ch()
    for img_idx in range(46):
        img_filename = os.path.join(folder, f'{prefix}_{img_idx:02}.czi')
        if not os.path.exists(img_filename):
            logger.info(f'{img_filename} does not exist')
            continue
        num_scenes = len(ZImg.readImgInfos(img_filename))
        ref_ra_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final.reganno')
        for scene in range(num_scenes):
            ra_filename = os.path.join(folder, f'{prefix}_{img_idx:02}_scene{scene}.reganno')
            if os.path.exists(ra_filename):
                logger.info(f'roi to ra {ra_filename} done')
            else:
                read_ratio = 16
                scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
                img = ZImg(img_filename, region=ZImgRegion(), scene=scene, ratio=read_ratio)
                logger.info(f'finish reading image from {img_filename}: {img}')
                img_data, _ = img_util.normalize_img_data(img.data[0], min_max_percentile=(2, 98))
                nchs, depth, height, width = img_data.shape

                ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
                for region_id, region_props in ra_dict['Regions'].items():
                    region_props['ROI'] = None

                for slice in range(depth):
                    logger.info(f'slice {slice}')

                    slice_img_data = np.moveaxis(img_data[0:2, slice, :, :], 0, -1)

                    detections = lbd.run_on_opencv_image(slice_img_data, tile_size=20000)
                    for region_id, label_image in detections['id_to_label_image'].items():
                        shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                        if len(shapes) > 0:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

                del ra_dict['Regions'][-1]
                ra_dict = region_annotation.transform_region_annotation_dict(ra_dict,
                                                                             lambda coords: coords * read_ratio)
                region_annotation.write_region_annotation_dict(ra_dict, ra_filename)

                logger.info(f'det big czi v3 {ra_filename} done')


def do_lemur_bigregion_detection_v4(folder):
    img_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    ref_ra_filename = os.path.join(io.fs3017_data_dir(), 'lemur', 'sh-edit', 'sh_edit-jiwon_v20200826.reganno')
    midline_filename = os.path.join(io.fs3017_data_dir(), 'lemur', 'sh-edit', 'sh_cut_in_half.reganno')
    ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                               f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v4_det_v4_1.reganno')
    ra_filename2 = os.path.join(folder, 'hj_aligned_annotation_merge',
                                f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v4_det_v4_2.reganno')
    if os.path.exists(ra_filename):
        logger.info(f'roi to ra {ra_filename} done')
    else:
        read_ratio = 16
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        img = ZImg(img_filename, region=ZImgRegion(), scene=0, xRatio=read_ratio, yRatio=read_ratio)
        logger.info(f'finish reading image from {img_filename}: {img}')
        img_data, _ = img_util.normalize_img_data(img.data[0], min_max_percentile=(2, 98))
        nchs, depth, height, width = img_data.shape

        valid_slices = set(
            [i for i in itertools.chain((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19,
                                         20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 38, 40, 42, 44, 46, 50, 52,
                                         54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 81, 82, 83,
                                         84, 88, 89, 91, 93, 94, 95, 97, 99, 103, 105, 107, 111, 112, 113,
                                         115, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141,
                                         143, 146, 147, 148, 151, 152, 156, 157, 158, 160, 163, 164, 174,
                                         175),
                                        )])
        valid_slices = set(
            [i for i in itertools.chain((0, 2, 4, 6, 8, 11, 12, 14, 18, 20, 22, 23, 24, 26, 30, 34, 38, 40,
                                         42, 44, 46, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
                                         78, 80, 81, 82, 83, 84, 88, 89, 91, 93, 95, 97, 99, 103, 105, 107,
                                         111, 112, 113, 115, 119, 121, 123, 125, 127, 129, 131, 133, 135,
                                         137, 139, 141, 143, 146, 147, 148, 151, 152, 156, 157, 158, 160,
                                         163, 164, 174, 175),
                                        )])

        ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {ref_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ref_ra_filename}')

        ra_dict2 = region_annotation.read_region_annotation(midline_filename)
        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict2, lambda coords: coords * scale_down)
        logger.info(f'finish reading {midline_filename}')
        midline_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict2)
        logger.info(f'finish reading masks from {midline_filename}')

        for region_id, region_props in ra_dict['Regions'].items():
            region_props['ROI'] = None

        from models.nuclei.nuclei.predictor import get_lemur_bigregion_detector_v4
        lbd = get_lemur_bigregion_detector_v4()

        subregion_list = [1939, 1949, 1929, 1979, 1969, 1959,
                          3159, 3158, 3157, 3156, 3155, 3154, 3153,
                          4779, 4778,
                          4777, 4776,
                          4773, 4774,
                          4775,
                          ]
        for slice in range(depth):
            logger.info(f'slice {slice}')
            if slice in valid_slices:
                annotation_mask = np.zeros(shape=(height, width), dtype=np.uint32)
                # return a map from region_id to
                # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
                for region_id, slice_rois in region_to_masks.items():
                    if region_id <= 0 or region_id == 73:
                        continue
                    if slice in slice_rois:
                        maskps = slice_rois[slice]
                        for compact_mask, x_start, y_start, _ in maskps:
                            if compact_mask.sum() == 0:
                                continue
                            assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                            mask = np.zeros(shape=(height, width), dtype=np.bool)
                            mask[y_start:y_start + compact_mask.shape[0],
                            x_start:x_start + compact_mask.shape[1]] = compact_mask
                            if region_id in subregion_list:
                                target_id = ra_dict['Regions'][region_id]['ParentID']
                            else:
                                target_id = region_id

                            mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                            annotation_mask[mask] = target_id
                slice_rois = midline_to_masks[-1]
                assert slice in slice_rois, slice
                maskps = slice_rois[slice]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = 0  # cut

                for region_id in np.unique(annotation_mask):
                    if region_id <= 0:
                        continue
                    if region_id == 315:
                        labeled_array, num_features = scipy.ndimage.label(
                            scipy.ndimage.binary_opening(annotation_mask == region_id, structure=np.ones((7, 7))))
                    elif region_id == 1009 and slice == 115:
                        labeled_array, num_features = scipy.ndimage.label(
                            scipy.ndimage.binary_opening(annotation_mask == region_id, structure=np.ones((7, 7))))
                    else:
                        labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                    for label in range(1, num_features + 1):
                        mask = labeled_array == label
                        shapes = nim_roi.label_image_2d_to_spline_shapes(mask)
                        if len(shapes) > 0:
                            if ra_dict['Regions'][region_id]['ROI'] is None:
                                ra_dict['Regions'][region_id]['ROI'] = {}
                            if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                            if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                            else:
                                ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)
            else:
                slice_img_data = np.moveaxis(img_data[:, slice, :, :], 0, -1)

                detections = lbd.run_on_opencv_image(slice_img_data, tile_size=20000)
                for region_id, label_image in detections['id_to_label_image'].items():
                    if region_id <= 0:
                        continue
                    shapes = nim_roi.label_image_2d_to_spline_shapes(label_image)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][region_id]['ROI'] is None:
                            ra_dict['Regions'][region_id]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                        if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                        else:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
        region_annotation.write_region_annotation_dict(ra_dict, ra_filename)
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s * 2)
        region_annotation.write_region_annotation_dict(ra_dict2, ra_filename2)

        logger.info(f'det big region v4 {ra_filename} done')


def merge_edited_annotations(folder: str):
    img_filename = os.path.join(folder, 'hj_aligned', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    ref_ra_filename = os.path.join(io.fs3017_data_dir(), 'lemur', 'sh-edit', 'sh_edit-jiwon_v20200826.reganno')
    midline_filename = os.path.join(io.fs3017_data_dir(), 'lemur', 'sh-edit', 'sh_cut_in_half.reganno')
    jy_filename = os.path.join(folder,
                               'interns_edited_results/jayoung-20200925_Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v4_det_v3_1.reganno')
    yy_filename = os.path.join(folder,
                               'interns_edited_results/youyoung-20200924_Lemur-H_SMI99_VGluT2_NeuN_all_tagged_final_v4_det_v3_1.reganno')
    out_filename = os.path.join(folder, 'interns_edited_results/edited_merge_20201001_1.reganno')
    out_filename2 = os.path.join(folder, 'interns_edited_results/edited_merge_20201001_2.reganno')
    if os.path.exists(out_filename):
        logger.info(f'roi to ra {out_filename} done')
    else:
        read_ratio = 16
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        img = ZImg(img_filename, region=ZImgRegion(), scene=0, xRatio=read_ratio, yRatio=read_ratio)
        logger.info(f'finish reading image from {img_filename}: {img}')
        nchs, depth, height, width = img.data[0].shape

        ra_dict = region_annotation.read_region_annotation(ref_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {ref_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ref_ra_filename}')

        ra_dict2 = region_annotation.read_region_annotation(midline_filename)
        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict2, lambda coords: coords * scale_down)
        logger.info(f'finish reading {midline_filename}')
        midline_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict2)
        logger.info(f'finish reading masks from {midline_filename}')

        for region_id, region_props in ra_dict['Regions'].items():
            region_props['ROI'] = None

        subregion_list = [1939, 1949, 1929, 1979, 1969, 1959,
                          3159, 3158, 3157, 3156, 3155, 3154, 3153,
                          4779, 4778,
                          4777, 4776,
                          4773, 4774,
                          4775,
                          ]
        valid_slices = set(
            [i for i in itertools.chain((0, 2, 4, 6, 8, 11, 12, 14, 18, 20, 22, 23, 24, 26, 30, 34, 38, 40,
                                         42, 44, 46, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
                                         78, 80, 81, 82, 83, 84, 88, 89, 91, 93, 95, 97, 99, 103, 105, 107,
                                         111, 112, 113, 115, 119, 121, 123, 125, 127, 129, 131, 133, 135,
                                         137, 139, 141, 143, 146, 147, 148, 151, 152, 156, 157, 158, 160,
                                         163, 164, 174, 175),
                                        )])
        for slice in range(depth):
            logger.info(f'slice {slice}')
            if slice not in valid_slices:
                continue
            annotation_mask = np.zeros(shape=(height, width), dtype=np.uint32)
            # return a map from region_id to
            # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
            for region_id, slice_rois in region_to_masks.items():
                if region_id <= 0 or region_id == 73:
                    continue
                if slice in slice_rois:
                    maskps = slice_rois[slice]
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(height, width), dtype=np.bool)
                        mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        if region_id in subregion_list:
                            target_id = ra_dict['Regions'][region_id]['ParentID']
                        else:
                            target_id = region_id

                        mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                        annotation_mask[mask] = target_id
            slice_rois = midline_to_masks[-1]
            assert slice in slice_rois, slice
            maskps = slice_rois[slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(height, width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                annotation_mask[mask] = 0  # cut

            for region_id in np.unique(annotation_mask):
                if region_id <= 0:
                    continue
                if region_id == 315:
                    labeled_array, num_features = scipy.ndimage.label(
                        scipy.ndimage.binary_opening(annotation_mask == region_id, structure=np.ones((7, 7))))
                elif region_id == 1009 and slice == 115:
                    labeled_array, num_features = scipy.ndimage.label(
                        scipy.ndimage.binary_opening(annotation_mask == region_id, structure=np.ones((7, 7))))
                else:
                    labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                for label in range(1, num_features + 1):
                    mask = labeled_array == label
                    shapes = nim_roi.label_image_2d_to_spline_shapes(mask)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][region_id]['ROI'] is None:
                            ra_dict['Regions'][region_id]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                        if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                        else:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)

        jy_edited = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                              1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                              0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                             dtype=np.bool)
        yy_edited = np.array((0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                              0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
                              0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                              1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
                              1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                              0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                              1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1),
                             dtype=np.bool)
        al_edited = np.array((0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1,
                              1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                              0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                              1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
                              1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                              1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1),
                             dtype=np.bool)
        ref_valid = np.invert(al_edited)
        assert np.all((jy_edited | yy_edited) == al_edited)
        logger.info(f'{jy_edited.shape[0]} slices')

        ra_dict_yy = region_annotation.read_region_annotation(yy_filename)
        ra_dict_jy = region_annotation.read_region_annotation(jy_filename)

        valid_regions = (315, 3111, 1089, 477, 803, 549, 1097, 313, 771, 354, 512, 1009, 9101, 9102, 9103)
        for region_id, region_props in ra_dict['Regions'].items():
            if region_id not in valid_regions:
                continue
            region_props['ROI']['SliceROIs'] = {k: v for k, v in region_props['ROI']['SliceROIs'].items() if
                                                ref_valid[k]}
            region_props['ROI']['SliceROIs'].update(
                {k: v for k, v in ra_dict_jy['Regions'][region_id]['ROI']['SliceROIs'].items() if jy_edited[k]})
            region_props['ROI']['SliceROIs'].update(
                {k: v for k, v in ra_dict_yy['Regions'][region_id]['ROI']['SliceROIs'].items() if yy_edited[k]})

        region_annotation.write_region_annotation_dict(ra_dict, out_filename)
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s * 2)
        region_annotation.write_region_annotation_dict(ra_dict2, out_filename2)

        logger.info(f'merge {out_filename} done')


def reduce_annotation_slices(folder: str):
    ra_filename = os.path.join(folder, 'fit/fit4/edited_merge_20201001_2_merge_align_fixed.reganno')
    reduced_ra_filename = os.path.join(folder, 'fit/fit4/reduced_edited_merge_20201001_2_merge_align_fixed.reganno')
    if os.path.exists(reduced_ra_filename):
        return

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s if s % 8 == 0 else -1)
    region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)

    logger.info(f'reduce {reduced_ra_filename} done')


def shift_jiwon_blockface_annotation(folder: str):
    ra_filename = os.path.join(folder, 'blockface/Hotsauce_blockface-outline.reganno')
    reduced_ra_filename = os.path.join(folder, 'blockface/shifted_Hotsauce_blockface-outline.reganno')
    if os.path.exists(reduced_ra_filename):
        return

    def transform_fun(input_coords: np.ndarray):
        assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
        res = input_coords.copy()
        res *= 15.32
        res[:, 0] += 512
        assert res.ndim == 2 and res.shape[1] == 2, res.shape
        return res

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    for region_id, region in ra_dict['Regions'].items():
        if region_id != -1:
            region['ROI'] = None
    ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: s + 14)
    ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict2, transform_fun=transform_fun)
    region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)

    logger.info(f'reduce {reduced_ra_filename} done')


def fix_jiwon_blockface_annotation(folder: str):
    ra_filename = os.path.join(folder, 'blockface/shifted_Hotsauce_blockface-outline_grouped.reganno')
    reduced_ra_filename = os.path.join(folder, 'blockface/shifted_Hotsauce_blockface-outline_grouped_fix.reganno')
    if os.path.exists(reduced_ra_filename):
        return

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    for region_id, region in ra_dict['Regions'].items():
        print(region_id)
        if region_id != -1 and region_id > 20:
            region['ROI'] = None
    region_annotation.write_region_annotation_dict(ra_dict, reduced_ra_filename)

    logger.info(f'reduce {reduced_ra_filename} done')


def build_sagittal_blockface_and_jiwon_annotation(folder: str):
    bf_filename = os.path.join(folder, 'blockface/Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    # make an 10um isotropic coronal blockface image
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = np.roll(img_data, 20, axis=-1)
        img_data[:, :, :, 0:20] = 0
        img_data = img_util.imresize(img_data, des_depth=depth * 5)
        img_data = img_data[:, 0:(depth * 5 - 4), :, :].copy()
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=10.)

    ra_filename = os.path.join(folder, 'blockface/Hotsauce_blockface-outline.reganno')
    reduced_ra_filename = os.path.join(folder,
                                       'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline-outline.reganno')
    if not os.path.exists(reduced_ra_filename):
        ra_dict = region_annotation.read_region_annotation(ra_filename)
        for region_id, region in ra_dict['Regions'].items():
            print(region_id)
            if region_id != -1:
                region['ROI'] = None

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res[:, 0] += 20
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun)
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict2, lambda s: (s + 14) * 5)
        ra_dict2['VoxelSizeXInUM'] = 10.
        ra_dict2['VoxelSizeYInUM'] = 10.
        ra_dict2['VoxelSizeZInUM'] = 10.
        region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)

    sagittal_bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    if not os.path.exists(sagittal_bf_filename):
        img = ZImg(coronal_bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        # to sagittal
        img_data = np.flip(np.swapaxes(img_data, 1, -1), axis=1).copy()
        img_util.write_img(sagittal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=10.)

    # use Atlas to interpolate coronal region annotation

    interpolated_bf_ra_filename = os.path.join(folder,
                                               'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline-outline_interpolated.reganno')
    coronal_bf_outline_label_filename = os.path.join(folder,
                                                     'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(coronal_bf_outline_label_filename):
        img_info = ZImg.readImgInfos(coronal_bf_filename)[0]
        ra_dict = region_annotation.read_region_annotation(interpolated_bf_ra_filename)
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        annotation_mask = np.zeros(shape=(img_info.depth,
                                          img_info.height,
                                          img_info.width),
                                   dtype=np.int8)
        annotation_mask.fill(-128)
        for region_id, slice_rois in region_to_masks.items():
            if region_id != -1:
                continue
            for img_slice, maskps in slice_rois.items():
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(annotation_mask.shape[-2], annotation_mask.shape[-1]), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[img_slice][mask] = region_id

        img_util.write_img(coronal_bf_outline_label_filename, annotation_mask, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=10.)

    coronal_bf_masked_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline_masked.nim')
    if not os.path.exists(coronal_bf_masked_filename):
        img = ZImg(coronal_bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        mask = ZImg(coronal_bf_outline_label_filename)
        mask_data = mask.data[0][0]
        for ch in range(nchs):
            img_data[ch][mask_data == -128] = 0
        img_util.write_img(coronal_bf_masked_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=10.)

    sagittal_bf_outline_label_filename = os.path.join(folder,
                                                      'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(sagittal_bf_outline_label_filename):
        img = ZImg(coronal_bf_outline_label_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        # to sagittal
        img_data = np.flip(np.swapaxes(img_data, 1, -1), axis=1).copy()
        img_util.write_img(sagittal_bf_outline_label_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=10.)

    bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_25_fixed_with_ice_half_midline.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 2.5)),
                                     des_height=int(math.ceil(height / 2.5)),
                                     des_width=int(math.ceil(width / 2.5)))
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=25., voxel_size_y=25., voxel_size_z=25.)

    bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_25_fixed_with_ice_half_midline.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 2.5)),
                                     des_height=int(math.ceil(height / 2.5)),
                                     des_width=int(math.ceil(width / 2.5)))
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=25., voxel_size_y=25., voxel_size_z=25.)

    bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    coronal_bf_filename = os.path.join(folder,
                                       'blockface/coronal_Hotsauce_25_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 2.5)),
                                     des_height=int(math.ceil(height / 2.5)),
                                     des_width=int(math.ceil(width / 2.5)),
                                     interpolant=Interpolant.Nearest)
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=25., voxel_size_y=25., voxel_size_z=25.)

    bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    coronal_bf_filename = os.path.join(folder,
                                       'blockface/sagittal_Hotsauce_25_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 2.5)),
                                     des_height=int(math.ceil(height / 2.5)),
                                     des_width=int(math.ceil(width / 2.5)),
                                     interpolant=Interpolant.Nearest)
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=25., voxel_size_y=25., voxel_size_z=25.)

    bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_100_fixed_with_ice_half_midline.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 10.)),
                                     des_height=int(math.ceil(height / 10.)),
                                     des_width=int(math.ceil(width / 10.)))
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=100., voxel_size_y=100., voxel_size_z=100.)

    bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_100_fixed_with_ice_half_midline.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 10.)),
                                     des_height=int(math.ceil(height / 10.)),
                                     des_width=int(math.ceil(width / 10.)))
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=100., voxel_size_y=100., voxel_size_z=100.)

    bf_filename = os.path.join(folder, 'blockface/coronal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    coronal_bf_filename = os.path.join(folder,
                                       'blockface/coronal_Hotsauce_100_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 10.)),
                                     des_height=int(math.ceil(height / 10.)),
                                     des_width=int(math.ceil(width / 10.)),
                                     interpolant=Interpolant.Nearest)
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=100., voxel_size_y=100., voxel_size_z=100.)

    bf_filename = os.path.join(folder, 'blockface/sagittal_Hotsauce_10_fixed_with_ice_half_midline-outline-label.nim')
    coronal_bf_filename = os.path.join(folder,
                                       'blockface/sagittal_Hotsauce_100_fixed_with_ice_half_midline-outline-label.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = img_util.imresize(img_data, des_depth=int(math.ceil(depth / 10.)),
                                     des_height=int(math.ceil(height / 10.)),
                                     des_width=int(math.ceil(width / 10.)),
                                     interpolant=Interpolant.Nearest)
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=100., voxel_size_y=100., voxel_size_z=100.)

    logger.info(f'done')


def cut_subregion_for_tagging(folder):
    ra_filename = os.path.join(folder, f'bigsub_merged_jiwon_20201204_fix_midline.reganno')
    out_ra_filename = os.path.join(folder, f'sh_subregion_interpolated_fix_cutline_20201219.reganno')
    img_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')

    if os.path.exists(out_ra_filename):
        logger.info(f'roi to ra {out_ra_filename} done')
    else:
        read_ratio = 8.0
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        img_info = ZImg.readImgInfos(img_filename)[0]
        depth = img_info.depth
        height = math.ceil(img_info.height * scale_down)
        width = math.ceil(img_info.width * scale_down)

        ra_dict = region_annotation.read_region_annotation(ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ra_filename}')

        parent_region_list = [313, 1097, 803, 477, 1089, 315]

        subregion_list = [1939, 1949, 1929, 1979, 1969, 1959,
                          3159, 3158, 3157, 3156, 3155, 3154, 3153,
                          4779, 4778,
                          4777, 4776,
                          4773, 4774,
                          4775,
                          ]

        for region_id, region_props in ra_dict['Regions'].items():
            if region_id in parent_region_list or region_id == -1:
                region_props['ROI'] = None

        for slice in range(depth):
            logger.info(f'slice {slice}')

            annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
            # return a map from region_id to
            # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
            for region_id, slice_rois in region_to_masks.items():
                if region_id not in parent_region_list:
                    continue
                if slice in slice_rois:
                    maskps = slice_rois[slice]
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(height, width), dtype=np.bool)
                        mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        # mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                        mask = cv2.dilate(mask.astype(np.uint8),
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                        annotation_mask[mask] = region_id
            slice_rois = region_to_masks[-1]
            if slice in slice_rois:
                maskps = slice_rois[slice]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = 0  # cut

            for region_id in np.unique(annotation_mask):
                if region_id == 0:
                    continue
                labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                for label in range(1, num_features + 1):
                    mask = labeled_array == label
                    # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5,5)))
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    shapes = nim_roi.label_image_2d_to_spline_shapes(mask)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][-1]['ROI'] is None:
                            ra_dict['Regions'][-1]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][-1]['ROI']:
                            ra_dict['Regions'][-1]['ROI']['SliceROIs'] = {}
                        if slice not in ra_dict['Regions'][-1]['ROI']['SliceROIs']:
                            ra_dict['Regions'][-1]['ROI']['SliceROIs'][slice] = shapes
                        else:
                            ra_dict['Regions'][-1]['ROI']['SliceROIs'][slice].extend(shapes)

        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
        region_annotation.write_region_annotation_dict(ra_dict, out_ra_filename)

        logger.info(f'cut subregions {out_ra_filename} done')


def map_subregion_to_isotropic_blockface(folder: str):
    ra_filename = os.path.join(folder, 'subregion/sh_subregion_interpolation_final_20201220_deform.reganno')
    reduced_ra_filename = os.path.join(folder,
                                       'subregion/sh_subregion_interpolation_final_20201220_deform_bf.reganno')
    if not os.path.exists(reduced_ra_filename):
        ra_dict = region_annotation.read_region_annotation(ra_filename)

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res /= 15.32
            res[:, 0] -= 20
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun)
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict2, lambda s: (s + 7) * 10)
        ra_dict2['VoxelSizeXInUM'] = 10.
        ra_dict2['VoxelSizeYInUM'] = 10.
        ra_dict2['VoxelSizeZInUM'] = 10.
        region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)

    half_ra_filename = os.path.join(folder,
                                    'subregion/sh_subregion_interpolation_final_20201220_deform_bf_half.reganno')
    # if not os.path.exists(half_ra_filename):
    #     ra_dict = region_annotation.read_region_annotation(reduced_ra_filename)
    #
    #     def transform_fun(input_coords: np.ndarray):
    #         assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
    #         res = input_coords.copy()
    #         return None if res[:, 0].mean() < 900 else res
    #
    #     ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun)
    #     ra_dict2['VoxelSizeXInUM'] = 10.
    #     ra_dict2['VoxelSizeYInUM'] = 10.
    #     ra_dict2['VoxelSizeZInUM'] = 10.
    #     region_annotation.write_region_annotation_dict(ra_dict2, half_ra_filename)

    if not os.path.exists(half_ra_filename):
        depth = 1861
        height = 1300
        width = 1800

        ra_dict = region_annotation.read_region_annotation(reduced_ra_filename)
        logger.info(f'finish reading {reduced_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {reduced_ra_filename}')

        for region_id, region_props in ra_dict['Regions'].items():
            region_props['ROI'] = None

        for slice in range(depth):
            logger.info(f'slice {slice}')

            annotation_mask = np.zeros(shape=(height, width), dtype=np.uint16)
            # return a map from region_id to
            # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
            has_shape = False
            for region_id, slice_rois in region_to_masks.items():
                if slice in slice_rois:
                    maskps = slice_rois[slice]
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        has_shape = True
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(height, width), dtype=np.bool)
                        mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        # mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                        mask = cv2.dilate(mask.astype(np.uint8),
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                        annotation_mask[mask] = region_id
            if not has_shape:
                continue
            annotation_mask[:, 0:900] = 0

            for region_id in np.unique(annotation_mask):
                if region_id == 0:
                    continue
                labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                for label in range(1, num_features + 1):
                    mask = labeled_array == label
                    # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5,5)))
                    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                    #                         kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    shapes = nim_roi.label_image_2d_to_spline_shapes(mask)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][region_id]['ROI'] is None:
                            ra_dict['Regions'][region_id]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                        if slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice] = shapes
                        else:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][slice].extend(shapes)

        region_annotation.write_region_annotation_dict(ra_dict, half_ra_filename)

        logger.info(f'{half_ra_filename} done')

    reduced_ra_filename = os.path.join(folder,
                                       'subregion/sh_subregion_interpolation_final_20201220_deform_bf_noi.reganno')
    if not os.path.exists(reduced_ra_filename):
        ra_dict = region_annotation.read_region_annotation(half_ra_filename)

        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict, lambda s: (s - 70) / 10)
        ra_dict2['VoxelSizeXInUM'] = 10.
        ra_dict2['VoxelSizeYInUM'] = 10.
        ra_dict2['VoxelSizeZInUM'] = 100.
        region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)
        logger.info(f'{reduced_ra_filename} done')


def get_cortex_surface(folder: str):
    ra_filename = os.path.join(folder, 'subregion/sh_subregion_interpolation_final_20210216_cortical_fix.reganno')
    ra_1_filename = os.path.join(folder, 'subregion/cortex_surface.reganno')
    ra_0_filename = os.path.join(folder, 'subregion/cortex_bottom.reganno')

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    logger.info(f'finish reading {ra_filename}')

    ra_dict1 = region_annotation.read_region_annotation(ra_1_filename)
    logger.info(f'finish reading {ra_1_filename}')

    ra_dict0 = region_annotation.read_region_annotation(ra_0_filename)
    logger.info(f'finish reading {ra_0_filename}')

    for slice in range(162):
        print(slice)
        if slice >= 24 and slice <= 106:
            continue
        for region_id, region_props in ra_dict['Regions'].items():
            if region_id == 315:
                all_shapes = region_props['ROI']['SliceROIs'][slice]
                assert len(all_shapes) == 1, slice
                all_sub_shapes = all_shapes[0]
                print(len(all_sub_shapes))
                if len(all_sub_shapes) == 1:
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice] = [[all_sub_shapes[0]]]
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Line'
                elif len(all_sub_shapes) == 2 or slice == 19:
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice] = [[all_sub_shapes[0]]]
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Line'
                    ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice] = [[all_sub_shapes[1]]]
                    ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Line'
                else:
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice] = [[all_sub_shapes[0]]]
                    ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Line'
                    all_sub_shapes = ra_dict['Regions'][1009]['ROI']['SliceROIs'][slice][0]
                    ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice] = [[all_sub_shapes[0]]]
                    ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Line'

    region_annotation.write_region_annotation_dict(ra_dict1, os.path.join(folder, 'subregion/cortex_surface_auto.reganno'))
    region_annotation.write_region_annotation_dict(ra_dict0, os.path.join(folder, 'subregion/cortex_bottom_auto.reganno'))

    logger.info(f'done')


def get_cortex_gradient(folder: str):
    ra_filename = os.path.join(folder, 'subregion/sh_subregion_interpolation_final_20210219_cortical_fix_cut_interpolate.reganno')
    ra_1_filename = os.path.join(folder, 'subregion/cortex_surface_auto.reganno')
    ra_0_filename = os.path.join(folder, 'subregion/cortex_bottom_auto.reganno')
    label_img_filename = os.path.join(folder, 'subregion/cortex_layer_label.nim')
    img_filename = os.path.expanduser('~/Documents/jinnylab-annotation-v3_1/Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    cortex_grad_filename = os.path.join(folder, 'subregion/cortex_gradient.nim')

    read_ratio = 16
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    img_info = ZImg.readImgInfos(img_filename)[0]

    ra_dict1 = region_annotation.read_region_annotation(ra_1_filename)
    ra_dict0 = region_annotation.read_region_annotation(ra_0_filename)
    import copy
    # import pyvista as pv
    # from pyvista import examples
    ra_dict_lines = copy.deepcopy(ra_dict1)

    valid_slices = list(range(9,152))
    img = ZImg(cortex_grad_filename)

    def evaluate_spline(pts):
        import scipy.interpolate
        times = np.ndarray(shape=(pts.shape[0],))
        times[0] = 0
        for i in range(1, times.shape[0]):
            times[i] = times[i - 1] + np.linalg.norm(pts[i, :] - pts[i-1, :])
        cs = scipy.interpolate.CubicSpline(times, pts, bc_type='natural')
        xs = np.linspace(times[0], times[-1], num=1000)
        return cs(xs)

    def caculate_pairwise_match_score(pts1, pts0):
        return np.linalg.norm(pts1[:, None, :] - pts0[None, :, :], axis=-1)

    for slice in range(img_info.depth):
        print(slice)
        if not slice in valid_slices:
            continue

        # img_data = img.data[0][0][slice].copy()
        # x = np.arange(0, img_data.shape[1], 1)
        # y = np.arange(0, img_data.shape[0], 1)
        # x, y = np.meshgrid(x, y)
        # zscale = 0.
        # z = img_data[y, x]
        # # Get the points as a 2D NumPy array (N by 3)
        # points = np.c_[x[z>0].reshape(-1), y[z>0].reshape(-1), z[z>0].reshape(-1)*zscale]
        # cloud = pv.PolyData(points)
        # surf = cloud.delaunay_2d()

        pts1 = ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'].copy()
        pts0 = ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'].copy()
        pts1 = evaluate_spline(pts1)
        pts0 = evaluate_spline(pts0)

        # # surf.plot(show_edges=True)
        # p = pv.Plotter(notebook=0)
        # pt1 = surf.find_closest_point((972, 390, 0.82*zscale))
        # pt0 = surf.find_closest_point((1077, 377, 0.895*zscale))
        # print(pt1, pt0)
        # a = surf.geodesic(pt1, pt0)
        # pt1 = surf.find_closest_point((888, 333, 1*zscale))
        # pt0 = surf.find_closest_point((1034, 497, 0.4*zscale))
        # print(pt1, pt0)
        # b = surf.geodesic(pt1, pt0)
        #
        # for i in range(pts1.shape[0]):
        #     pt1 = surf.find_closest_point((pts1[i,0]*scale_down, pts1[i,1]*scale_down, 1*zscale))
        #     min_dist = np.inf
        #     j_idx = 0
        #     for j in range(pts0.shape[0]):
        #         pt0 = surf.find_closest_point((pts0[j,0]*scale_down, pts0[j,1]*scale_down, 0.4*zscale))
        #         dist = surf.geodesic_distance(pt1, pt0)
        #         if dist < min_dist:
        #             min_dist = dist
        #             j_idx = j
        #     pt0 = surf.find_closest_point((pts0[j_idx,0]*scale_down, pts0[j_idx,1]*scale_down, 0.4*zscale))
        #     c = surf.geodesic(pt1, pt0)
        #     p.add_mesh(c, line_width=10, color="red", label="Geodesic Path")
        #
        # p.add_mesh(a + b, line_width=10, color="red", label="Geodesic Path")
        # p.add_mesh(surf, show_edges=True)
        # p.add_legend()
        # p.show()
        # return

        pts1_pts0_dist = caculate_pairwise_match_score(pts1, pts0)
        ind = np.unravel_index(np.argmin(pts1_pts0_dist, axis=None), pts1_pts0_dist.shape)
        first_pt1_idx = ind[0]
        pts1_t = np.vstack((pts1[first_pt1_idx:, :], pts1[0:first_pt1_idx, :]))
        pts1 = pts1_t
        first_pt0_idx = ind[1]
        pts0_t = np.vstack((pts0[first_pt0_idx:, :], pts0[0:first_pt0_idx, :]))
        pts0 = pts0_t
        pts1_pts0_dist = caculate_pairwise_match_score(pts1, pts0)
        ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'][slice] = []
        ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'][slice].append([
            {
                'Type': 'Line',
                'IsAdd': True,
                'Points': np.vstack((pts1[0, :], pts0[0, :]))
            }
        ])
        last_pt0_idx = 0
        for i in range(1, pts1.shape[0]):
            pt0_idx = np.argmin(pts1_pts0_dist[i, last_pt0_idx:])
            pts = np.vstack((pts1[i, :], pts0[last_pt0_idx + pt0_idx, :]))
            last_pt0_idx += pt0_idx
            # print(pts.shape)
            ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'][slice].append([
                {
                    'Type': 'Line',
                    'IsAdd': True,
                    'Points': pts
                }
            ])
        # ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'] = evaluate_spline(pts1)
        # ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Points'] = evaluate_spline(pts0)
        # ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Polygon'
        # ra_dict0['Regions'][-1]['ROI']['SliceROIs'][slice][0][0]['Type'] = 'Polygon'

    # region_annotation.write_region_annotation_dict(ra_dict1, os.path.join(folder, 'subregion/cortex_surface_auto_test.reganno'))
    # region_annotation.write_region_annotation_dict(ra_dict0, os.path.join(folder, 'subregion/cortex_bottom_auto_test.reganno'))
    # return

    region_annotation.write_region_annotation_dict(ra_dict_lines, os.path.join(folder, 'subregion/cortex_surface_lines.reganno'))
    return

    res_height = img_info.height // read_ratio
    res_width = img_info.width // read_ratio
    res = np.zeros((img_info.depth, res_height, res_width), dtype=np.float64)

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
    logger.info(f'finish reading masks from {ra_filename}')

    ra_dict1 = region_annotation.read_region_annotation(ra_1_filename)
    ra_dict1 = region_annotation.transform_region_annotation_dict(ra_dict1, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_1_filename}')
    ra1_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict1)
    logger.info(f'finish reading masks from {ra_1_filename}')

    ra_dict0 = region_annotation.read_region_annotation(ra_0_filename)
    ra_dict0 = region_annotation.transform_region_annotation_dict(ra_dict0, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_0_filename}')
    ra0_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict0)
    logger.info(f'finish reading masks from {ra_0_filename}')

    def is_cortex(region_id):
        if region_id == 315:
            return True
        if region_id <= 0:
            return False
        pid = ra_dict['Regions'][region_id]['ParentID']
        if pid == 315:
            return True
        else:
            return is_cortex(pid)

    valid_slices = list(range(9,152))

    for slice in range(img_info.depth):
        print(slice)
        if not slice in valid_slices:
            continue

        cortex_mask = np.ones(shape=(res_height, res_width), dtype=np.bool)
        # return a map from region_id to
        # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
        for region_id, slice_rois in region_to_masks.items():
            if not is_cortex(region_id):
                continue
            if slice in slice_rois:
                maskps = slice_rois[slice]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(res_height, res_width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                    cortex_mask[mask] = 0

        mask1 = np.zeros(shape=(res_height, res_width), dtype=np.float64)
        slice_rois = ra1_to_masks[-1]
        if slice in slice_rois:
            maskps = slice_rois[slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(res_height, res_width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                mask1[mask] = 1.

        slice_rois = ra0_to_masks[-1]
        if slice in slice_rois:
            maskps = slice_rois[slice]
            for compact_mask, x_start, y_start, _ in maskps:
                if compact_mask.sum() == 0:
                    continue
                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                mask = np.zeros(shape=(res_height, res_width), dtype=np.bool)
                mask[y_start:y_start + compact_mask.shape[0], x_start:x_start + compact_mask.shape[1]] = compact_mask
                mask1[mask] = .4

        import scipy.sparse
        import scipy.interpolate
        coo = scipy.sparse.coo_matrix(mask1)
        zfun_smooth_rbf = scipy.interpolate.Rbf(coo.row, coo.col, coo.data, function='cubic', smooth=0)  # default smooth=0 for interpolation

        mask_out = np.fromfunction(np.vectorize(zfun_smooth_rbf), (res_height, res_width), dtype=np.float64)
        mask_out[cortex_mask] = 0.
        res[slice, :, :] = np.clip(mask_out, 0., 1.0)

    img_util.write_img(os.path.join(folder, 'subregion/cortex_gradient.nim'), res)
    logger.info(f'done')


def get_cortex_layers(folder: str):
    ra_filename = os.path.join(folder, 'subregion/sh_subregion_interpolation_final_20210219_cortical_fix_cut_interpolate.reganno')
    ra_1_filename = os.path.join(folder, 'subregion/cortex_surface_auto.reganno')
    ra_0_filename = os.path.join(folder, 'subregion/cortex_bottom_auto.reganno')
    img_filename = os.path.expanduser('~/Documents/jinnylab-annotation-v3_1/Lemur-H_SMI99_VGluT2_NeuN_all.nim')
    grad_filename = os.path.join(folder, 'subregion/cortex_gradient.nim')

    grad_img = ZImg(grad_filename)
    img_info = grad_img.info
    res = np.zeros((img_info.depth, img_info.height, img_info.width), dtype=np.uint16)
    grad_data = grad_img.data[0][0]
    res[grad_data >= 0.9] = 3221
    res[(grad_data < 0.9) & (grad_data >= 0.7)] = 3222
    res[(grad_data < 0.7) & (grad_data >= 0.6)] = 3223
    res[(grad_data < 0.6) & (grad_data >= 0.5)] = 3224
    res[(grad_data < 0.5) & (grad_data >= 0.4)] = 3225

    img_util.write_img(os.path.join(folder, 'subregion/cortex_layers.nim'), res)
    logger.info(f'done')


def get_cortex_cutlines():
    ra_filename = os.path.join(folder, 'subregion/bigsub_merged_jiwon_20201204.reganno')
    img_filename = os.path.expanduser('~/Documents/jinnylab-annotation-v3_1/Lemur-H_SMI99_VGluT2_NeuN_all.nim')

    read_ratio = 16
    scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
    img_info = ZImg.readImgInfos(img_filename)[0]
    depth = img_info.depth
    height = math.ceil(img_info.height * scale_down)
    width = math.ceil(img_info.width * scale_down)

    ra_dict = region_annotation.read_region_annotation(ra_filename)
    logger.info(f'finish reading {ra_filename}')
    ra_dict_lines = copy.deepcopy(ra_dict)
    for region, region_props in ra_dict_lines['Regions'].items():
        region_props['ROI'] = None

    ra_dict1 = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
    logger.info(f'finish reading {ra_filename}')
    region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict1)
    logger.info(f'finish reading masks from {ra_filename}')

    for slice in range(depth):
        logger.info(f'slice {slice}')

        annotation_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        # return a map from region_id to
        # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
        has_cortex = False
        for region_id, slice_rois in region_to_masks.items():
            if region_id == 315 and slice in slice_rois:
                maskps = slice_rois[slice]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    has_cortex = True
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = 1
        if has_cortex:
            for line in ra_dict1['Regions'][-1]['ROI']['SliceROIs'][slice]:
                pts = line[0]['Points']
                is_cortex_line = annotation_mask[pts[:, 1].astype(int), pts[:, 0].astype(int)].sum() / (1. * pts.shape[0]) > 0.5
                if is_cortex_line:
                    shapes = [[
                        {
                            'Type': 'Line',
                            'IsAdd': True,
                            'Points': pts * read_ratio
                        }
                    ]]
                    if ra_dict_lines['Regions'][-1]['ROI'] is None:
                        ra_dict_lines['Regions'][-1]['ROI'] = {}
                    if 'SliceROIs' not in ra_dict_lines['Regions'][-1]['ROI']:
                        ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'] = {}
                    if slice not in ra_dict_lines['Regions'][-1]['ROI']['SliceROIs']:
                        ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'][slice] = shapes
                    else:
                        ra_dict_lines['Regions'][-1]['ROI']['SliceROIs'][slice].extend(shapes)

    region_annotation.write_region_annotation_dict(ra_dict_lines, os.path.join(folder, 'subregion/cortex_cutlines.reganno'))

    logger.info(f'done')


def make_blockface_ref(folder: str):
    bf_filename = os.path.join(folder, 'blockface/Hotsauce_10_fixed_with_ice_half_midline.nim')
    coronal_bf_filename = os.path.join(folder, 'subregion/coronal_Hotsauce_10x10x100_fixed_with_ice_half_midline.nim')
    if not os.path.exists(coronal_bf_filename):
        img = ZImg(bf_filename)
        img_data = img.data[0]
        nchs, depth, height, width = img_data.shape
        img_data = np.roll(img_data, 20, axis=-1)
        img_data[:, :, :, 0:20] = 0
        img_data = img_data[:, 0:depth:2, :, :].copy()
        img_util.write_img(coronal_bf_filename, img_data, voxel_size_unit=VoxelSizeUnit.um,
                           voxel_size_x=10., voxel_size_y=10., voxel_size_z=100.)

    ra_filename = os.path.join(folder, 'subregion/04_scaled_deformed_annotation.reganno')
    reduced_ra_filename = os.path.join(folder,
                                       'subregion/04_scaled_deformed_annotation_bf.reganno')
    if not os.path.exists(reduced_ra_filename):
        ra_dict = region_annotation.read_region_annotation(ra_filename)

        def transform_fun(input_coords: np.ndarray):
            assert input_coords.ndim == 2 and input_coords.shape[1] == 2, input_coords.shape
            res = input_coords.copy()
            res /= 15.32
            res[:, 0] -= 20
            assert res.ndim == 2 and res.shape[1] == 2, res.shape
            return res

        ra_dict2 = region_annotation.transform_region_annotation_dict(ra_dict, transform_fun)
        ra_dict2 = region_annotation.map_region_annotation_dict_slices(ra_dict2, lambda s: (s + 7) * 1)
        ra_dict2['VoxelSizeXInUM'] = 10.
        ra_dict2['VoxelSizeYInUM'] = 10.
        ra_dict2['VoxelSizeZInUM'] = 100.
        region_annotation.write_region_annotation_dict(ra_dict2, reduced_ra_filename)


def close_and_make_cortex_layers(folder: str):
    ra_filename = os.path.join(folder, 'subregion/04_scaled_deformed_annotation_bf.reganno')
    refined_ra_filename = os.path.join(folder,
                                       'subregion/04_scaled_deformed_annotation_bf_refine2.reganno')
    def reduce_pts(pts):
        assert pts.shape[1] == 2, pts.shape
        res = pts[0, :]
        lastIdx = 0
        for i in range(1, pts.shape[0] - 1):
            if np.linalg.norm(pts[i, :] - pts[lastIdx, :]) >= 30:
                res = np.vstack([res, pts[i, :]])
                lastIdx = i
        res = np.vstack([res, pts[pts.shape[0] - 1, :]])
        return res

    if not os.path.exists(refined_ra_filename):
        ra_dict = region_annotation.read_region_annotation(ra_filename)
        for region_id, region_prop in ra_dict['Regions'].items():
            if region_id != -1:
                continue
            for slice in range(7, 169):
                if slice < 28 or slice > 113:
                    if slice < 164:
                        assert len(region_prop['ROI']['SliceROIs'][slice]) == 4, slice
                    for shape in region_prop['ROI']['SliceROIs'][slice]:
                        pts = shape[0]['Points']
                        pts = np.vstack([pts, pts[0, :]])
                        shape[0]['Points'] = pts
                for shape in region_prop['ROI']['SliceROIs'][slice]:
                    shape[0]['Points'] = reduce_pts(shape[0]['Points'])
        region_annotation.write_region_annotation_dict(ra_dict, refined_ra_filename)

    def first_nonzero_row(a):
        if a.flags.c_contiguous:
            b = a.ravel().view(bool)
            res = b.argmax()
            return res // (a.shape[1] * a.itemsize) if res or b[res] else a.shape[0]
        else:
            b = a.astype(bool).ravel()
            res = b.argmax()
            return res // a.shape[1] if res or b[res] else a.shape[0]

    cut_ra_filename = os.path.join(folder, f'subregion/04_scaled_deformed_annotation_bf_refine4_cut.reganno')
    cut_img_filename = os.path.join(folder, f'subregion/04_scaled_deformed_annotation_bf_refine4_cut.nim')
    if not os.path.exists(cut_ra_filename):
        read_ratio = 0.5
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        depth = 170
        height = math.ceil(1300 * scale_down)
        width = math.ceil(1800 * scale_down)

        ra_dict = region_annotation.read_region_annotation(refined_ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {refined_ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {refined_ra_filename}')

        parent_region_list = [315]

        for region_id, region_props in ra_dict['Regions'].items():
            region_props['ROI'] = None

        out_annotation_mask = np.zeros(shape=(depth, height, width), dtype=np.uint16)
        for slice in range(depth):
            logger.info(f'slice {slice}')

            annotation_mask = np.zeros(shape=(height, width), dtype=np.uint8)
            # return a map from region_id to
            # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
            for region_id, slice_rois in region_to_masks.items():
                if region_id not in parent_region_list:
                    continue
                if slice in slice_rois:
                    maskps = slice_rois[slice]
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        mask = np.zeros(shape=(height, width), dtype=np.bool)
                        mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        # mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                        mask = cv2.dilate(mask.astype(np.uint8),
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                        annotation_mask[mask] = region_id
            slice_rois = region_to_masks[-1]
            if slice in slice_rois:
                maskps = slice_rois[slice]
                for compact_mask, x_start, y_start, _ in maskps:
                    if compact_mask.sum() == 0:
                        continue
                    assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                    mask = np.zeros(shape=(height, width), dtype=np.bool)
                    mask[y_start:y_start + compact_mask.shape[0],
                    x_start:x_start + compact_mask.shape[1]] = compact_mask
                    annotation_mask[mask] = 0  # cut

            for region_id in np.unique(annotation_mask):
                if region_id == 0:
                    continue
                labeled_array, num_features = scipy.ndimage.label(annotation_mask == region_id)
                first_nonzero_rows = np.zeros((num_features,), dtype=np.uint32)
                for label in range(1, num_features + 1):
                    mask = labeled_array == label
                    first_nonzero_rows[label - 1] = first_nonzero_row(mask)
                sorted_labels = np.argsort(first_nonzero_rows) + 1
                for idx, label in enumerate(sorted_labels):
                    if idx > 4:
                        break
                    mask = labeled_array == label
                    # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5,5)))
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    region_id = 3159 - idx
                    out_slice = slice
                    out_annotation_mask[out_slice][mask > 0] = region_id
                    shapes = nim_roi.label_image_2d_to_polygon_shapes(mask)
                    if len(shapes) > 0:
                        if ra_dict['Regions'][region_id]['ROI'] is None:
                            ra_dict['Regions'][region_id]['ROI'] = {}
                        if 'SliceROIs' not in ra_dict['Regions'][region_id]['ROI']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'] = {}
                        if out_slice not in ra_dict['Regions'][region_id]['ROI']['SliceROIs']:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][out_slice] = shapes
                        else:
                            ra_dict['Regions'][region_id]['ROI']['SliceROIs'][out_slice].extend(shapes)

        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
        ra_dict['VoxelSizeXInUM'] = 10.
        ra_dict['VoxelSizeYInUM'] = 10.
        ra_dict['VoxelSizeZInUM'] = 100.
        region_annotation.write_region_annotation_dict(ra_dict, cut_ra_filename)
        out_annotation_mask = img_util.imresize(out_annotation_mask, des_height=1300, des_width=1800,
                                                interpolant=Interpolant.Nearest)
        img_util.write_img(cut_img_filename, out_annotation_mask)


def merge_region_annotation(folder: str):
    start_slice_and_ra_dict = []
    fn1 = os.path.join(folder, 'Garlic', '02_scaled_aligned_annotation.reganno')
    ra_dict1 = region_annotation.read_region_annotation(fn1)
    del ra_dict1['Regions'][315]
    fn2 = os.path.join(folder, 'Garlic', '02_scaled_aligned_annotation_subregion_layer_tagged.reganno')
    ra_dict2 = region_annotation.read_region_annotation(fn2)
    start_slice_and_ra_dict.append((0, ra_dict1))
    start_slice_and_ra_dict.append((0, ra_dict2))
    merged_ra = region_annotation.merge_region_annotation_dicts(start_slice_and_ra_dict)
    combined_filename = os.path.join(folder, 'Garlic', 'combined.reganno')
    region_annotation.write_region_annotation_dict(merged_ra, combined_filename)

    logger.info(f'roi {combined_filename} done')


def cut_cortex_subregion_for_tagging(folder):
    ra_filename = os.path.join(folder, 'Garlic', f'02_scaled_aligned_annotation_subregion_cutline.reganno')
    out_ra_filename = os.path.join(folder, 'Garlic', f'02_scaled_aligned_annotation_subregion_cortex_separated.reganno')
    img_filename = os.path.join(folder, 'jinnylab-annotation-v3_1', f'Lemur-H_SMI99_VGluT2_NeuN_all.nim')

    if os.path.exists(out_ra_filename):
        logger.info(f'roi to ra {out_ra_filename} done')
    else:
        read_ratio = 8.0
        scale_down = 1.0 / read_ratio  # otherwise the mask will be too big
        img_info = ZImg.readImgInfos(img_filename)[0]
        depth = img_info.depth
        height = math.ceil(img_info.height * scale_down)
        width = math.ceil(img_info.width * scale_down)

        # return 0 for left and 1 for right
        def mask_location(mask):
            mask_centroid = scipy.ndimage.measurements.center_of_mass(mask)
            return 0 if mask_centroid[1] < width / 2. else 1

        ra_dict = region_annotation.read_region_annotation(ra_filename)
        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * scale_down)
        logger.info(f'finish reading {ra_filename}')
        region_to_masks = region_annotation.convert_region_annotation_dict_to_masks(ra_dict)
        logger.info(f'finish reading masks from {ra_filename}')

        parent_region_list = [3159, 3158, 3157, 3156, 3155]

        for region_id, region_props in ra_dict['Regions'].items():
            if region_id in parent_region_list or region_id == -1:
                region_props['ROI'] = None

        for slice in range(depth):
            logger.info(f'slice {slice}')

            annotation_mask = np.zeros(shape=(height, width), dtype=np.bool)
            # return a map from region_id to
            # a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
            for region_id, slice_rois in region_to_masks.items():
                if region_id not in parent_region_list:
                   continue
                if slice in slice_rois:
                    maskps = slice_rois[slice]
                    for compact_mask, x_start, y_start, _ in maskps:
                        if compact_mask.sum() == 0:
                            continue
                        assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                        annotation_mask = np.zeros(shape=(height, width), dtype=np.bool)
                        annotation_mask[y_start:y_start + compact_mask.shape[0],
                        x_start:x_start + compact_mask.shape[1]] = compact_mask
                        # mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                        annotation_mask = cv2.dilate(annotation_mask.astype(np.uint8),
                                                     kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                        annotation_mask_location = mask_location(annotation_mask)
                        print(f'annotation mask location {annotation_mask_location}')

                        cutlines_slice_rois = region_to_masks[-1]
                        if slice in cutlines_slice_rois:
                            maskps = cutlines_slice_rois[slice]
                            for compact_mask, x_start, y_start, _ in maskps:
                                if compact_mask.sum() == 0:
                                    continue
                                assert x_start >= 0 and y_start >= 0, (x_start, y_start, compact_mask.shape)
                                mask = np.zeros(shape=(height, width), dtype=np.bool)
                                mask[y_start:y_start + compact_mask.shape[0],
                                x_start:x_start + compact_mask.shape[1]] = compact_mask
                                cutlines_mask_location = mask_location(mask)
                                print(f'cutlines mask location {cutlines_mask_location}')
                                if cutlines_mask_location == annotation_mask_location:
                                    annotation_mask[mask] = False  # cut

                        labeled_array, num_features = scipy.ndimage.label(annotation_mask)
                        for label in range(1, num_features + 1):
                            mask = labeled_array == label
                            # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5,5)))
                            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                            shapes = nim_roi.label_image_2d_to_polygon_shapes(mask)
                            if len(shapes) > 0:
                                if ra_dict['Regions'][-1]['ROI'] is None:
                                    ra_dict['Regions'][-1]['ROI'] = {}
                                if 'SliceROIs' not in ra_dict['Regions'][-1]['ROI']:
                                    ra_dict['Regions'][-1]['ROI']['SliceROIs'] = {}
                                if slice not in ra_dict['Regions'][-1]['ROI']['SliceROIs']:
                                    ra_dict['Regions'][-1]['ROI']['SliceROIs'][slice] = shapes
                                else:
                                    ra_dict['Regions'][-1]['ROI']['SliceROIs'][slice].extend(shapes)

        ra_dict = region_annotation.transform_region_annotation_dict(ra_dict, lambda coords: coords * read_ratio)
        region_annotation.write_region_annotation_dict(ra_dict, out_ra_filename)

        logger.info(f'cut subregions {out_ra_filename} done')


def print_latex_structure_list():
    annotation_filename = '/Users/feng/Documents/Garlic/combined.reganno'
    import anytree
    id_to_node = region_annotation.read_region_annotation_tree(annotation_filename)
    for node in anytree.PreOrderIter(id_to_node[997]):
        if node.name.startswith('Layer'):
            continue
        str = ''
        for i in range(node.depth):
            str += '\lvl'
        name = node.name.replace('&', '/').replace('_', '/')
        abbreviation = node.abbreviation.replace('&', '/').replace('_', '/')
        str += f' {name} & {abbreviation} & \cellcolor[RGB]{{ {node.color[0]},{node.color[1]},{node.color[2]} }}\\\\'
        print(str)

    print()
    for node in anytree.PreOrderIter(id_to_node[997]):
        if node.name.startswith('Layer'):
            continue
        name = node.name.replace('&', '/').replace('_', '/')
        str = f'.{node.depth + 1} {name}.'
        print(str)

    print()
    for node in anytree.PreOrderIter(id_to_node[997]):
        if node.name.startswith('Layer'):
            continue
        abbreviation = node.abbreviation.replace('&', '/').replace('_', '/')
        str = f'.1 {abbreviation}.'
        print(str)

    print()
    for node in anytree.PreOrderIter(id_to_node[997]):
        if node.name.startswith('Layer'):
            continue
        str = f'\cellcolor[RGB]{{ {node.color[0]},{node.color[1]},{node.color[2]} }}\\\\'
        print(str)

    print()
    for node in anytree.PreOrderIter(id_to_node[997]):
        if node.name.startswith('Layer'):
            continue
        name = node.name.replace('&', '/').replace('_', '/')
        abbreviation = node.abbreviation.replace('&', '/').replace('_', '/')
        str = f'\\newabbreviation{{{abbreviation}}}{{{abbreviation}}}{{{name}}}'
        print(str)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # path = '/Volumes/T7 Touch/'
    # folder = os.path.join(path)
    folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')

    folder = os.path.join(os.path.expanduser('~/Documents'))

    # align_with_hj_transform_all_images(folder)
    # align_rois_with_hj_transform_all_images(folder)
    # flip_rois_for_manual_tagging(folder)
    # convert_rois_to_region_annotation_for_tagging(folder)
    # do_tag_interpolation(folder)
    # flip_region_annotation_back_after_finishing_tagging(folder)
    # do_lemur_bigregion_detection(folder)
    # do_lemur_bigregion_detection_v2(folder)
    # do_lemur_bigregion_detection_v4(folder)
    # merge_edited_annotations(folder)
    # reduce_annotation_slices(folder)
    # shift_jiwon_blockface_annotation(folder)
    # fix_jiwon_blockface_annotation(folder)
    # build_sagittal_blockface_and_jiwon_annotation(folder)
    # cut_subregion_for_tagging(folder)
    # map_subregion_to_isotropic_blockface(folder)
    # get_cortex_surface(folder)
    # get_cortex_gradient(folder)
    # get_cortex_layers(folder)
    # get_cortex_cutlines()
    # make_blockface_ref(folder)
    # close_and_make_cortex_layers(folder)
    # do_tag_interpolation(folder)
    # merge_region_annotation(folder)
    # cut_cortex_subregion_for_tagging(folder)
    print_latex_structure_list()
    sys.exit(1)
