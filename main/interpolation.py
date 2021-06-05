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
import lap
import cv2

from zimg import *
from utils import io
from utils import img_util
from utils import nim_roi
from utils import region_annotation
from utils.logger import setup_logger


logger = setup_logger()

def do_tag_interpolation(folder: str):
    input_ra_filename = os.path.join(folder, f'sh_subregion_interpolated_20201220.reganno')

    interpolated_ra_filename = os.path.join(folder, f'sh_subregion_tag_interpolated_str1_20201220.reganno')

    if os.path.exists(interpolated_ra_filename):
        logger.info(f'tag interpolation {interpolated_ra_filename} done')
    else:
        scale_down = 1.0/8  # otherwise the mask will be too big
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
            # print("region to annotated slices : ", len(region_to_annotated_slices), region_to_annotated_slices)

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
                    assert ref_maskp[0].sum() > 0
                    ref_region_list.append(region)
                    ref_mask, ref_x, ref_y, _ = ref_maskp
                    n1_n2_match_IOU_metric[n1_idx, n2_idx] = get_overlap_between_masks(maskp, ref_maskp)
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
            # if type(masks) == list:
            #     if masks <= 0:
            #         masks.clear()
            #         pass
            #     else:
            #         continue

            if len(region_to_ref_masks) == 0: # if there's no region left to be tagged
                for region, annotated_slices in region_to_annotated_slices.items():
                    for annotated_slice in annotated_slices:
                        if annotated_slice < slice: # if first slice is less than last slice, that is if we have to keep tagging, then !
                            region_to_ref_masks[region] = region_to_masks[region][annotated_slice] # tag the region in region_to_ref_masks to undefined region
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
            matching_strategy = 1
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

                        # print("region : ", region)

                        continue
                    else:
                        for ref_mask in ref_masks:
                            all_ref_masks.append((ref_mask, region))

                            # print("ref mask : ", len(ref_mask))
                            # print("all ref masks : ", len(all_ref_masks))

                n1_n2_match, ref_region_list = caculate_pairwise_match_score(masks, all_ref_masks)
                # Assignment. `x[i]` specifies the column to which row `i` is assigned.
                # Assignment. `y[j]` specifies the row to which column `j` is assigned.

                # print("match score : ", len(n1_n2_match), "\n", n1_n2_match)

                cost, x, y = lap.lapjv(n1_n2_match, extend_cost=True, cost_limit=2.0, return_cost=True)
                # print("lapjv : ", cost, x, y)
                # print("region to ref masks : ", len(region_to_ref_masks))
                # print("ref region list : ", len(ref_region_list), ref_region_list)

                print("annotated regions in current slice : ", annotated_regions_in_current_slice)

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

    input_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                     f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging_sh_interpolated.reganno')
    interpolated_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                            f'Lemur-H_SMI99_VGluT2_NeuN_all_tagged.reganno')
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


if __name__ == "__main__":
    folder = os.path.dirname('/Users/fenglab/Downloads/subregion_annotation/')
    #align_with_hj_transform_all_images(folder)
    #align_rois_with_hj_transform_all_images(folder)
    #flip_rois_for_manual_tagging(folder)
    # convert_rois_to_region_annotation_for_tagging(folder)
    do_tag_interpolation(folder)
    #flip_region_annotation_back_after_finishing_tagging(folder)
