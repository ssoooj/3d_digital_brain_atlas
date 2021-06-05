import h5py
import logging
import cv2
import os
import copy
import numpy as np
import scipy.interpolate
import scipy.ndimage
import imageio
import PIL.ImageDraw
import PIL.Image

from zimg import *
from utils import io

logger = logging.getLogger(__name__)


def read_roi_group(roi_group):
    res = {}
    roi_attrs = roi_group.attrs
    res['Version'] = int(roi_attrs['Version'])
    res['SliceNumber'] = int(roi_attrs['SliceNumber'])
    res['SliceROIs'] = {}
    max_slice = -1
    for slice_idx in range(res['SliceNumber']):
        slice_group = roi_group[f'Slice{slice_idx + 1}']
        slice_attrs = slice_group.attrs
        slice = int(slice_attrs['Slice'])
        max_slice = max(max_slice, slice)
        shape_number = int(slice_attrs['ShapeNumber'])
        res['SliceROIs'][slice] = []
        for shape_idx in range(shape_number):
            if roi_attrs['Version'] == 100:
                shape_group = slice_group[f'Shape{shape_idx + 1}']
                shape_attrs = shape_group.attrs
                type = shape_attrs['Type']
                is_add = shape_attrs['IsAdd'] != 0
                points = shape_group['Points'][()]
                res['SliceROIs'][slice].append([{
                    'Type': type,
                    'IsAdd': is_add,
                    'Points': points
                }])
            else:
                all_shape_group = slice_group[f'Shape{shape_idx + 1}']
                all_shape_attrs = all_shape_group.attrs
                sub_shape_number = int(all_shape_attrs['SubShapeNumber'])
                all_sub_shapes = []
                for subshape_idx in range(sub_shape_number):
                    shape_group = all_shape_group[f'SubShape{subshape_idx + 1}']
                    shape_attrs = shape_group.attrs
                    type = shape_attrs['Type']
                    is_add = shape_attrs['IsAdd'] != 0
                    points = shape_group['Points'][()]
                    all_sub_shapes.append({
                        'Type': type,
                        'IsAdd': is_add,
                        'Points': points
                    })
                res['SliceROIs'][slice].append(all_sub_shapes)
    res['MaxSlice'] = max_slice
    return res


def read_roi(roi_name: str):
    with h5py.File(roi_name, mode='r', libver='latest') as roi:
        assert 'ROI' in roi, 'can not locate ROI in file'
        roi_group = roi['ROI']
        return read_roi_group(roi_group)


def write_roi_dict_to_group(roi_dict: dict, roi_group):
    roi_attrs = roi_group.attrs
    roi_attrs.create('Version', 200, dtype=np.int32)
    slice_number = 0
    for slice, sliceROIs in roi_dict['SliceROIs'].items():
        if not sliceROIs:
            continue
        slice_group = roi_group.create_group(f'Slice{slice_number + 1}')
        slice_attrs = slice_group.attrs
        slice_attrs.create('Slice', slice, dtype=np.int32)
        slice_attrs.create('ShapeNumber', len(sliceROIs), dtype=np.int32)
        for shape_idx, shapeOps in enumerate(sliceROIs):
            all_shape_group = slice_group.create_group(f'Shape{shape_idx + 1}')
            all_shape_attrs = all_shape_group.attrs
            all_shape_attrs.create('SubShapeNumber', len(shapeOps), dtype=np.int32)
            for subshape_idx, shapeOp in enumerate(shapeOps):
                shape_group = all_shape_group.create_group(f'SubShape{subshape_idx + 1}')
                shape_attrs = shape_group.attrs
                shape_attrs['Type'] = shapeOp['Type'].encode('utf-8')
                shape_attrs.create('IsAdd', 1 if shapeOp['IsAdd'] else 0, dtype=np.int32)
                shape_group.create_dataset('Points', data=shapeOp['Points'].astype(np.float64))
        slice_number += 1

    roi_attrs.create('SliceNumber', slice_number, dtype=np.int32)


def write_roi_dict(roi_dict: dict, roi_name: str):
    with h5py.File(roi_name, mode='w', libver='latest') as roi:
        roi_group = roi.create_group('ROI')
        write_roi_dict_to_group(roi_dict, roi_group)


def merge_roi_dicts(start_slice_and_roi_dict: list):
    res = None
    for start_slice, roi_dict in start_slice_and_roi_dict:
        if res is None:
            res = copy.deepcopy(roi_dict)
            res['SliceROIs'] = {}

        for slice, sliceROIs in roi_dict['SliceROIs'].items():
            target_slice = start_slice + slice
            if target_slice in res['SliceROIs']:
                res['SliceROIs'][target_slice].extend(sliceROIs)
            else:
                res['SliceROIs'][target_slice] = sliceROIs
    res['SliceNumber'] = len(res['SliceROIs'])
    return res


# transform_fun take nx2 ndarray as input and generate nx2 ndarray or None
def transform_roi_dict(roi_dict: dict, transform_fun, only_apply_to_slices = None):
    res = copy.deepcopy(roi_dict)
    for slice, sliceROIs in res['SliceROIs'].items():
        if only_apply_to_slices is not None and slice not in only_apply_to_slices:
            continue
        for si, shape in enumerate(sliceROIs):
            for subShape in shape:
                subShape['Points'] = transform_fun(subShape['Points'])
            sliceROIs[si] = [subshape for subshape in shape if subshape['Points'] is not None]
        res['SliceROIs'][slice] = [shape for shape in sliceROIs if shape]
    res['SliceROIs'] = {slice: sliceROIs for (slice, sliceROIs) in res['SliceROIs'].items() if sliceROIs}
    return res


def map_roi_dict_slices(roi_dict: dict, map_slice_fun):
    res = copy.deepcopy(roi_dict)
    res['SliceROIs'].clear()
    for slice, sliceROIs in roi_dict['SliceROIs'].items():
        if map_slice_fun(slice) is not None and map_slice_fun(slice) >= 0:
            res['SliceROIs'][map_slice_fun(slice)] = copy.deepcopy(sliceROIs)
    if not res['SliceROIs']:
        res = None
    return res


# return map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
def convert_roi_dict_to_masks(roi_dict: dict):
    masks = {}
    for slice, shapes in roi_dict['SliceROIs'].items():
        masks[slice] = []
        for shape in shapes:
            sub_shapes = []
            for sub_shape in shape:
                # print(shape['Points'].shape, shape['Type'], shape['IsAdd'])
                sub_shapes.append((sub_shape['Points'], sub_shape['Type'], sub_shape['IsAdd']))
            mask_img, x_start, y_start = ZROIUtils.shapeToMask(sub_shapes)
            masks[slice].append((np.array([]) if mask_img.isEmpty() else mask_img.data[0][0][0] > 0, x_start, y_start, shape))
    return masks


# return list (slice) of list (instance) of (mask (np.uint8 2d), x_start, y_start), mask can be empty, x/y_start can be negative
def read_roi_to_masks(roi_name: str):
    return convert_roi_dict_to_masks(read_roi(roi_name))


def read_roi_to_label_image(roi_name: str, *, depth: int = 0, height: int = 0, width: int = 0,
                            labe_image_dtype = None, save_label_filename: str = ""):
    res = read_roi(roi_name)
    max_slice = res['MaxSlice']
    all_masks = []
    if depth == 0:
        depth = max_slice + 1
    cal_height = height == 0
    cal_width = width == 0
    shape_id = 0
    for slice in range(depth):
        slice_masks = []
        if slice in res['SliceROIs']:
            for shapes in res['SliceROIs'][slice]:
                sub_shapes = []
                for shape in shapes:
                    # print(shape['Points'].shape, shape['Type'], shape['IsAdd'])
                    sub_shapes.append((shape['Points'], shape['Type'], shape['IsAdd']))
                mask_img, x_start, y_start = ZROIUtils.shapeToMask(sub_shapes)
                if mask_img.isEmpty():
                    continue
                mask = mask_img.data[0][0][0] > 0
                if cal_height:
                    height = max(height, mask.shape[0] + y_start)
                if cal_width:
                    width = max(width, mask.shape[1] + x_start)
                x_valid_start = max(0, x_start)
                y_valid_start = max(0, y_start)
                y_end = min(y_start + mask.shape[0], height)
                x_end = min(x_start + mask.shape[1], width)
                if x_end > x_valid_start and y_end > y_valid_start:
                    shape_id += 1
                    slice_masks.append((mask, x_start, y_start, x_valid_start, x_end, y_valid_start, y_end, shape_id))
        all_masks.append(slice_masks)

    if labe_image_dtype is None:
        if shape_id <= np.iinfo(np.uint8).max:
            labe_image_dtype = np.uint8
        elif shape_id <= np.iinfo(np.uint16).max:
            labe_image_dtype = np.uint16
        elif shape_id <= np.iinfo(np.uint32).max:
            labe_image_dtype = np.uint32
        else:
            assert shape_id <= np.iinfo(np.uint64).max, shape_id
            labe_image_dtype = np.uint64
    else:
        assert shape_id <= np.iinfo(labe_image_dtype).max

    label_image = np.full(shape=(1, depth, height, width), fill_value=0, dtype=labe_image_dtype)
    for slice in range(depth):
        for mask, x_start, y_start, x_valid_start, x_end, y_valid_start, y_end, shape_id in all_masks[slice]:
            label_image[0, slice, y_valid_start:y_end, x_valid_start:x_end] = \
                mask[y_valid_start - y_start:y_end - y_start, x_valid_start - x_start:x_end - x_start] * shape_id
    if save_label_filename:
        img = ZImg(label_image)
        img.save(save_label_filename)
    return label_image[0]


# return list of nx2 polygons
def mask_to_polygons(mask: np.ndarray, x_start: int = 0, y_start: int = 0, use_cv2: bool = True,
                     remove_small: bool = True):
    res = []
    if mask.size == 0 or mask.max() == 0:
        return res
    if use_cv2:
        contours, hierarchy = cv2_util.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # assert len(contours) == 1, contours
        # if len(contours) > 1:
        #     logger.warning(f'get multiple contours {contours}')
    else:
        import skimage.measure
        contours = skimage.measure.find_contours(mask.astype('uint8'), level=0.5)
    for contour in contours:
        if use_cv2:
            assert contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, contour
            contour = np.squeeze(contour, axis=1)
            # print(contour.dtype)
        else:
            assert contour.ndim == 2 and contour.shape[0] >= 3 and contour.shape[1] == 2, contour
            contour = np.flip(contour, axis=1)
            # print(contour.dtype)
        if remove_small and contour.shape[0] <= 3:
            continue
        # make it close
        if not np.all(contour[0, :] == contour[-1, :]):
            contour = np.vstack([contour, contour[0, :]])
        contour[:, 0] += x_start
        contour[:, 1] += y_start
        res.append(contour.astype(np.float64))
    return res


# return list of nx2 splines sampled from polygon
def mask_to_sampled_splines(mask: np.ndarray, x_start: int = 0, y_start: int = 0, use_cv2: bool = True,
                            remove_small: bool = True):
    res = []
    if mask.size == 0 or mask.max() == 0:
        return res
    if use_cv2:
        contours, hierarchy = cv2_util.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # assert len(contours) == 1, contours
        # if len(contours) > 1:
        #     logger.warning(f'get multiple contours {contours}')
    else:
        import skimage.measure
        contours = skimage.measure.find_contours(mask.astype('uint8'), level=0.5)
    for contour in contours:
        if use_cv2:
            assert contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, contour
            contour = np.squeeze(contour, axis=1)
        else:
            assert contour.ndim == 2 and contour.shape[1] == 2, contour
            contour = np.flip(contour, axis=1)
        if remove_small and contour.shape[0] <= 3:
            continue
        # take control points every dst pixels
        dst = np.clip(contour.shape[0] // 20, 1, 30)
        contour = contour[::dst, :]
        # make it close
        if not np.all(contour[0, :] == contour[-1, :]):
            contour = np.vstack([contour, contour[0, :]])
        contour[:, 0] += x_start
        contour[:, 1] += y_start
        res.append(contour.astype(np.float64))
    return res


# return mask (np.uint8 2d), x_start, y_start, mask can be empty, x/y_start can be negative
def spline_to_tight_mask(spline: np.ndarray):
    # print()
    # print(spline)
    mask_img, x_start, y_start = ZROIUtils.splineToMask(spline)
    # print(mask_img)
    # print(mask_img.data)
    # print(mask_img.data[0][0][0].shape)
    return np.array([]) if mask_img.isEmpty() else mask_img.data[0][0][0].copy(), x_start, y_start


def polygon_to_tight_mask(poly: np.ndarray):
    mask_img, x_start, y_start = ZROIUtils.polygonToMask(poly)
    return np.array([]) if mask_img.isEmpty() else mask_img.data[0][0][0].copy(), x_start, y_start


def rect_to_tight_mask(rect: np.ndarray):
    mask_img, x_start, y_start = ZROIUtils.rectToMask(rect)
    return np.array([]) if mask_img.isEmpty() else mask_img.data[0][0][0].copy(), x_start, y_start


def ellipse_to_tight_mask(ellipse: np.ndarray):
    mask_img, x_start, y_start = ZROIUtils.ellipseToMask(ellipse)
    return np.array([]) if mask_img.isEmpty() else mask_img.data[0][0][0].copy(), x_start, y_start


# nx2 spline to list of nx2 polygons
def spline_to_polygons(spline: np.ndarray):
    return mask_to_polygons(*spline_to_tight_mask(spline))


# 2x2 ellipse rect bound to nx2 polygon
def ellipse_to_polygon(ellipse: np.ndarray):
    polys = mask_to_polygons(*ellipse_to_tight_mask(ellipse))
    assert len(polys) == 1, ellipse
    return polys[0]


# 2x2 rect to nx2 polygon
def rect_to_polygon(rect: np.ndarray):
    polys = mask_to_polygons(*rect_to_tight_mask(rect))
    assert len(polys) == 1, rect
    return polys[0]


def tight_mask_to_single_connected_mask(mask: np.ndarray, x_start: int = 0, y_start: int = 0,
                                        height: int = 0, width: int = 0):
    print(mask.shape)
    imllabel, numlabel = scipy.ndimage.label(mask, structure=scipy.ndimage.generate_binary_structure(2, 2))
    assert numlabel == 1, (mask.nonzero(), x_start, y_start, height, width)
    if height == 0:
        height = mask.shape[0] + y_start
    if width == 0:
        width = mask.shape[1] + x_start
    if height <= 0 or width <= 0:
        return np.array([])
    x_valid_start = max(0, x_start)
    y_valid_start = max(0, y_start)
    res = np.full(shape=(height, width), fill_value=False, dtype=np.bool)
    y_end = min(y_start + mask.shape[0], height)
    x_end = min(x_start + mask.shape[1], width)
    res[y_valid_start:y_end, x_valid_start:x_end] = \
        mask[y_valid_start - y_start:y_end - y_start, x_valid_start - x_start:x_end - x_start]
    return res


def spline_to_single_connected_mask(spline: np.ndarray, height: int = 0, width: int = 0):
    mask, x_start, y_start = spline_to_tight_mask(spline)
    try:
        return tight_mask_to_single_connected_mask(mask, x_start, y_start, height, width)
    except AssertionError:
        write_spline_rois([[spline]], os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_spline_error.nimroi'))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_spline_error.tif'), mask)
        raise


def polygon_to_single_connected_mask(poly: np.ndarray, height: int = 0, width: int = 0):
    mask, x_start, y_start = polygon_to_tight_mask(poly)
    try:
        return tight_mask_to_single_connected_mask(mask, x_start, y_start, height, width)
    except AssertionError:
        write_polygon_rois([[poly]], os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.nimroi'))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.tif'), mask)
        raise


def polygon_to_single_connected_mask_coco(poly: np.ndarray, height: int = 0, width: int = 0):
    import pycocotools.mask as mask_util
    max_v = np.ceil(poly.max(axis=0))
    if height == 0:
        height = max_v[1] + 2
    if width == 0:
        width = max_v[0] + 2
    rle = mask_util.frPyObjects([poly.flatten(order='C').tolist()], height, width)[0]
    mask = mask_util.decode(rle) > 0
    imllabel, numlabel = scipy.ndimage.label(mask, structure=scipy.ndimage.generate_binary_structure(2, 2))
    if numlabel != 1:
        write_polygon_rois([[poly]], os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.nimroi'))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.tif'), mask)
    assert numlabel == 1, poly
    return mask


def polygon_to_single_connected_mask_exact(poly: np.ndarray, height: int = 0, width: int = 0):
    max_v = np.ceil(poly.max(axis=0))
    if height == 0:
        height = max_v[1] + 2
    if width == 0:
        width = max_v[0] + 2
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    PIL.ImageDraw.Draw(mask).polygon(xy=poly.flatten(order='C').tolist(), outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    imllabel, numlabel = scipy.ndimage.label(mask, structure=scipy.ndimage.generate_binary_structure(2, 2))
    if numlabel != 1:
        write_polygon_rois([[poly]], os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.nimroi'))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_error.tif'), mask)
    assert numlabel == 1, poly
    return mask


# return rect rois, list of nx4 ndarray in xyxy format
def read_rect_rois(roi_name: str):
    res = read_roi(roi_name)
    max_slice = res['MaxSlice']
    rois = []
    for slice in range(max_slice + 1):
        rects = []
        if slice in res['SliceROIs']:
            for shapes in res['SliceROIs'][slice]:
                for shape in shapes:
                    if shape['Type'] == 'Rect':
                        rects.append([shape['Points'][0, 0], shape['Points'][0, 1],
                                      shape['Points'][1, 0], shape['Points'][1, 1]])
        rects = np.array(rects)
        rois.append(rects)
    return rois


# rois, list of nx4 ndarray in xyxy format
def write_rect_rois(rois: list, roi_name: str):
    with h5py.File(roi_name, mode='w', libver='latest') as roi:
        roi_group = roi.create_group('ROI')
        roi_attrs = roi_group.attrs
        roi_attrs.create('Version', 100, dtype=np.int32)
        slice_number = 0
        for slice, bboxarray in enumerate(rois):
            if bboxarray.size == 0:
                continue
            assert bboxarray.shape[1] == 4, bboxarray.shape
            if bboxarray.dtype != np.float64:
                bboxarray = bboxarray.astype(np.float64)
            slice_group = roi_group.create_group(f'Slice{slice_number + 1}')
            slice_attrs = slice_group.attrs
            slice_attrs.create('Slice', slice, dtype=np.int32)
            slice_attrs.create('ShapeNumber', bboxarray.shape[0], dtype=np.int32)
            for shape_idx, bbox in enumerate(bboxarray):
                shape_group = slice_group.create_group(f'Shape{shape_idx + 1}')
                shape_attrs = shape_group.attrs
                shape_attrs.create('IsAdd', 1, dtype=np.int32)
                shape_attrs['Type'] = b'Rect'
                shape_group.create_dataset('Points', shape=(2, 2), data=bbox)

            slice_number += 1

        roi_attrs.create('SliceNumber', slice_number, dtype=np.int32)


# return polygon rois, list of list of nx2 points: xyxyxyxy...
def read_polygon_rois(roi_name: str):
    res = read_roi(roi_name)
    max_slice = res['MaxSlice']
    rois = []
    for slice in range(max_slice + 1):
        polys = []
        if slice in res['SliceROIs']:
            for shapes in res['SliceROIs'][slice]:
                for shape in shapes:
                    if shape['Type'] == 'Polygon':
                        polys.append(shape['Points'])
        rois.append(polys)
    return rois


# rois, list of list of nx2 points: xyxyxyxy...
def write_polygon_rois(rois: list, roi_name: str, type: str='Polygon'):
    with h5py.File(roi_name, mode='w', libver='latest') as roi:
        roi_group = roi.create_group('ROI')
        roi_attrs = roi_group.attrs
        roi_attrs.create('Version', 100, dtype=np.int32)
        slice_number = 0
        for slice, polylist in enumerate(rois):
            if not polylist:
                continue
            slice_group = roi_group.create_group(f'Slice{slice_number + 1}')
            slice_attrs = slice_group.attrs
            slice_attrs.create('Slice', slice, dtype=np.int32)
            slice_attrs.create('ShapeNumber', len(polylist), dtype=np.int32)
            for shape_idx, poly in enumerate(polylist):
                assert poly.ndim == 2 and poly.shape[0] >= 4 and poly.shape[1] == 2 and (poly[0, :] == poly[-1, :]).all(), poly
                shape_group = slice_group.create_group(f'Shape{shape_idx + 1}')
                shape_attrs = shape_group.attrs
                shape_attrs.create('IsAdd', 1, dtype=np.int32)
                shape_attrs['Type'] = type.encode('utf-8')
                if poly.dtype != np.float64:
                    poly = poly.astype(np.float64)
                shape_group.create_dataset('Points', data=poly)

            slice_number += 1

        roi_attrs.create('SliceNumber', slice_number, dtype=np.int32)


# return spline rois, list of list of nx2 points: xyxyxyxy...
def read_spline_rois(roi_name: str, convert_to_polygon: bool=False):
    res = read_roi(roi_name)
    max_slice = res['MaxSlice']
    rois = []
    for slice in range(max_slice + 1):
        splines = []
        if slice in res['SliceROIs']:
            for shapes in res['SliceROIs'][slice]:
                for shape in shapes:
                    if shape['Type'] == 'Spline':
                        if convert_to_polygon:
                            poly = spline_to_polygons(shape['Points'])
                            if len(poly) == 0:
                                logger.warning(f'empty after converting {shape["Points"]} from slice {slice}')
                            elif len(poly) > 1:
                                logger.warning(f'more than 1 polys after converting {shape["Points"]} from slice {slice}')
                            else:
                                splines.append(poly[0])
                        else:
                            splines.append(shape['Points'])
        rois.append(splines)
    return rois


# rois, list of list of nx2 points: xyxyxyxy...
def write_spline_rois(rois: list, roi_name: str):
    write_polygon_rois(rois, roi_name, type='Spline')


def label_image_to_roi(label_image: str, roi_name: str, sample_to_spline: bool=False):
    label_img = ZImg(label_image)
    assert len(label_img.data) == 1 and label_img.data[0].shape[0] == 1, label_img.data[0].shape
    label_data = label_img.data[0][0]
    print(label_data.shape)
    rois = []
    for slice_label in label_data:
        labels = np.unique(slice_label)
        roi = []
        labels = labels[labels != 0]
        for label in labels:
            mask = slice_label == label
            if sample_to_spline:
                for spline in mask_to_sampled_splines(mask):
                    roi.append(spline)
            else:
                for poly in mask_to_polygons(mask):
                    roi.append(poly)
        rois.append(roi)

    if sample_to_spline:
        write_spline_rois(rois, roi_name)
    else:
        write_polygon_rois(rois, roi_name)


# rois, list of nx2 spline points: xyxyxyxy...
def spline_rois_to_label_image(rois: list, height: int, width: int, dtype=None):
    if dtype is None:
        if len(rois) <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        elif len(rois) <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif len(rois) <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            assert len(rois) <= np.iinfo(np.uint64).max, len(rois)
            dtype = np.uint64
    else:
        assert len(rois) <= np.iinfo(dtype).max

    res = np.full(shape=(height, width), fill_value=0, dtype=dtype)
    current_label = 1

    for roi in rois:
        res[spline_to_single_connected_mask(roi, height, width)] = current_label
        current_label += 1

    return res


def mask_2d_to_spline_shapes(mask: np.ndarray, x_start: int = 0, y_start: int = 0):
    assert mask.ndim == 2, mask.shape
    res = []
    if mask.size == 0 or mask.max() == 0:
        return res
    contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE,
                                           offset=(x_start, y_start))

    import anytree
    idx_to_node = {}
    for idx, contour in enumerate(contours):
        assert contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, contour
        contour = np.squeeze(contour, axis=1)
        if contour.shape[0] <= 3:
            continue
        # take control points every dst pixels
        dst = np.clip(contour.shape[0] // 20, 1, 30)
        contour = contour[::dst, :]
        # make it close
        if not np.all(contour[0, :] == contour[-1, :]):
            contour = np.vstack([contour, contour[0, :]])
        idx_to_node[idx] = anytree.AnyNode(id=idx,
                                           subshape={
                                               'Type': 'Spline',
                                               'IsAdd': True,
                                               'Points': contour
                                           }
                                           )
    assert hierarchy.shape[0] == 1 and hierarchy.shape[1] == len(contours), (hierarchy.shape, len(contours))
    for idx, node_info in enumerate(hierarchy[0]):
        if idx not in idx_to_node:
            continue
        assert node_info.shape[0] == 4, node_info.shape
        if node_info[3] >= 0:
            if node_info[3] not in idx_to_node:
                continue
            idx_to_node[idx].parent = idx_to_node[node_info[3]]

    for idx, node in idx_to_node.items():
        if not node.is_root:
            continue
        shape = []
        for des in anytree.LevelOrderIter(node):
            if len(des.ancestors) % 2 == 1:
                des.subshape['IsAdd'] = False
            shape.append(des.subshape)
        res.append(shape)

    return res


def label_image_2d_to_spline_shapes(label_image: np.ndarray, x_start: int = 0, y_start: int = 0):
    assert label_image.ndim == 2, label_image.shape
    labels = np.unique(label_image)
    res = []
    labels = labels[labels != 0]
    for label in labels:
        mask = label_image == label
        res.extend(mask_2d_to_spline_shapes(mask, x_start, y_start))
    return res


def mask_2d_to_polygon_shapes(mask: np.ndarray, x_start: int = 0, y_start: int = 0):
    assert mask.ndim == 2, mask.shape
    res = []
    if mask.size == 0 or mask.max() == 0:
        return res
    contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                           offset=(x_start, y_start))

    import anytree
    idx_to_node = {}
    for idx, contour in enumerate(contours):
        assert contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, contour
        contour = np.squeeze(contour, axis=1)
        if contour.shape[0] <= 3:
            continue
        # make it close
        if not np.all(contour[0, :] == contour[-1, :]):
            contour = np.vstack([contour, contour[0, :]])
        idx_to_node[idx] = anytree.AnyNode(id=idx,
                                           subshape={
                                               'Type': 'Polygon',
                                               'IsAdd': True,
                                               'Points': contour
                                           }
                                           )
    assert hierarchy.shape[0] == 1 and hierarchy.shape[1] == len(contours), (hierarchy.shape, len(contours))
    for idx, node_info in enumerate(hierarchy[0]):
        if idx not in idx_to_node:
            continue
        assert node_info.shape[0] == 4, node_info.shape
        if node_info[3] >= 0:
            if node_info[3] not in idx_to_node:
                continue
            idx_to_node[idx].parent = idx_to_node[node_info[3]]

    for idx, node in idx_to_node.items():
        if not node.is_root:
            continue
        shape = []
        for des in anytree.LevelOrderIter(node):
            if len(des.ancestors) % 2 == 1:
                des.subshape['IsAdd'] = False
            shape.append(des.subshape)
        res.append(shape)

    return res


def label_image_2d_to_polygon_shapes(label_image: np.ndarray, x_start: int = 0, y_start: int = 0):
    assert label_image.ndim == 2, label_image.shape
    labels = np.unique(label_image)
    res = []
    labels = labels[labels != 0]
    for label in labels:
        mask = label_image == label
        res.extend(mask_2d_to_polygon_shapes(mask, x_start, y_start))
    return res


if __name__ == "__main__":
    roi_name = os.path.join(os.path.expanduser('~'), 'Downloads', 'test.nimroi')
    rois = read_roi(roi_name)
    masks = read_roi_to_masks(roi_name)
    read_roi_to_label_image(roi_name, save_label_filename=os.path.join(os.path.expanduser('~'), 'Downloads', 'test.label.tif'))
    if False:
        roi_name = os.path.join(io.fs3017_dir(), 'eeum', 'nuclei',
                                'test_b.nimroi')
        res = read_roi(roi_name)
        spline_rois = read_spline_rois(roi_name)
        poly_rois = read_spline_rois(roi_name, convert_to_polygon=True)
        spline_mask = spline_to_single_connected_mask(spline_rois[0][0])
        poly_mask = polygon_to_single_connected_mask(poly_rois[0][0])
        poly_mask_coco = polygon_to_single_connected_mask_coco(poly_rois[0][0])
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_spline_mask.tif'),
                        spline_mask.astype(np.uint8))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_mask.tif'),
                        poly_mask.astype(np.uint8))
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly_mask_coco.tif'),
                        poly_mask_coco.astype(np.uint8))
        roi_name_convert = os.path.join(io.fs3017_dir(), 'eeum', 'nuclei',
                                        'test_test.nimroi')
        write_polygon_rois(poly_rois, roi_name_convert)

        poly1 = mask_to_polygons(poly_mask, use_cv2=False)
        poly1_mask = polygon_to_single_connected_mask_exact(poly1[0])
        imageio.imwrite(os.path.join(io.fs3017_dir(), 'eeum', 'nuclei', 'test_poly1_mask.tif'),
                        poly1_mask.astype(np.uint8))
        write_polygon_rois([poly1], os.path.join(io.fs3017_dir(), 'eeum', 'nuclei',
                                                 'test_mask_poly1.nimroi'))

    if False:
        roi_name = os.path.join(io.fs3017_dir(), 'eeum', 'Tracing',
                                '20190319_JK721_ZsGreen_PV', 'F1', 'JK721-1_stitched',
                                'JK721-1__20x_2zoom_STN_PV_ZsGreen_2_L9_Sum.lsm_ch2_bbox.nimroi')
        res = read_roi(roi_name)
        rois = read_rect_rois(roi_name)

        label_image = os.path.join(io.fs3017_dir(), 'eeum', 'Tracing',
                                   '20190319_JK721_ZsGreen_PV', 'F1', 'JK721-1_stitched',
                                   'JK721-1__20x_2zoom_STN_PV_ZsGreen_2_L9_Sum.lsm_ch1_label.nim')
        roi_name = os.path.join(io.fs3017_dir(), 'eeum', 'Tracing',
                                '20190319_JK721_ZsGreen_PV', 'F1', 'JK721-1_stitched',
                                'JK721-1__20x_2zoom_STN_PV_ZsGreen_2_L9_Sum.lsm_ch1_label_spline.nimroi')
        label_image_to_roi(label_image, roi_name, sample_to_spline=True)
