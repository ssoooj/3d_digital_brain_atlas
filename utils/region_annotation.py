import h5py
import logging
import copy

from zimg import *
from utils import io
from utils import nim_roi

logger = logging.getLogger(__name__)


def read_region_annotation(ra_name: str):
    with h5py.File(ra_name, mode='r', libver='latest') as region_annotation:
        assert 'RegionAnnotation' in region_annotation, 'can not locate RegionAnnotation in file'
        res = {}
        ra_group = region_annotation['RegionAnnotation']
        ra_attrs = ra_group.attrs
        res['Version'] = int(ra_attrs['Version'])
        res['VoxelSizeXInUM'] = ra_attrs['VoxelSizeXInUM']
        res['VoxelSizeYInUM'] = ra_attrs['VoxelSizeYInUM']
        res['VoxelSizeZInUM'] = ra_attrs['VoxelSizeZInUM']
        res['RegionNumber'] = int(ra_attrs['RegionNumber'])
        res['Regions'] = {}
        res['ID_To_ParentID'] = {}
        for region_idx in range(res['RegionNumber']):
            region_group = ra_group[f'Region{region_idx + 1}']
            region_attrs = region_group.attrs
            region_id = int(region_attrs['ID'])
            res['Regions'][region_id] = {
                'ParentID': int(region_attrs['ParentID']),
                'Color': [int(region_attrs['Red']), int(region_attrs['Green']), int(region_attrs['Blue'])],
                'Name': region_attrs['Name'],
                'Abbreviation': region_attrs['Abbreviation'],
                'ROI': None
            }
            res['ID_To_ParentID'][region_id] = res['Regions'][region_id]['ParentID']
            if 'ROI' in region_group:
                res['Regions'][region_id]['ROI'] = nim_roi.read_roi_group(region_group['ROI'])

    return res


def write_region_annotation_dict_to_group(ra_dict: dict, ra_group):
    ra_attrs = ra_group.attrs
    ra_attrs.create('Version', 100, dtype=np.int32)
    ra_attrs.create('VoxelSizeXInUM', ra_dict['VoxelSizeXInUM'], dtype=np.float64)
    ra_attrs.create('VoxelSizeYInUM', ra_dict['VoxelSizeYInUM'], dtype=np.float64)
    ra_attrs.create('VoxelSizeZInUM', ra_dict['VoxelSizeZInUM'], dtype=np.float64)
    region_number = 0
    for region_id, region in ra_dict['Regions'].items():
        region_group = ra_group.create_group(f'Region{region_number + 1}')
        region_attrs = region_group.attrs
        region_attrs.create('ID', region_id, dtype=np.int64)
        region_attrs.create('ParentID', region['ParentID'], dtype=np.int64)
        region_attrs.create('Red', region['Color'][0], dtype=np.int32)
        region_attrs.create('Green', region['Color'][1], dtype=np.int32)
        region_attrs.create('Blue', region['Color'][2], dtype=np.int32)
        region_attrs['Name'] = region['Name'].encode('utf-8')
        region_attrs['Abbreviation'] = region['Abbreviation'].encode('utf-8')
        if region['ROI'] is not None:
            roi_group = region_group.create_group('ROI')
            nim_roi.write_roi_dict_to_group(region['ROI'], roi_group)
        region_number += 1

    ra_attrs.create('RegionNumber', region_number, dtype=np.int32)


def write_region_annotation_dict(ra_dict: dict, ra_name: str):
    with h5py.File(ra_name, mode='w', libver='latest') as ra:
        ra_group = ra.create_group('RegionAnnotation')
        write_region_annotation_dict_to_group(ra_dict, ra_group)


def merge_region_annotation_dicts(start_slice_and_ra_dict: list):
    res = None
    region_id_to_start_slice_and_roi_dict = {}
    for start_slice, ra_dict in start_slice_and_ra_dict:
        if res is None:
            res = copy.deepcopy(ra_dict)
            for region_id, region in res['Regions'].items():
                region['ROI'] = None
                region_id_to_start_slice_and_roi_dict[region_id] = []

        for region_id, region in ra_dict['Regions'].items():
            if region_id not in res['Regions']:
                res['Regions'][region_id] = copy.deepcopy(region)
                res['Regions'][region_id]['ROI'] = None
                region_id_to_start_slice_and_roi_dict[region_id] = []

            if region['ROI'] is not None:
                region_id_to_start_slice_and_roi_dict[region_id].append((start_slice, region['ROI']))

    for region_id, region in res['Regions'].items():
        if len(region_id_to_start_slice_and_roi_dict[region_id]) > 0:
            region['ROI'] = nim_roi.merge_roi_dicts(region_id_to_start_slice_and_roi_dict[region_id])

    res['RegionNumber'] = len(res['Regions'])
    return res


# transform_fun take nx2 ndarray as input and generate nx2 ndarray or None (delete subshape)
def transform_region_annotation_dict(ra_dict: dict, transform_fun, only_apply_to_slices = None):
    res = copy.deepcopy(ra_dict)
    for _, region in res['Regions'].items():
        if region['ROI'] is not None:
            region['ROI'] = nim_roi.transform_roi_dict(region['ROI'], transform_fun, only_apply_to_slices)
    return res


# map_slice_fun take slice and map to new slice, if new slice is None or less than 0, delete slice
def map_region_annotation_dict_slices(ra_dict: dict, map_slice_fun):
    res = copy.deepcopy(ra_dict)
    for _, region in res['Regions'].items():
        if region['ROI'] is not None:
            region['ROI'] = nim_roi.map_roi_dict_slices(region['ROI'], map_slice_fun)
    return res


# return a map from region_id to
# a map of (slice) to list (instance) of (mask (np.bool 2d), x_start, y_start, shape), mask can be empty, x/y_start can be negative
def convert_region_annotation_dict_to_masks(ra_dict: dict):
    region_to_masks = {}
    for region_id, region in ra_dict['Regions'].items():
        if region['ROI']:
            region_to_masks[region_id] = nim_roi.convert_roi_dict_to_masks(region['ROI'])
    return region_to_masks


def convert_region_annotation_dict_to_binary_mask(ra_dict: dict, height: int, width: int, *, slice_idx=None):
    region_to_masks = convert_region_annotation_dict_to_masks(ra_dict=ra_dict)
    max_slice = 0
    if slice_idx is None:
        for region_id, slice_rois in region_to_masks.items():
            for slice in slice_rois:
                max_slice = max(slice, max_slice)
        if max_slice == 0:
            annotation_mask = np.zeros(shape=(height, width), dtype=np.bool)
        else:
            annotation_mask = np.zeros(shape=(max_slice + 1, height, width), dtype=np.bool)
    else:
        annotation_mask = np.zeros(shape=(height, width), dtype=np.bool)

    for slice in range(max_slice + 1):
        for region_id, slice_rois in region_to_masks.items():
            if slice_idx is not None and slice_idx != slice:
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
                    if max_slice > 0:
                        annotation_mask[slice][mask] = True
                    else:
                        annotation_mask[mask] = True

    return annotation_mask


# need to 'pip install anytree' first
def read_region_annotation_tree(ra_name: str):
    res = read_region_annotation(ra_name)
    import anytree
    id_to_node = {}
    for region_id, region in res['Regions'].items():
        id_to_node[region_id] = anytree.AnyNode(id=region_id,
                                                name=region['Name'],
                                                abbreviation=region['Abbreviation'],
                                                color=region['Color'])
    for id, pid in res['ID_To_ParentID'].items():
        if pid in id_to_node:
            id_to_node[id].parent = id_to_node[pid]

    for id, node in id_to_node.items():
        if node.is_root:
            print(anytree.RenderTree(node).by_attr('id'))
            print(anytree.RenderTree(node).by_attr('name'))
            print(anytree.RenderTree(node).by_attr('abbreviation'))
            print()
    return id_to_node


def convert_roi_dict_to_region_annotation_dict(roi_dict: dict):
    res = {}
    res['Version'] = 100
    res['VoxelSizeXInUM'] = 1.0
    res['VoxelSizeYInUM'] = 1.0
    res['VoxelSizeZInUM'] = 1.0
    res['RegionNumber'] = 1
    res['ID_To_ParentID'] = {}
    res['ID_To_ParentID'][-1] = -2
    res['Regions'] = {}
    res['Regions'][-1] = {
        'ParentID': -2,
        'Color': [255, 255, 255],
        'Name': 'Undefined',
        'Abbreviation': 'Undefined',
        'ROI': roi_dict
    }
    return res


if __name__ == "__main__":
    folder = os.path.join(io.fs3017_dir(), 'eeum', 'lemur', 'Hotsauce_334A',
                          '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
    input_ra_filename = os.path.join(folder, 'hj_aligned_annotation_merge',
                                     f'Lemur-H_SMI99_VGluT2_NeuN_all_flipped_for_tagging_sh.reganno')
    # ra_name = os.path.join(os.path.expanduser('~'), 'Downloads', 'test.reganno')
    #ra_dict = read_region_annotation(input_ra_filename)
    # masks = convert_region_annotation_dict_to_masks(ra_dict)
    #trees = read_region_annotation_tree(ra_name)

    folder = os.path.join(io.jinny_nas_dir(), 'Mouse_Lemur', 'Lemur_sh', 'jiwon-temporary')
    input_ra_filename = os.path.join(folder, f'result+ctx+fix_jiwon.reganno')
    # ra_name = os.path.join(os.path.expanduser('~'), 'Downloads', 'test.reganno')
    ra_dict = read_region_annotation(input_ra_filename)
