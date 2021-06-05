from zimg import *


def pad_img(img_data: np.ndarray, *,
            des_channel: int = 0, des_depth: int = 0, des_height: int = 0, des_width: int = 0,
            pad_before_ratio: float = 0.5
            ):
    """
    img_data: 1-4 dimensions ndarray (C x D x H) x W
    pad or remove before and after to match des dimension if des_* > 0
    return the padded img_data
    """
    num_dims = 4
    assert 4 >= img_data.ndim > 0 and img_data.size > 0, img_data.shape
    img_data_ndim = img_data.ndim
    width = img_data.shape[-1]
    height = depth = nchs = 1
    if img_data_ndim > 1:
        height = img_data.shape[-2]
    if img_data_ndim > 2:
        depth = img_data.shape[-3]
    if img_data_ndim > 3:
        nchs = img_data.shape[-4]

    do_padding = (des_channel > 0 and des_channel != nchs) or \
                 (des_depth > 0 and des_depth != depth) or \
                 (des_height > 0 and des_height != height) or \
                 (des_width > 0 and des_width != width)

    if do_padding:
        dim_ends = [None] * num_dims
        dim_starts = [0] * num_dims
        dim_pad_befores = [0] * num_dims
        dim_pad_afters = [0] * num_dims
        des_dims = (des_channel, des_depth, des_height, des_width)

        for dim in range(num_dims):
            if des_dims[dim] > 0:
                pad_dim = des_dims[dim] - img_data.shape[dim]
                if pad_dim > 0:
                    dim_pad_befores[dim] = int(pad_dim * pad_before_ratio)
                    dim_pad_afters[dim] = pad_dim - dim_pad_befores[dim]
                elif pad_dim < 0:
                    dim_starts[dim] = int(-pad_dim * pad_before_ratio)
                    dim_ends[dim] = dim_starts[dim] + des_dims[dim]

        if img_data_ndim == 1:
            img_data = img_data[np.newaxis, np.newaxis, np.newaxis, :]
        elif img_data_ndim == 2:
            img_data = img_data[np.newaxis, np.newaxis, :, :]
        elif img_data_ndim == 3:
            img_data = img_data[np.newaxis, :, :, :]
        else:
            assert img_data_ndim == 4, img_data.shape
        res = np.pad(img_data,
                     ((dim_pad_befores[0], dim_pad_afters[0]),
                      (dim_pad_befores[1], dim_pad_afters[1]),
                      (dim_pad_befores[2], dim_pad_afters[2]),
                      (dim_pad_befores[3], dim_pad_afters[3])))[
              dim_starts[0]:dim_ends[0],
              dim_starts[1]:dim_ends[1],
              dim_starts[2]:dim_ends[2],
              dim_starts[3]:dim_ends[3],
              ]
        if img_data_ndim < 4 and res.shape[0] == 1:
            res = res[0]
            if img_data_ndim < 3 and res.shape[0] == 1:
                res = res[0]
                if img_data_ndim < 2 and res.shape[0] == 1:
                    res = res[0]
        return res
    else:
        return img_data


def write_img(filename: str, img_data: np.ndarray, *,
              des_channel: int = 0, des_depth: int = 0, des_height: int = 0, des_width: int = 0,
              pad_before_ratio: float = 0.5,
              voxel_size_unit: VoxelSizeUnit = VoxelSizeUnit.none,
              voxel_size_x: float = 1., voxel_size_y: float = 1.,voxel_size_z: float = 1.):
    """
    img_data: 1-4 dimensions ndarray (C x D x H) x W
    pad or remove before and after to match des dimension if des_* > 0
    """
    assert img_data.ndim > 0 and img_data.size > 0, img_data.shape
    if img_data.ndim == 1:
        img_data = img_data[np.newaxis, np.newaxis, np.newaxis, :]
    elif img_data.ndim == 2:
        img_data = img_data[np.newaxis, np.newaxis, :, :]
    elif img_data.ndim == 3:
        img_data = img_data[np.newaxis, :, :, :]
    else:
        assert img_data.ndim == 4, img_data.shape

    pad_data = np.ascontiguousarray(
        pad_img(img_data, des_channel=des_channel, des_depth=des_depth, des_height=des_height,
                des_width=des_width, pad_before_ratio=pad_before_ratio))

    info = ZImgInfo()
    info.voxelSizeUnit = voxel_size_unit
    info.voxelSizeX = voxel_size_x
    info.voxelSizeY = voxel_size_y
    info.voxelSizeZ = voxel_size_z
    res_img = ZImg(pad_data, info)
    res_img.save(filename)
    # else:
    #     assert pad_data.shape[0] in [1, 3] and pad_data.shape[1] == 1, pad_data.shape
    #     res_data = np.moveaxis(pad_data[::-1, 0, :, :], 0, -1)
    #     cv2.imwrite(filename, np.ascontiguousarray(res_data))


def read_img_as_RGB(filename: str):
    img = ZImg(filename)
    nchs, depth, height, width = img.data[0].shape
    if img.data[0].dtype == np.uint8:
        img_min = 0.
        img_max = 255.
    else:
        img_min = img.data[0].min() * 1.0
        img_max = img.data[0].max() * 1.0
    res = np.zeros(shape=(3, depth, height, width), dtype=np.uint8)
    if img_min == img_max:
        return res
    for ch in range(nchs):
        image = (img.data[0][ch, :, :, :] - img_min) / (img_max - img_min)
        res[0, :, :, :] = np.maximum(res[0, :, :, :], (img.info.channelColors[ch].r * image).astype(np.uint8))
        res[1, :, :, :] = np.maximum(res[1, :, :, :], (img.info.channelColors[ch].g * image).astype(np.uint8))
        res[2, :, :, :] = np.maximum(res[2, :, :, :], (img.info.channelColors[ch].b * image).astype(np.uint8))
    return res


def read_img_as_HWC(filename: str, channels = None):
    img = ZImg(filename)
    nchs, depth, height, width = img.data[0].shape
    assert depth == 1 and img.data[0].dtype == np.uint8, img
    if channels is None:
        return np.moveaxis(img.data[0][:, 0, :, :], 0, -1).copy()
    else:
        return np.moveaxis(img.data[0][channels, 0, :, :], 0, -1).copy()


def normalize_img_data(img_data: np.ndarray, min_max_percentile=(2, 98)):
    nchs, depth, height, width = img_data.shape
    ch_min_values = []
    ch_max_values = []
    for ch, chdata in enumerate(img_data):
        if min_max_percentile[0] <= 0 and min_max_percentile[1] >= 100:
            minmax_value = (chdata.min(), chdata.max())
        else:
            minmax_value = np.percentile(chdata[chdata > 0], [min_max_percentile[0], min_max_percentile[1]])
        ch_min_values.append(minmax_value[0] * 1.0)
        ch_max_values.append(minmax_value[1] * 1.0)

    res_img_data = np.zeros(shape=(nchs, depth, height, width), dtype=np.uint8)
    for img_slice in range(depth):
        for ch in range(nchs):
            ch_data = (img_data[ch, img_slice, :, :].astype(np.float64) - ch_min_values[ch]) / \
                      (ch_max_values[ch] - ch_min_values[ch])
            res_img_data[ch, img_slice, :, :] = (np.clip(ch_data, 0.0, 1.0) * 255).astype(np.uint8)
    return res_img_data, (ch_min_values, ch_max_values)


def imresize(img_data: np.ndarray, *,
             des_depth: int = 0, des_height: int = 0, des_width: int = 0,
             interpolant: Interpolant = Interpolant.Cubic, antialiasing: bool = True,
             antialiasingForNearest: bool = False
             ):
    """
    similar to matlab imresize
    img_data: ndarray (N x ... x C x D x H) x W
    keep original size for dimension if des_* <= 0
    return the resized img_data
    """
    assert img_data.ndim > 0 and img_data.size > 0, img_data.shape
    width = img_data.shape[-1]
    height = depth = 1
    if img_data.ndim > 1:
        height = img_data.shape[-2]
    if img_data.ndim > 2:
        depth = img_data.shape[-3]

    do_resizing = (des_depth > 0 and des_depth != depth) or \
                  (des_height > 0 and des_height != height) or \
                  (des_width > 0 and des_width != width)

    if do_resizing:
        if img_data.ndim == 1:
            img = ZImg(img_data[np.newaxis, np.newaxis, np.newaxis, :])
            img.resize(desWidth=des_width if des_width > 0 else width,
                       desHeight=des_height if des_height > 0 else height,
                       desDepth=des_depth if des_depth > 0 else depth,
                       interpolant=interpolant, antialiasing=antialiasing,
                       antialiasingForNearest=antialiasingForNearest)
            if img.info.depth > 1:
                return img.data[0][0, :, :, :].copy()
            elif img.info.height > 1:
                return img.data[0][0, 0, :, :].copy()
            else:
                return img.data[0][0, 0, 0, :].copy()
        elif img_data.ndim == 2:
            img = ZImg(img_data[np.newaxis, np.newaxis, :, :])
            img.resize(desWidth=des_width if des_width > 0 else width,
                       desHeight=des_height if des_height > 0 else height,
                       desDepth=des_depth if des_depth > 0 else depth,
                       interpolant=interpolant, antialiasing=antialiasing,
                       antialiasingForNearest=antialiasingForNearest)
            if img.info.depth > 1:
                return img.data[0][0, :, :, :].copy()
            else:
                return img.data[0][0, 0, :, :].copy()
        elif img_data.ndim == 3:
            img = ZImg(img_data[np.newaxis, :, :, :])
            img.resize(desWidth=des_width if des_width > 0 else width,
                       desHeight=des_height if des_height > 0 else height,
                       desDepth=des_depth if des_depth > 0 else depth,
                       interpolant=interpolant, antialiasing=antialiasing,
                       antialiasingForNearest=antialiasingForNearest)
            return img.data[0][0, :, :, :].copy()
        elif img_data.ndim >= 4:
            img = ZImg(img_data)
            img.resize(desWidth=des_width if des_width > 0 else width,
                       desHeight=des_height if des_height > 0 else height,
                       desDepth=des_depth if des_depth > 0 else depth,
                       interpolant=interpolant, antialiasing=antialiasing,
                       antialiasingForNearest=antialiasingForNearest)
            return img.data[0].copy()
        else:
            orig_shape = img_data.shape
            img_data = img_data.reshape((np.prod(orig_shape[0:-3]), depth, height, width))
            img = ZImg(img_data)
            img.resize(desWidth=des_width if des_width > 0 else width,
                       desHeight=des_height if des_height > 0 else height,
                       desDepth=des_depth if des_depth > 0 else depth,
                       interpolant=interpolant, antialiasing=antialiasing,
                       antialiasingForNearest=antialiasingForNearest)
            return img.data[0].reshape(orig_shape[0:-3] + (img.info.depth, img.info.height, img.info.width)).copy()
    else:
        return img_data