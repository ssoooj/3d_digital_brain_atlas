import os
import sys
import numpy as np
import scipy
import scipy.fft
import logging

from zimg import *
from utils import io
from utils import img_util
from utils import region_annotation

logger = logging.getLogger(__name__)


def BaSiC_parameters():
    return {
        'lambdaa': None,  # default value estimated from input images directly, high values (eg. 9.5) increase the
        # spatial regularization strength, yielding a more smooth flatfield
        'estimation_mode': 'l0',
        'max_iterations': 500,
        'optimization_tol': 1e-6,
        'darkfield': False,  # whether you would like to estimate darkfield, keep 'false' if the input images are
        # brightfield images or bright fluoresence images, set 'true' only if only if the input images are fluorescence
        # images are dark and have a strong darkfield contribution.
        'lambda_darkfield': None,  # default value estimated from input images directly, , high values (eg. 9.5)
        # increase the spatial regularization strength, yielding a more smooth darkfield
        'working_size': 128,
        'max_reweightiterations': 10,
        'eplson': .1,  # reweighting parameter
        'varying_coeff': True,
        'reweight_tol': 1e-3,  # reweighting tolerance
    }


def inexact_alm_rspca_l1(D, *, lambdaa=None, lambda_darkfield=None, tol=1e-6,
                         maxIter=500, weight=1., estimatedarkfield=False, darkfieldlimit=1e7):
    '''
     l1 minimizatioin, background has a ratio, rank 1 but not the
     same
     This matlab code implements the inexact augmented Lagrange multiplier
     method for Sparse low rank matrix recovery

     modified from Robust PCA
     reference:
     Peng et al. "A BaSiC tool for background and shading correction
     of optical microscopy images" Nature Communications, 14836(2017)
     Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal Component
     Analysis?" J. ACM (58) 2011

     D - n x m x m  matrix of observations/data (required input)

     while ~converged
       minimize (inexactly, update A and E only once)
       L(W, E,Y,u) = |E|_1+lambda * |W|_1 + <Y2,D-repmat(QWQ^T)-E> + +mu/2 * |D-repmat(QWQ^T)-E|_F^2;
       Y1 = Y1 + \mu * (D - repmat(QWQ^T) - E);
       \mu = \rho * \mu;
     end
    :param D: m x m x n  matrix of observations/data (required input)
    :return: A1_hat, E1_hat, A_offset
    '''
    p, q, n = D.shape
    m = p * q
    D = np.reshape(D, (m, n))
    if isinstance(weight, np.ndarray):
        weight = np.reshape(weight, D.shape)

    temp = np.linalg.svd(D, full_matrices=False, compute_uv=False, hermitian=False)
    norm_two = temp[0]
    Y1 = Y2 = 0.
    ent1 = 1.
    ent2 = 10.

    A1_hat = np.zeros(D.shape)
    E1_hat = np.zeros(D.shape)
    W_hat = scipy.fft.dctn(np.mean(np.reshape(A1_hat, (p, q, n)), axis=-1), norm='ortho')
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(D, 'fro')
    A1_coeff = np.ones(shape=(1, n))
    A_offset = np.zeros(shape=(m, 1))
    B1_uplimit = D.min()
    B1_offset = 0.
    A_uplimit = np.min(D, axis=-1)
    A_inmask = np.zeros(shape=(p, q))
    A_inmask[int(np.round(p / 6.)):int(np.round(p * 5. / 6)), int(np.round(q / 6.)):int(np.round(q * 5. / 6))] = 1
    # main iterations
    iter = 0
    total_svd = 0
    converged = False
    W_idct_hat = None
    while not converged:
        iter += 1
        W_idct_hat = scipy.fft.idctn(W_hat, norm='ortho')
        A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
        temp_W = (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1

        temp_W = np.mean(np.reshape(temp_W, (p, q, n)), axis=-1)
        W_hat += scipy.fft.dctn(temp_W, norm='ortho')
        W_hat = np.fmax(W_hat - lambdaa / (ent1 * mu), 0.) + np.fmin(W_hat + lambdaa / (ent1 * mu), 0.)
        W_idct_hat = scipy.fft.idctn(W_hat, norm='ortho')
        A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
        # update E1 using l0 norm
        E1_hat = E1_hat + (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        E1_hat = np.fmax(E1_hat - weight / (ent1 * mu), 0.) + np.fmin(E1_hat + weight / (ent1 * mu), 0.)
        # update A1_coeff, A2_coeff and A_offset
        R1 = D - E1_hat
        A1_coeff = np.mean(R1, axis=0) / np.mean(R1)
        A1_coeff[A1_coeff < 0] = 0
        if estimatedarkfield:
            validA1coeff_idx = A1_coeff < 1.
            W_idct_hat_flatten = W_idct_hat.flatten()
            B1_coeff = (np.mean(R1[W_idct_hat_flatten > np.mean(W_idct_hat_flatten) - 1e-6][:, validA1coeff_idx], axis=0) -
                        np.mean(R1[W_idct_hat_flatten < np.mean(W_idct_hat_flatten) + 1e-6][:, validA1coeff_idx], axis=0)) / \
                       np.mean(R1)
            k = validA1coeff_idx.sum()
            temp1 = np.sum(np.square(A1_coeff[validA1coeff_idx]), axis=0)
            temp2 = np.sum(A1_coeff[validA1coeff_idx], axis=0)
            temp3 = np.sum(B1_coeff, axis=0)
            temp4 = np.sum(A1_coeff[validA1coeff_idx] * B1_coeff, axis=0)
            temp5 = temp2 * temp3 - k * temp4
            if temp5 == 0:
                B1_offset = 0.
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit
            B1_offset = np.fmax(B1_offset, 0.0)
            B1_offset = np.fmin(B1_offset, B1_uplimit / np.mean(W_idct_hat_flatten))
            B_offset = B1_offset * np.mean(W_idct_hat_flatten) - B1_offset * W_idct_hat_flatten
            A1_offset = np.mean(R1[:, validA1coeff_idx], axis=1) - np.mean(A1_coeff[validA1coeff_idx]) * W_idct_hat_flatten
            A1_offset -= np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset
            # smooth A_offset
            W_offset = scipy.fft.dctn(np.reshape(A_offset, (p, q)), norm='ortho')
            W_offset = np.fmax(W_offset - lambda_darkfield / (ent2 * mu), 0) + \
                       np.fmin(W_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = scipy.fft.idctn(W_offset, norm='ortho')
            A_offset = A_offset.flatten()
            # encourage sparse A_offset
            A_offset = np.fmax(A_offset - lambda_darkfield / (ent2 * mu), 0) + \
                       np.fmin(A_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = A_offset + B_offset
        Z1 = D - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = min(mu * rho, mu_bar)

        # stop criterion
        stopCriterion = np.linalg.norm(Z1, 'fro')/ d_norm
        if stopCriterion < tol:
            converged = True

        if np.mod(total_svd, 10) == 0:
            logger.info(f'Iteration {iter} |W|_0 {(np.abs(W_hat) > 0).sum()} |E1|_0 {(np.abs(E1_hat) > 0).sum()}'
                        f' stopCriterion {stopCriterion} B1_offset {B1_offset}')

        if not converged and iter >= maxIter:
            logger.info('Maximum iterations reached')
            converged = True

    A_offset += B1_offset * W_idct_hat.flatten()
    return A1_hat, E1_hat, A_offset


def BaSiC(img_tiles: np.ndarray, *, lambdaa=None, estimation_mode='l0', max_iterations=500, optimization_tol=1e-6,
          estimate_darkfield=False, lambda_darkfield=None, working_size=128, max_reweightiterations=10, eplson=.1,
          varying_coeff=True, reweight_tol=1e-3):
    '''
    Estimation of flatfield for optical microscopy. Apply to a collection of
    monochromatic images. Multi-channel images should be separated, and each
    channel corrected separately.

    :param IF: nimg x nrows x ncols ndarray
    :param lambdaa: default value estimated from input images directly, high values (eg. 9.5) increase the spatial regularization strength, yielding a more smooth flatfield
    :param estimation_mode: default l0
    :param max_iterations: default 500
    :param optimization_tol: default 1e-6
    :param estimate_darkfield: whether you would like to estimate darkfield, keep 'false' if the input images are brightfield images or bright fluoresence images, set 'true' only if only if the input images are fluorescence images are dark and have a strong darkfield contribution.
    :param lambda_darkfield: default value estimated from input images directly, , high values (eg. 9.5) increase the spatial regularization strength, yielding a more smooth darkfield
    :param working_size: downsample to working_size x working_size before processing
    :param max_reweightiterations: default 10
    :param eplson: default .1, reweighting parameter
    :param varying_coeff:
    :param reweight_tol: default 1e-3, reweighting tolerance
    :return:
        - flatfield: estimated flatfield
        - darkfield: estimated darkfield

    reference: Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications, 14836(2017)
    '''
    tile_height, tile_width = img_tiles.shape[1:]
    D = np.moveaxis(img_util.imresize(img_tiles, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64),
                    0, -1)
    nrows, ncols, nimgs = D.shape
    meanD = np.mean(D, axis=-1)
    meanD /= np.mean(meanD)
    W_meanD = scipy.fft.dctn(meanD, norm='ortho')

    if lambdaa is None:
        lambdaa = np.abs(W_meanD).sum() / 400. * 0.5
    if lambda_darkfield is None:
        lambda_darkfield = np.abs(W_meanD).sum() / 400. * 0.2

    D.sort(axis=-1)
    XAoffset = np.zeros(shape=meanD.shape)

    weight = np.ones(shape=D.shape)
    i = 0
    flag_reweighting = True
    flatfield_last = np.ones(shape=meanD.shape)
    darkfield_last = np.random.randn(meanD.shape[0], meanD.shape[1])
    XA = None
    while flag_reweighting:
        i += 1
        logger.info(f'Reweighting Iteration {i}')
        X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, lambdaa=lambdaa, lambda_darkfield=lambda_darkfield,
                                                         tol=optimization_tol, maxIter=max_iterations,
                                                         weight=weight, estimatedarkfield=estimate_darkfield)
        XA = X_k_A.reshape((nrows, ncols, -1))
        XE = X_k_E.reshape((nrows, ncols, -1))
        XAoffset = X_k_Aoffset.reshape((nrows, ncols))
        XE_norm = XE / (np.mean(XA, axis=(0,1), keepdims=True) + 1e-6)
        weight = 1. / (np.abs(XE_norm) + eplson)
        weight = weight * weight.size / weight.sum()
        temp = np.mean(XA, axis=-1) - XAoffset
        flatfield_current = temp / np.mean(temp)
        darkfield_current = XAoffset
        mad_flatfield = np.abs(flatfield_current - flatfield_last).sum() / np.abs(flatfield_last).sum()
        temp_diff = np.abs(darkfield_current - darkfield_last).sum()
        if temp_diff < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = temp_diff / max(1e-6, np.abs(darkfield_last).sum())
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if max(mad_flatfield, mad_darkfield) <= reweight_tol or i > max_reweightiterations:
            flag_reweighting = False

    # print(XA.shape, XAoffset.shape)
    shading = np.mean(XA, axis=-1) - XAoffset
    flatfield = img_util.imresize(shading, des_height=tile_height, des_width=tile_width)
    flatfield /= np.mean(flatfield)
    XAoffset = img_util.imresize(XAoffset, des_height=tile_height, des_width=tile_width)
    darkfiled = XAoffset if estimate_darkfield else None
    return flatfield, darkfiled


def BaSiC_basefluor(img_tiles: np.ndarray, flatfield: np.ndarray, *, darkfield=None, working_size=128):
    '''
    Estimation of background fluoresence signal for time-lapse movie. Used in conjunction with BaSiC

    :param IF: nimg x nrows x ncols ndarray
    :param flatfield: estimated flatfield
    :param darkfield: to supply your darkfield, note it should be the same size as your fieldfield
    :param working_size: downsample to working_size x working_size before processing
    :return:
        - fi_base: estimated background

    reference: Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications, 14836(2017)
    '''
    tile_height, tile_width = img_tiles.shape[1:]
    D = np.moveaxis(img_util.imresize(img_tiles, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64),
                    0, -1)
    nrows, ncols, nimgs = D.shape
    D = D.reshape((nrows * ncols, -1))
    flatfield = img_util.imresize(flatfield, des_height=working_size, des_width=working_size,
                                  interpolant=Interpolant.Linear).astype(np.float64)
    if darkfield is None:
        darkfield = np.zeros(shape=flatfield.shape)
    else:
        darkfield = img_util.imresize(darkfield, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64)

    weight = np.ones(shape=D.shape)
    eplson = 0.1
    tol = 1e-6
    for reweighting_iter in range(5):
        W_idct_hat = flatfield.flatten()
        A_offset = darkfield.flatten()
        A1_coeff = np.mean(D, axis=0)
        # main iteration loop starts
        temp = np.linalg.svd(D, full_matrices=False, compute_uv=False, hermitian=False)
        norm_two = temp[0]
        mu = 12.5/norm_two  # this one can be tuned
        mu_bar = mu * 1e7
        rho = 1.5  # this one can be tuned
        d_norm = np.linalg.norm(D, 'fro')
        ent1 = 1
        iter = 0
        total_svd = 0
        converged = False
        A1_hat = np.zeros(shape=D.shape)
        E1_hat = np.zeros(shape=D.shape)
        Y1 = 0
        while not converged:
            iter += 1
            A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
            # update E1 using l0 norm
            E1_hat = E1_hat + (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
            E1_hat = np.fmax(E1_hat - weight / (ent1 * mu), 0.) + np.fmin(E1_hat + weight / (ent1 * mu), 0.)
            # update A1_coeff, A2_coeff and A_offset
            R1 = D - E1_hat
            A1_coeff = np.mean(R1, axis=0) - np.mean(A_offset)
            A1_coeff[A1_coeff < 0] = 0

            Z1 = D - A1_hat - E1_hat
            Y1 = Y1 + mu * Z1
            mu = min(mu * rho, mu_bar)

            # stop criterion
            stopCriterion = np.linalg.norm(Z1, 'fro')/ d_norm
            if stopCriterion < tol:
                converged = True

            if np.mod(total_svd, 10) == 0:
                logger.info(f'Iteration {iter} |E1|_0 {(np.abs(E1_hat) > 0).sum()}'
                            f' stopCriterion {stopCriterion}')

        # update weight
        # XE_norm = bsxfun(@ldivide, E1_hat, mean(A1_hat))
        XE_norm = np.mean(A1_hat, axis=0) / E1_hat
        weight = 1. / (np.abs(XE_norm) + eplson)
        weight = weight * weight.size / weight.sum()

    fi_base = A1_coeff
    return fi_base


def correct_shading(input_filename, scene: int, *,
                    output_filename: str=None,
                    inverse_channels: tuple=tuple(),
                    correct_background_channels: tuple=tuple(),
                    correct_background_method: str = 'annotation',  # annotation or BaSiC
                    correct_background_annotation: str = None,
                    correct_background_annotation_slice_idx = None,
                    ):
    infoList = ZImg.readImgInfos(input_filename)
    # print('image', infoList[scene])
    blockList = ZImg.getInternalSubRegions(input_filename)
    # np.set_printoptions(threshold=sys.maxsize)
    # print('czi blocks in image', blockList[scene])
    tile_width = blockList[scene][0].end.x - blockList[scene][0].start.x
    tile_height = blockList[scene][0].end.y - blockList[scene][0].start.y
    nchs = blockList[scene][0].end.c - blockList[scene][0].start.c
    ntiles = len(blockList[scene])
    # print(tile_width, tile_height, nchs)
    tile_img = ZImg(input_filename, region=blockList[scene][0], scene=scene)
    img_dtype = tile_img.data[0].dtype
    stacked_tiles = np.zeros(shape=(nchs, ntiles, tile_height, tile_width), dtype=img_dtype)
    res_mask = np.zeros(shape=(infoList[scene].depth, infoList[scene].height, infoList[scene].width),
                        dtype=np.uint8)
    for tile_idx, tile in enumerate(blockList[scene]):
        tile_img = ZImg(input_filename, region=tile, scene=scene)
        stacked_tiles[:, tile_idx, :, :] = tile_img.data[0][:, 0, :, :]
        res_mask[tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += 1
    res_mask[res_mask == 0] = 1
    res_img = np.zeros(shape=(nchs, infoList[scene].depth, infoList[scene].height, infoList[scene].width),
                       dtype=np.float64)

    for ch in inverse_channels:
        # print(ch, np.iinfo(img_dtype).max)
        stacked_tiles[ch, :, :, :] = np.iinfo(img_dtype).max - stacked_tiles[ch, :, :, :]

    if correct_background_method == 'annotation':
        assert correct_background_annotation is not None


    for ch in range(nchs):
        flatfield, darkfield = BaSiC(stacked_tiles[ch, :, :, :], estimate_darkfield=True)
        if ch in correct_background_channels:
            if correct_background_method == 'BaSiC':
                basefluor =  BaSiC_basefluor(stacked_tiles[ch, :, :, :], flatfield=flatfield, darkfield=darkfield)
                for tile_idx, tile in enumerate(blockList[scene]):
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield - basefluor[tile_idx]
                    res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
            elif correct_background_method == 'annotation':
                ra_dict = region_annotation.read_region_annotation(correct_background_annotation)
                annotation_mask = region_annotation.convert_region_annotation_dict_to_binary_mask(ra_dict,
                                                                                                  height=infoList[scene].height,
                                                                                                  width=infoList[scene].width,
                                                                                                  slice_idx=correct_background_annotation_slice_idx)
                for tile_idx, tile in enumerate(blockList[scene]):
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                    tile_region_mask = annotation_mask[tile.start.y:tile.end.y, tile.start.x:tile.end.x]
                    if tile_region_mask.sum() / tile_region_mask.size < 0.9:
                        corrected_tile[np.logical_not(tile_region_mask)] -= \
                            np.median(corrected_tile[np.logical_not(tile_region_mask)])
                    res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
            else:
                assert False, f'unknown background correction method: {correct_background_method}'
        else:
            for tile_idx, tile in enumerate(blockList[scene]):
                corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
        res_img[ch, :, :, :] /= res_mask.astype(np.float64)

    res_img = np.clip(res_img, a_min=np.iinfo(img_dtype).min, a_max=np.iinfo(img_dtype).max).astype(img_dtype)

    for ch in inverse_channels:
        res_img[ch, :, :, :][res_img[ch, :, :, :] == np.iinfo(img_dtype).min] = np.iinfo(img_dtype).max
        res_img[ch, :, :, :] = np.iinfo(img_dtype).max - res_img[ch, :, :, :]

    if output_filename is not None:
        img = ZImg(res_img, infoList[scene])
        img.save(output_filename)
    return res_img, infoList[scene]


if __name__ == "__main__":
    from utils import logger as logg
    logg.setup_logger()

    # compare with matlab
    # import glob
    # folder = os.path.join(os.path.expanduser('~/code/matlab/BaSiC-master/Demoexamples/WSI_Brain'), 'Uncorrected_tiles')
    # filelist = glob.glob(os.path.join(folder, '*.tif'))
    # img = ZImg(filelist, catDim=Dimension.Z)
    # flatfield, darkfield = BaSiC(img.data[0][0], estimate_darkfield=True)

    if False:
        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                              '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_21.czi')
        scene = 0
        output_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_21_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename)

    if False:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37.czi')
        scene = 1
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

    if False:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24.czi')
        scene = 3
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

    if True:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27.czi')
        scene = 3
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

