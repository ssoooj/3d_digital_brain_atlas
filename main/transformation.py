import numpy as np
import cv2
import matplotlib.pyplot as plt


def points_generator(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords


def get_matrix(r, tx, ty, flip: bool):
    rigid_M = np.array([[np.cos(np.radians(r)), - np.sin(np.radians(r)), tx],
                        [np.sin(np.radians(r)), np.cos(np.radians(r)), ty], [0, 0, 1]])
    if flip == True:
        flip_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])  # vertical flip
        M = rigid_M @ flip_M
    else:
        M = rigid_M
    return M


def get_rotation(r, flip: bool):
    rotation_M = np.array([[np.cos(np.radians(r)), np.sin(np.radians(r)), 0],
                           [-np.sin(np.radians(r)), np.cos(np.radians(r)), 0], [0, 0, 1]])
    if flip == True:
        flip_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = rotation_M @ flip_M
    else:
        M = rotation_M
    return M


def get_translation(tx, ty, flip: bool):
    trans_M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    if flip == True:
        flip_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = trans_M @ flip_M
    else:
        M = trans_M
    return trans_M


def get_scale(s, flip: bool):
    scale_M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    if flip == True:
        flip_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = scale_M @ flip_M
    else:
        M = scale_M
    return scale_M


def two_by_three(x: float, y: float, r, s, flip: bool):
    a = s * np.cos(np.radians(r))
    b = s * np.sin(np.radians(r))
    c = np.array([x, y])
    two_by_three_M = np.array([[a, b, (1 - a) * c[0] - b * c[1]],
                               [-b, a, b * c[0] + (1 - a) * c[1]]])
    if flip == True:
        flip_M = np.array([[1, 0, 0], [0, -1, 1000], [0, 0, 1]])
        M = two_by_three_M @ flip_M
    else:
        M = two_by_three_M
    return M


print("maaaaaatrix : ", two_by_three(500.0, 500.0, 90, 1, flip=True), "\n")

img = plt.imread('Banana.png')
rows, cols = img.shape[:2]
sh_M = two_by_three(500.0, 500.0, 90, 1, flip=True)
dst2 = cv2.warpAffine(img, sh_M, (cols, rows))

cv2.imwrite('Rotated sh.png', dst2)
cv2.imshow('Original_sh', img)
cv2.imshow('Rotation_and_reflection_sh', dst2)
cv2.waitKey()

# cv2.destroyAllWindows()

# Interpolation

def bilinear_interpolation(x, y, dx, dy):  # not readyyyy
    p1 = (np.array([x, y]))
    p2 = (np.array([x + 1, y]))
    p3 = (np.array([x, y + 1]))
    p4 = (np.array([x + 1, y + 1]))
    P = (1 - dx) * (1 - dy) * p1 + dx * (1 - dy) * p2 + (1 - dx) * dy * p3 + dx * dy * p4
    return P


print("interpolation : ", bilinear_interpolation(2, 4, 2, 4))


def get_cv_affine(t, r, s):
    sin_theta = np.sin(r)
    cos_theta = np.cos(r)

    element11 = s * cos_theta
    element21 = -s * sin_theta

    element12 = s * sin_theta
    element22 = s * cos_theta

    element13 = t[0] * (1 - s * cos_theta) - s * sin_theta * t[1]
    element23 = t[1] * (1 - s * cos_theta) + s * sin_theta * t[0]

    return np.array([[element11, element12, element13],
                     [element21, element22, element23]])


if __name__ == "__main__":

    # Test with cv2

    img = plt.imread('Banana.png')
    print("img shape : ", img.shape, "\n")
    print("img type : ", type(img), "\n")
    rows, cols = img.shape[:2]
    print("center of the image : ", cols / 2, rows / 2, "\n")

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    flip_M = np.array([[1, 0, 0], [0, -1, 1000], [0, 0, 1]])
    combined_M = M @ flip_M
    print("cv2 matrix : ", M, "\n")
    print("cv2 combined : ", combined_M, "\n")
    dst = cv2.warpAffine(img, combined_M, (cols, rows))
    # print("dst : ", dst)

    cv2.imwrite('Rotated Banana.png', dst)
    cv2.imshow('Original', img)
    cv2.imshow('Rotation and reflection', dst)
    cv2.waitKey()
    # cv2.destroyAllWindows()

    # horizontally_flipped = np.fliplr(img)
    # shiftimg = scipy.ndimage.shift(horizontally_flipped, (10, 10, 0))
    # center = scipy.ndimage.measurements.center_of_mass(img)
    # print("center : ", center)

    """
    # slice 182

    iso_coord = np.array([8368, 2432]).astype(np.float32)
    hpf_coord = np.array([24848, 14160]).astype(np.float32)
    th_coord = np.array([13840, 8080]).astype(np.float32)

    iso_change = np.array([7877.81, 2373.61])
    hpf_change = np.array([25578.4, 14756.7])
    th_change = np.array([14076.7, 8211.29])

    get_coord_iso = iso_coord - iso_change
    print("iso_coordinate : ", get_coord_iso, type(get_coord_iso), "\n")

    get_coord_hpf = hpf_coord - hpf_change
    print("hpf_coordinate : ", get_coord_hpf, type(get_coord_hpf), "\n")

    get_coord_th = th_coord - th_change
    print("th_coordinate : ", get_coord_th, type(get_coord_th), "\n")

    flip_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print("flip matrix : ", flip_M, "\n")

    translate_M = np.array([[1, 0, -380], [0, 1, -450], [0, 0, 1]])
    print("translate matrix : " ,translate_M, "\n")

    scale_M = np.array([[1.05, 0, 0], [0, 1.06, 0], [0, 0, 1]])
    print("scale matrix : ", scale_M, "\n")

    rotation_M = np.array([[math.cos(490.19), math.sin(58.39), 0], [math.sin(-490.19), math.cos(58.39), 0], [0, 0, 1]])
    print("rotation matrix ", rotation_M, "\n")

    testttt = rotation_M * translate_M
    """