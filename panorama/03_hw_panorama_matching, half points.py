import json
import os
import cv2
import glob
import numpy as np
from skimage import io

# ============================================
# ЗАДАНИЕ 1: Manual Panorama
# ============================================
# Считываем изображения из базовой директории
base_path = "D:/Shit Delete/"
image_dict = {}
for filename in sorted(glob.glob(os.path.join(base_path, "*.[Jj][Pp][Ee][Gg]"))):
    name = os.path.basename(filename)
    prefix = name.split("_")[0]
    img = io.imread(filename)
    if prefix in image_dict:
        image_dict[prefix].append(img)
    else:
        image_dict[prefix] = [img]

# Выбираем пару изображений по ключу (измените, если у вас другой ключ)
image1, image2 = image_dict["example1"]

X_SHIFT = 100
Y_SHIFT = 100
# Подберите вручную значения смещения
tx = 0
ty = 0
size_manual = (image1.shape[0] + image2.shape[0], image1.shape[1] + image2.shape[1], 3)
image_trans = np.uint8(np.zeros(size_manual))
image_trans[Y_SHIFT:Y_SHIFT + image1.shape[0], X_SHIFT:X_SHIFT + image1.shape[1], :] = image1
image_trans[Y_SHIFT + ty:Y_SHIFT + ty + image2.shape[0], X_SHIFT + tx:X_SHIFT + tx + image2.shape[1], :] = image2

# Сохраняем значения смещения в manual_panorama.json (в текущей директории)
with open("manual_panorama.json", "w") as f:
    json.dump([tx, ty], f)

# ============================================
# ЗАДАНИЕ 2: h_submission_dict
# ============================================
def extract_key_points(img1, img2):
    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    return kpts1, desc1, kpts2, desc2

def match_key_points_cv(des1, des2):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def dlt_homography_normalized(pts1, pts2):
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    mean1 = np.mean(pts1, axis=0)
    mean2 = np.mean(pts2, axis=0)
    dists1 = np.linalg.norm(pts1 - mean1, axis=1)
    dists2 = np.linalg.norm(pts2 - mean2, axis=1)
    scale1 = np.sqrt(2) / np.mean(dists1)
    scale2 = np.sqrt(2) / np.mean(dists2)
    T1 = np.array([[scale1, 0, -scale1 * mean1[0]],
                   [0, scale1, -scale1 * mean1[1]],
                   [0, 0, 1]])
    T2 = np.array([[scale2, 0, -scale2 * mean2[0]],
                   [0, scale2, -scale2 * mean2[1]],
                   [0, 0, 1]])
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts1_norm = (T1 @ pts1_h.T).T
    pts2_norm = (T2 @ pts2_h.T).T
    A = []
    for (x, y), (xp, yp) in zip(pts1_norm[:, :2], pts2_norm[:, :2]):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]
    return H

def findHomography_dlt_numpy(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return dlt_homography_normalized(src_pts, dst_pts), None

def panorama(img1, img2, H, dsize):
    # dsize: (width, height)
    base = cv2.warpPerspective(img1, np.eye(3), dsize)
    warp_img2 = cv2.warpPerspective(img2, H, dsize, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
    pano = base.copy()
    # Вычисляем маску как (height, width)
    mask = (warp_img2.sum(axis=2) > 0)
    pano[mask] = warp_img2[mask]
    return pano

def panorama_pipeline(img1, img2, dsize):
    kp1, desc1, kp2, desc2 = extract_key_points(img1, img2)
    matches = match_key_points_cv(desc1, desc2)
    best_matches = matches[:50]
    H, _ = findHomography_dlt_numpy(best_matches, kp1, kp2)
    _ = panorama(img1, img2, H, dsize)  # построение панорамы для тестирования (опционально)
    return H

# Определите размер итогового изображения как кортеж (width, height)
dsize = (1920, 1280)
h_dict = {}
for key, imgs in image_dict.items():
    if len(imgs) < 2:
        continue
    img1, img2 = imgs[:2]
    H = panorama_pipeline(img1, img2, dsize)
    h_dict[key] = H.tolist()

with open("h_submission_dict.json", "w") as f:
    json.dump(h_dict, f)
