import numpy as np

class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    d1_sq = np.sum(des1**2, axis=1).reshape(-1, 1)
    d2_sq = np.sum(des2**2, axis=1).reshape(1, -1)
    dot = np.dot(des1, des2.T)
    distances = np.sqrt(np.maximum(d1_sq + d2_sq - 2 * dot, 0))
    best_for_1 = np.argmin(distances, axis=1)
    best_for_2 = np.argmin(distances, axis=0)
    matches = []
    for i, j in enumerate(best_for_1):
        if best_for_2[j] == i:
            matches.append(DummyMatch(i, j, distances[i, j]))
    matches = sorted(matches, key=lambda m: m.distance)
    return matches
