import cv2
import numpy as np
import json
from pathlib import Path


class StitchQualityError(Exception):
    """Исключение при плохом качестве склейки."""
    pass


class ImageStitcher:
    """Симметричный stitcher с кэшированием и проверкой качества."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}

        self.nfeatures = config.get("nfeatures", 3000)
        self.ratio_threshold = config.get("ratio_threshold", 0.75)
        self.ransac_threshold = config.get("ransac_threshold", 5.0)
        self.min_inliers_ratio = config.get("min_inliers_ratio", 0.3)
        self.min_good_matches = config.get("min_good_matches", 10)
        method = config.get("method", "sift")

        if method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=self.nfeatures)
            self.matcher = cv2.BFMatcher()
        elif method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=self.nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.cached_kp = None
        self.cached_desc = None
        self.cached_shape = None

    def find_keypoints(self, image, use_cache=False):
        current_shape = image.shape[:2]
        if use_cache and self.cached_shape == current_shape:
            return self.cached_kp, self.cached_desc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.detector.detectAndCompute(gray, None)

        if use_cache:
            self.cached_kp = kp
            self.cached_desc = desc
            self.cached_shape = current_shape

        return kp, desc

    def match_keypoints(self, desc1, desc2):
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)
        return good

    def compute_homography(self, kp1, kp2, good_matches, min_matches=4):
        if len(good_matches) < min_matches:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
        return H, mask

    def stitch(self, img1, img2, use_cache=True):
        """Склейка с проверкой качества."""
        # 1. Ключевые точки
        kp1, desc1 = self.find_keypoints(img1, use_cache=use_cache)
        kp2, desc2 = self.find_keypoints(img2, use_cache=False)

        # 2. Совпадения
        good_matches = self.match_keypoints(desc1, desc2)

        # ПРОВЕРКА КАЧЕСТВА #1: минимум совпадений
        if len(good_matches) < self.min_good_matches:
            raise StitchQualityError(
                f"Мало совпадений: {len(good_matches)} < {self.min_good_matches}"
            )

        # 3. Гомография
        H, mask = self.compute_homography(kp1, kp2, good_matches)
        if H is None:
            raise StitchQualityError("Не удалось вычислить гомографию")

        inliers = int(mask.sum())
        inliers_ratio = inliers / len(good_matches)

        # ПРОВЕРКА КАЧЕСТВА #2: процент inliers
        if inliers_ratio < self.min_inliers_ratio:
            raise StitchQualityError(
                f"Низкое качество: inliers {inliers_ratio:.1%} < {self.min_inliers_ratio:.1%}"
            )

        # 4. Размеры и границы
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        warped_corners1 = cv2.perspectiveTransform(corners1, H)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

        all_corners = np.concatenate([warped_corners1, corners2], axis=0)
        x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int) - 1
        x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int) + 1

        out_w = x_max - x_min
        out_h = y_max - y_min

        # 5. Трансляция
        T = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # 6. Варпим оба изображения
        warped1 = cv2.warpPerspective(img1, T @ H, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        warped2 = cv2.warpPerspective(img2, T, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        # 7. Маски
        mask1_orig = np.ones((h1, w1), dtype=np.uint8) * 255
        mask2_orig = np.ones((h2, w2), dtype=np.uint8) * 255

        warped_mask1 = cv2.warpPerspective(mask1_orig, T @ H, (out_w, out_h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
        warped_mask2 = cv2.warpPerspective(mask2_orig, T, (out_w, out_h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)

        # 8. Топорное наложение
        valid2 = warped_mask2 > 128
        valid2_3ch = np.stack([valid2] * 3, axis=-1)
        result = np.where(valid2_3ch, warped2, warped1)

        # 9. Автокроп
        result = self._autocrop(result)

        return result, H

    @staticmethod
    def _autocrop(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y + h, x:x + w]