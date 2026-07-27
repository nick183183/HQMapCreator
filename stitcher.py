import cv2
import numpy as np
import json
from pathlib import Path


class StitchQualityError(Exception):
    """Исключение при плохом качестве склейки."""
    pass


class ImageStitcher:
    """Симметричный stitcher с оптимизацией через предсказание области."""

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
        self.search_padding_ratio = config.get("search_padding_ratio", 0.5)
        self.transform_type = config.get("transform_type", "affine")
        method = config.get("method", "sift")

        if method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=self.nfeatures)
            self.matcher = cv2.BFMatcher()
        elif method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=self.nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        # Кэш keypoints для img1 (текущего накопленного изображения)
        self.cached_kp = None
        self.cached_desc = None
        self.cached_shape = None

    def find_keypoints(self, image, use_cache=False):
        """Находит keypoints с опциональным кэшированием."""
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

    def find_keypoints_in_region(self, image, region):
        """
        Находит keypoints в указанной области (x, y, w, h).
        Возвращает keypoints в координатах ПОЛНОГО изображения.
        """
        h, w = image.shape[:2]
        rx, ry, rw, rh = region

        rx = max(0, int(rx))
        ry = max(0, int(ry))
        rw = min(int(rw), w - rx)
        rh = min(int(rh), h - ry)

        if rw <= 10 or rh <= 10:
            return None, None

        roi = image[ry:ry + rh, rx:rx + rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kp, desc = self.detector.detectAndCompute(gray, None)

        if kp is None or len(kp) == 0 or desc is None:
            return None, None

        # Сдвигаем координаты keypoints в систему полного изображения
        shifted_kp = []
        for k in kp:
            shifted_kp.append(cv2.KeyPoint(
                x=k.pt[0] + rx, y=k.pt[1] + ry,
                size=k.size, angle=k.angle,
                response=k.response, octave=k.octave,
                class_id=k.class_id
            ))

        return shifted_kp, desc

    def match_keypoints(self, desc1, desc2):
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)
        return good

    def compute_transform(self, kp1, kp2, good_matches, min_matches=4):
        """
        Вычисляет преобразование между точками.
        В зависимости от transform_type:
        - "affine" → аффинное (6 параметров, БЕЗ перспективы)
        - "homography" → гомография (8 параметров, С перспективой)
        """
        if len(good_matches) < min_matches:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        if self.transform_type == "affine":
            # Аффинное преобразование (6 параметров) — без перспективы
            M, mask = cv2.estimateAffine2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )
            if M is None:
                return None, None
            # Превращаем 2x3 аффинную матрицу в 3x3 для совместимости
            H = np.vstack([M, [0, 0, 1]])
        else:
            # Гомография (8 параметров) — с перспективой
            H, mask = cv2.findHomography(
                pts1, pts2,
                cv2.RANSAC,
                self.ransac_threshold
            )

        return H, mask

    def _try_match(self, kp1, desc1, kp2, desc2):
        """
        Пытается сматчить keypoints и вычислить преобразование.
        Возвращает (H, mask, good_matches, inliers_ratio) или None при неудаче.
        """
        if kp1 is None or desc1 is None or kp2 is None or desc2 is None:
            return None

        good_matches = self.match_keypoints(desc1, desc2)
        if len(good_matches) < self.min_good_matches:
            return None

        H, mask = self.compute_transform(kp1, kp2, good_matches)
        if H is None:
            return None

        inliers = int(mask.sum())
        inliers_ratio = inliers / len(good_matches)
        if inliers_ratio < self.min_inliers_ratio:
            return None

        return H, mask, good_matches, inliers_ratio

    def stitch(self, img1, img2, use_cache=True, search_region=None, strict_mode=True):
        """
        Склейка с оптимизацией через предсказание области.

        search_region: (x, y, w, h) в координатах img1 — область, где ожидаем img2.
                       Если None — поиск по всему img1.
        strict_mode: если True и search_region задан, то при неудаче в области
                     НЕ проваливается в fallback, а сразу выбрасывает StitchQualityError.

        Возвращает (result, H, last_capture_rect, used_region).
        used_region: True если использовалась оптимизация, False если fallback.
        """
        # 1. Keypoints для img2 (всегда заново — это новый скрин)
        kp2, desc2 = self.find_keypoints(img2, use_cache=False)

        # 2. Пытаемся найти keypoints в предсказанной области
        used_region = False
        match_result = None

        if search_region is not None:
            kp1_region, desc1_region = self.find_keypoints_in_region(img1, search_region)
            match_result = self._try_match(kp1_region, desc1_region, kp2, desc2)
            if match_result is not None:
                used_region = True
            elif strict_mode:
                # ← СТРОГИЙ РЕЖИМ: не проваливаемся в fallback
                raise StitchQualityError("Поиск в области не удался (строгий режим)")

        # 3. Fallback: поиск по всему img1 (только если не strict_mode)
        if match_result is None:
            kp1, desc1 = self.find_keypoints(img1, use_cache=use_cache)
            match_result = self._try_match(kp1, desc1, kp2, desc2)

            if match_result is None:
                good_matches = self.match_keypoints(desc1, desc2)
                if len(good_matches) < self.min_good_matches:
                    raise StitchQualityError(
                        f"Мало совпадений: {len(good_matches)} < {self.min_good_matches}"
                    )
                raise StitchQualityError("Не удалось вычислить преобразование")

        H, mask, good_matches, inliers_ratio = match_result

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

        # 8. Топорное наложение (без градиентов)
        valid2 = warped_mask2 > 128
        valid2_3ch = np.stack([valid2] * 3, axis=-1)
        result = np.where(valid2_3ch, warped2, warped1)

        # 9. Вычисляем last_capture_rect ДО автокропа
        img2_corners_on_canvas = cv2.perspectiveTransform(corners2, T)
        x2_min = int(np.floor(img2_corners_on_canvas[:, 0, 0].min()))
        y2_min = int(np.floor(img2_corners_on_canvas[:, 0, 1].min()))
        x2_max = int(np.ceil(img2_corners_on_canvas[:, 0, 0].max()))
        y2_max = int(np.ceil(img2_corners_on_canvas[:, 0, 1].max()))
        last_rect_before_crop = (x2_min, y2_min, x2_max - x2_min, y2_max - y2_min)

        # 10. Автокроп
        result, crop_offset = self._autocrop_with_offset(result)

        last_capture_rect = (
            last_rect_before_crop[0] - crop_offset[0],
            last_rect_before_crop[1] - crop_offset[1],
            last_rect_before_crop[2],
            last_rect_before_crop[3]
        )

        return result, H, last_capture_rect, used_region

    @staticmethod
    def _autocrop_with_offset(image):
        """Убирает черные поля и возвращает (cropped_image, (x_offset, y_offset))."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image, (0, 0)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y + h, x:x + w], (x, y)

    @staticmethod
    def _autocrop(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y + h, x:x + w]


def main():
    path1 = "map1.png"
    path2 = "map2.png"

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print("Ошибка загрузки изображений!")
        return

    print(f"img1: {img1.shape}, img2: {img2.shape}")

    stitcher = ImageStitcher()
    try:
        result, H, last_rect, used_region = stitcher.stitch(img1, img2)
        print(f"Результат: {result.shape}")
        print(f"Последний кадр: {last_rect}")
        print(f"Режим: {'область' if used_region else 'fallback'}")
        cv2.imwrite("stitched_result.png", result)
        print("Сохранено: stitched_result.png")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()