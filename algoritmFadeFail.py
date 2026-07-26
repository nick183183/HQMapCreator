import cv2
import numpy as np


class ImageStitcher:
    """Симметричный stitcher - оба изображения варпятся на общий холст."""

    def __init__(self, method='sift', ratio_threshold=0.75, ransac_threshold=5.0):
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold

        if method == 'sift':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher()
        elif method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=10000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

    def find_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)

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

    def stitch(self, img1, img2):
        """Симметричное склеивание - оба изображения трансформируются."""
        # 1. Ключевые точки
        kp1, desc1 = self.find_keypoints(img1)
        kp2, desc2 = self.find_keypoints(img2)
        print(f"Ключевые точки: img1={len(kp1)}, img2={len(kp2)}")

        # 2. Совпадения
        good_matches = self.match_keypoints(desc1, desc2)
        print(f"Совпадений после теста Лоу: {len(good_matches)}")
        if len(good_matches) < 4:
            raise ValueError("Слишком мало совпадений!")

        # 3. Гомография (img1 -> img2)
        H, mask = self.compute_homography(kp1, kp2, good_matches)
        if H is None:
            raise ValueError("Не удалось вычислить гомографию!")
        print(f"Inliers: {int(mask.sum())}/{len(good_matches)}")

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
        print(f"Размер холста: {out_w}x{out_h}")

        # 5. Трансляция
        T = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        # 6. Варпим ОБА изображения на общий холст
        warped1 = cv2.warpPerspective(img1, T @ H, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        T2 = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        warped2 = cv2.warpPerspective(img2, T2, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        # 7. Создаем маски для ОБА изображений
        mask1_orig = np.ones((h1, w1), dtype=np.uint8) * 255
        mask2_orig = np.ones((h2, w2), dtype=np.uint8) * 255

        warped_mask1 = cv2.warpPerspective(mask1_orig, T @ H, (out_w, out_h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
        warped_mask2 = cv2.warpPerspective(mask2_orig, T2, (out_w, out_h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)

        # 8. Симметричный блендинг
        valid1 = warped_mask1 > 128
        valid2 = warped_mask2 > 128

        overlap = valid1 & valid2
        only1 = valid1 & ~valid2
        only2 = ~valid1 & valid2

        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Зоны только img1
        only1_3ch = np.stack([only1] * 3, axis=-1)
        result = np.where(only1_3ch, warped1, result)

        # Зоны только img2
        only2_3ch = np.stack([only2] * 3, axis=-1)
        result = np.where(only2_3ch, warped2, result)

        # Зона перекрытия - плавный блендинг
        if overlap.any():
            # 🔧 ИСПРАВЛЕНИЕ: расстояние до ближайшей точки only1/only2,
            # а не до границ всей маски. Корректно работает при сложной
            # форме перекрытия (диагональ, L-форма, несколько областей).

            # Инвертируем only-маски: 0 где есть only, 255 где нет
            # Тогда distanceTransform даст расстояние до ближайшей only-точки
            inv_only1 = (~only1).astype(np.uint8) * 255
            inv_only2 = (~only2).astype(np.uint8) * 255

            dist_to_only1 = cv2.distanceTransform(inv_only1, cv2.DIST_L2, 5)
            dist_to_only2 = cv2.distanceTransform(inv_only2, cv2.DIST_L2, 5)

            # Веса: чем дальше от only2, тем больше вес img1
            total = dist_to_only1 + dist_to_only2 + 1e-6
            w1_map = (dist_to_only2 / total)[:, :, np.newaxis]
            w2_map = (dist_to_only1 / total)[:, :, np.newaxis]

            # 🔧 УБРАЛИ GaussianBlur — он ломал размерность массивов.
            # Основной фикс с distanceTransform до only1/only2 уже даёт
            # корректные веса при любой форме перекрытия.

            overlap_3ch = np.stack([overlap] * 3, axis=-1)

            blended = (
                warped1.astype(np.float32) * w1_map +
                warped2.astype(np.float32) * w2_map
            ).astype(np.uint8)

            result = np.where(overlap_3ch, blended, result)

        # 9. Автокроп
        result = self._autocrop(result)

        return result, H

    @staticmethod
    def _autocrop(image):
        """Убирает черные поля по краям."""
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

    stitcher = ImageStitcher(method='sift')
    try:
        result, H = stitcher.stitch(img1, img2)
        print(f"Результат: {result.shape}")
        cv2.imwrite("stitched_result.png", result)
        print("Сохранено: stitched_result.png")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()