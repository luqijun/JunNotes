import cv2
import numpy as np
from typing import List, Tuple


def find_concave_contours(image: np.ndarray, min_depth=10, min_angle=45,
                          visualize: bool = False) -> List[np.ndarray]:
    """
    提取图像轮廓后，筛选轮廓中凹陷的部分，返回凹陷的轮廓

    参数:
        image: 输入图像 (BGR或灰度图)
        visualize: 是否显示可视化结果

    返回:
        凹陷轮廓列表
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    concave_contours = []

    for contour in contours:
        # 忽略太小的轮廓
        if cv2.contourArea(contour) < 100:
            continue

        # 计算凸包
        hull = cv2.convexHull(contour, returnPoints=True)

        # 计算凸包缺陷
        hull_indices = cv2.convexHull(contour, returnPoints=False)

        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)

            if defects is not None:
                # 提取明显的凹陷点
                concave_points = []

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # 计算凹陷角度（余弦定理）
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                    # d是凹陷深度(距离*256)
                    # 筛选深度超过阈值且角度较小的尖锐凹陷点
                    depth = d / 256.0
                    if depth > min_depth and angle < min_angle:  # 约10个像素的深度
                        concave_points.append({
                            'start': start,
                            'end': end,
                            'far': far,
                            'depth': depth,
                            'angle': np.degrees(angle)
                        })

                # 如果有明显凹陷，则认为是凹陷轮廓
                if len(concave_points) > 0:
                    concave_contours.append({
                        'contour': contour,
                        'hull': hull,
                        'concave_points': concave_points,
                        'defects': defects
                    })

    # 可视化
    if visualize:
        vis_img = image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        for item in concave_contours:
            contour = item['contour']
            hull = item['hull']
            concave_points = item['concave_points']['far']

            # 绘制原始轮廓 (绿色)
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)

            # 绘制凸包 (蓝色)
            cv2.drawContours(vis_img, [hull], -1, (255, 0, 0), 2)

            # 绘制凹陷点 (红色)
            for point in concave_points:
                cv2.circle(vis_img, point, 5, (0, 0, 255), -1)

        cv2.imshow('Concave Contours Detection', vis_img)
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return concave_contours


def example_usage():
    """使用示例"""
    # 创建一个测试图像（带有凹陷形状）
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # 绘制一个凹陷的形状（星形）
    pts = np.array([
        [300, 50], [320, 150], [420, 150], [340, 210],
        [370, 310], [300, 250], [230, 310], [260, 210],
        [180, 150], [280, 150]
    ], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # 绘制一个矩形（无凹陷）
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)

    # 绘制一个带凹口的矩形
    pts2 = np.array([
        [450, 250], [550, 250], [550, 280], [520, 280],
        [520, 320], [550, 320], [550, 350], [450, 350]
    ], np.int32)
    cv2.fillPoly(img, [pts2], (255, 255, 255))

    # 检测凹陷轮廓
    concave_contours = find_concave_contours(img, visualize=True)

    # 打印结果
    print(f"找到 {len(concave_contours)} 个凹陷轮廓")
    for i, item in enumerate(concave_contours):
        print(f"轮廓 {i+1}: {len(item['concave_points'])} 个凹陷点")


if __name__ == "__main__":
    example_usage()
