import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def camera_axes(position, focal_point, view_up):
    """计算相机右、上、前方向"""
    pos = np.array(position, dtype=float)
    fpt = np.array(focal_point, dtype=float)
    up  = np.array(view_up, dtype=float)

    z = normalize(fpt - pos)   # viewing direction
    x = normalize(np.cross(z, up))
    y = np.cross(x, z)
    return x, y, z

def pixel_to_normalized(px, py, width, height):
    """像素归一化到 [-1,1]"""
    nx = (px - width/2) / (width/2)
    ny = (height/2 - py) / (height/2)  # 翻转 y
    return nx, ny

def pixel_to_world_on_plane(pos, focal_point, view_up,
                            width, height, parallel_scale,
                            px, py):
    """
    将像素点 (px,py) 转换到世界平面上的坐标
    以相机位置 pos 为参考中心
    """
    x_axis, y_axis, z_axis = camera_axes(pos, focal_point, view_up)
    nx, ny = pixel_to_normalized(px, py, width, height)

    dx = nx * parallel_scale * (width / height)
    dy = ny * parallel_scale

    P = np.array(pos) + dx * x_axis + dy * y_axis
    return P, z_axis

def reconstruct_point_from_ortho_with_position(cameras, pixels):
    """
    使用相机 position 作为平面中心，正交投影的多相机 3D 重建
    cameras: list of dict，每个包含
        'position','focal_point','view_up','parallel_scale','width','height'
    pixels: list of (px,py)，与 cameras 一一对应
    return: 重建的 3D 点
    """
    A = np.zeros((3,3))
    b = np.zeros(3)

    for cam, (px, py) in zip(cameras, pixels):
        pos = np.array(cam['position'], dtype=float)
        fpt = np.array(cam['focal_point'], dtype=float)
        vup = np.array(cam['view_up'], dtype=float)

        P, z = pixel_to_world_on_plane(pos, fpt, vup,
                                       cam['width'], cam['height'],
                                       cam['parallel_scale'],
                                       px, py)
        # 投影矩阵： I - z z^T
        I = np.eye(3)
        M = I - np.outer(z, z)

        A += M
        b += M @ P

    # 解最小二乘
    X = np.linalg.solve(A, b)
    return X


if __name__ == "__main__":
    cameras = [
        {
            'position': [0,0,10],
            'focal_point': [0,0,0],
            'view_up': [0,1,0],
            'parallel_scale': 5.0,
            'width': 1000,
            'height': 1000
        },
        {
            'position': [10,0,0],
            'focal_point': [0,0,0],
            'view_up': [0,1,0],
            'parallel_scale': 5.0,
            'width': 1000,
            'height': 1000
        }
    ]

    # 两张图像上都选中中心点
    pixels = [(500, 500), (500, 500)]

    X = reconstruct_point_from_ortho_with_position(cameras, pixels)
    print("Reconstructed 3D point:", X)