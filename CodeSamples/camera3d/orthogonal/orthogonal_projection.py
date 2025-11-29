"""
需求：
三维坐标系中，给定一个虚拟相机，使用正射投影，从不同角度拍摄的n张图像，给定它们的相机vtk参数，每个vtk参数包含相机位置、焦点位置、向上方向。
已知n张图像的宽和高，现在从n张图像中同时选择同一物体的同一个位置的2d图像点坐标，根据这n个2d图像点坐标计算该点实际的3d坐标。
"""


import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def camera_axes(position, focal_point, view_up):
    z = normalize(np.array(focal_point) - np.array(position))  # viewing direction
    up = normalize(view_up)
    x = normalize(np.cross(z, up))
    y = np.cross(x, z)
    return x, y, z

def pixel_to_normalized(px, py, width, height):
    # 将像素映射到 [-1,1]，并将y翻转（图像坐标 -> 数学坐标）
    nx = (px - (width / 2.0)) / (width / 2.0)
    ny = ((height / 2.0) - py) / (height / 2.0)
    return nx, ny

def reconstruct_point_from_ortho(cameras, pixels):
    """
    cameras: list of dicts, each with keys:
      'position', 'focal_point', 'view_up', 'parallel_scale', 'width', 'height'
    pixels: list of (px, py) pairs (same length as cameras)
    returns: 3D point X (shape (3,))
    """
    assert len(cameras) == len(pixels)
    rows = []
    rhs = []
    for cam, (px, py) in zip(cameras, pixels):
        pos = np.array(cam['position'], dtype=float)
        fpt = np.array(cam['focal_point'], dtype=float)
        view_up = np.array(cam['view_up'], dtype=float)
        s = float(cam['parallel_scale'])  # half-height in world units
        W = cam['width']
        H = cam['height']

        x_axis, y_axis, z_axis = camera_axes(pos, fpt, view_up)
        nx, ny = pixel_to_normalized(px, py, W, H)

        # b values: dot(X, axis) = b
        b_x = np.dot(pos, x_axis) + nx * s
        b_y = np.dot(pos, y_axis) + ny * s

        rows.append(x_axis)  # equation: x_axis^T X = b_x
        rhs.append(b_x)
        rows.append(y_axis)
        rhs.append(b_y)

    A = np.vstack(rows)   # shape (2n, 3)
    b = np.array(rhs)     # shape (2n,)

    # least squares solve
    X, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    return X

# === 示例（伪数据） ===
if __name__ == "__main__":
    cameras = [
        {
            'position': [0,0,10],
            'focal_point': [0,0,0],
            'view_up': [0,1,0],
            'parallel_scale': 5.0,  # 视窗半高 world units
            'width': 640, 'height': 480
        },
        {
            'position': [10,0,0],
            'focal_point': [0,0,0],
            'view_up': [0,1,0],
            'parallel_scale': 5.0,
            'width': 640, 'height': 480
        }
    ]
    pixels = [(500,500), (500,500)]  # 中心像素
    X = reconstruct_point_from_ortho(cameras, pixels)
    print("Reconstructed X:", X)
