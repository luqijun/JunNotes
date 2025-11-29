# 思路（简要）

在正交投影（orthographic）下，每台相机只给出**两个线性约束**：点在相机坐标系的 `x` 和 `y` 分量（即沿相机的右向和上向的投影）等于该像素对应的图像坐标对应的世界平面点（即把像素坐标换算到世界单位坐标）。因此从 $n$ 台相机我们得到 $2n$ 个线性方程，未知量是世界坐标 $X=(X_x,X_y,X_z)^T$。把这些方程堆成矩阵形式后用线性最小二乘解出 $X$。

**关键要素**（必须有或需要明确）：

* 每个相机的方向基（在世界坐标下）：右向 $\mathbf{r}_i$，上向 $\mathbf{u}_i$，视线方向（投影方向）$\mathbf{w}_i$（通常 $\mathbf{w}_i$ = normalize(focal-point − position)。
* 像素坐标 $(p_{x},p_{y})$ 到相机“图像平面坐标”（世界单位）的映射尺度（即每像素对应多少世界长度，或者等价地相机的**parallel scale** / 视窗半高）。

  * 如果没有给出尺度，只用像素单位做归一化（例如把像素映射到 $[-1,1]$），则只能恢复点的位置 **相对于每台相机视窗的尺度**，整体绝对大小会有尺度模糊（需用户给出至少一个相机的平行缩放参数）。
* 我们把每个相机的图像平面中心（像主点）取为相机的 position，并基于它把像素映射到世界单位坐标（下面代码用 position 作为中心点）。


在 VTK 中，`parallel_scale` 表示：

> 图像高度的一半在世界坐标中的大小（沿着相机 up 向量的方向）。

具体来说：

* 如果相机图像的像素高度是 `H`，
* 并且 `parallel_scale = s`，
  那么 **整个图像在世界坐标系中的高度就是 `2 * s`**。
  于是：
* 世界空间中 1 个单位长度 = 图像高度对应的 $H / (2s)$ 个像素。
* 或者反过来：1 像素对应的世界长度 = $2s / H$。

---

# 数学推导（简短）

对第 (i) 台相机，定义单位向量：
$$
\mathbf{z}_i = \mathrm{normalize}(\mathbf{f}_i - \mathbf{p}_i) \quad(\text{视线方向})
$$
$$
\mathbf{x}_i = \mathrm{normalize}(\mathbf{z}_i \times \mathbf{v}_i) \quad(\text{相机右向})
$$
$$
\mathbf{y}_i = \mathbf{x}_i \times \mathbf{z}_i \quad(\text{相机上向})
$$

把像素 $(p_x,p_y)$ 规范化到 $(\tilde x,\tilde y)$（范围 $[-1,1]$）：
$$
\tilde x = \frac{p_x - (W/2)}{W/2},\quad \tilde y = \frac{(H/2) - p_y}{H/2}
$$
（注意 y 通常要翻转）

给定该相机的 `parallel_scale` $s_i$（表示图像平面从中心到顶边的世界单位距离），假设横向和纵向平行尺度相同，像素在世界平面上的坐标：
$$
\mathbf{P}_i = \mathbf{pos}_i + (\tilde{x} \cdot s_i) \mathbf{x}_i + (\tilde{y} \cdot s)\mathbf{y}_i
$$
或者在横向和纵向平行尺度不相同情况下，像素在世界平面上的坐标：
$$
\mathbf{P}_i = \mathbf{pos}_i + (\tilde{x} \cdot s_i \cdot \tfrac{W}{H}) \mathbf{x}_i + (\tilde{y} \cdot s)\mathbf{y}_i 
$$
其中：

* $\mathbf{pos}$：相机位置（作为平面坐标系的中心）。
* $\mathbf{x}, \mathbf{y}$：相机的右 / 上方向向量。
* $\tilde{x}, \tilde{y}$：归一化像素坐标$[-1,1]$。
* $s$：parallel_scale。

计算$\mathbf{P}_i$在$\mathbf{x}_i$和$\mathbf{y}_i$方向的投影值，即
$$
b_{i,x} = \mathbf{P}_i x = \mathbf{pos}_i \cdot \mathbf{x}_i + \tilde x \cdot s_i \\
b_{i,y} = \mathbf{P}_i y = \mathbf{pos}_i \cdot \mathbf{y}_i + \tilde y \cdot s_i
$$
因为对于目标 3D 点 $\mathbf{X}=(\mathbf{X}_x,\mathbf{X}_y,\mathbf{X}_z)^T$：
$$
\mathbf{x}_i^T \mathbf{X} = b_{i,x},\quad \mathbf{y}_i^T \mathbf{X} = b_{i,y}.
$$
把所有摄像机的这些方程堆叠成线性系统 (A X = b)（其中每台相机贡献两行）。用最小二乘解：
$$
X = (A^T A)^{-1} A^T b
$$
或用稳健数值方法 `np.linalg.lstsq`。

---

# Python 实现（假设已有 parallel_scale）

下面代码实现了上述步骤，输入为每台相机的 `position,focal_point,view_up,parallel_scale,width,height` 与像素点 `px,py`（对应同一 3D 点在每张图像的像素坐标）。

```python
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

```

---

# 关于尺度与不确定性的说明

1. **必须的尺度信息**：正交投影缺少“透视收敛”信息，因此在没有 **世界单位的像素尺度（parallel_scale / 每像素对应的世界长度）** 的情况下，问题**对整体尺度存在模糊**：你只能恢复点在每个相机视窗坐标系下的位置比例，但不能得到唯一的绝对世界坐标（除非你知道至少一个尺度参数或场景中有尺度参照）。
2. **如果没有 parallel_scale**，常见替代：

   * 若你只关心**相对位置**或想在某个参考尺度下工作：可把 `parallel_scale = 1.0`（或任意常数），得到按比例的解；后续用已有的实际测量来缩放回真实世界。
   * 或者如果你有相机的“视域在世界坐标的边界”或“像素对应的真实世界间距”，可以用它来算 `parallel_scale`：例如 `parallel_scale = (real_view_height/2)`，而 `real_view_height` = pixels_in_height * meter_per_pixel。
3. **数值稳定性**：当所有相机视线非常接近共面或平行（观测几何退化）时，矩阵 (A) 会接近欠定，解不可靠。最好保证相机视角分布良好（视线方向差异大）。

---

