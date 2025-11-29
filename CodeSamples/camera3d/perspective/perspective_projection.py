import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Triangulation:
    def __init__(self):
        pass
    
    def triangulate_points(self, points_2d, projection_matrices):
        """
        使用SVD方法通过叉积方程三角化3D点
        
        参数:
        points_2d: list of numpy arrays, 每个元素是 [u, v] 坐标
        projection_matrices: list of numpy arrays, 每个元素是 3x4 投影矩阵
        
        返回:
        point_3d: numpy array [X, Y, Z] 3D坐标
        """
        
        n_views = len(points_2d)
        A = np.zeros((2 * n_views, 4))
        
        for i in range(n_views):
            # 获取第i个视图的2D点和投影矩阵
            u, v = points_2d[i]
            M = projection_matrices[i]
            
            # 投影矩阵的行
            m1 = M[0, :]  # 第一行
            m2 = M[1, :]  # 第二行  
            m3 = M[2, :]  # 第三行
            
            # 构建叉积方程 (从叉积 p × (MP) = 0 得到两个独立方程)
            # 方程1: v * (m3·P) - (m2·P) = 0
            A[2*i, :] = v * m3 - m2
            
            # 方程2: (m1·P) - u * (m3·P) = 0
            A[2*i + 1, :] = m1 - u * m3
        
        # 使用SVD求解 AX = 0
        U, S, Vt = np.linalg.svd(A)
        
        # 解是V的最后一列（对应最小奇异值）
        point_3d_homogeneous = Vt[-1, :]
        
        # 从齐次坐标转换到3D坐标
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
        
        return point_3d
    
    def reproject_points(self, point_3d, projection_matrices):
        """
        将3D点重投影到各个视图，用于验证结果
        
        参数:
        point_3d: numpy array [X, Y, Z] 3D坐标
        projection_matrices: list of numpy arrays, 每个元素是 3x4 投影矩阵
        
        返回:
        reprojected_points: list of numpy arrays, 重投影的2D点
        """
        point_3d_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        reprojected_points = []
        
        for M in projection_matrices:
            # 投影到图像平面
            point_2d_homogeneous = M @ point_3d_homogeneous
            # 从齐次坐标转换到2D坐标
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            reprojected_points.append(point_2d)
        
        return reprojected_points
    
    def calculate_reprojection_error(self, original_points, reprojected_points):
        """
        计算重投影误差
        
        参数:
        original_points: 原始的2D观测点
        reprojected_points: 重投影的2D点
        
        返回:
        errors: 每个视图的重投影误差
        total_error: 总误差
        """
        errors = []
        for orig, reproj in zip(original_points, reprojected_points):
            error = np.linalg.norm(orig - reproj)
            errors.append(error)
        
        total_error = np.mean(errors)
        return errors, total_error

def create_example_cameras():
    """
    创建示例相机配置
    """
    # 内参矩阵 (假设所有相机相同)
    K = np.array([
        [800, 0, 320],
        [0, 800, 240], 
        [0, 0, 1]
    ])
    
    # 创建4个不同位置的相机
    cameras = []
    
    # 相机1: 在原点，看向z轴正方向
    R1 = np.eye(3)
    t1 = np.array([0, 0, 0]).reshape(3, 1)
    M1 = K @ np.hstack([R1, t1])
    cameras.append(M1)
    
    # 相机2: 在x轴偏移，看向同一个点
    R2 = np.eye(3)
    t2 = np.array([1, 0, 0]).reshape(3, 1)
    M2 = K @ np.hstack([R2, t2])
    cameras.append(M2)
    
    # 相机3: 在y轴偏移
    R3 = np.eye(3)
    t3 = np.array([0, 1, 0]).reshape(3, 1)
    M3 = K @ np.hstack([R3, t3])
    cameras.append(M3)
    
    # 相机4: 在z轴偏移
    R4 = np.eye(3)
    t4 = np.array([0, 0, 1]).reshape(3, 1)
    M4 = K @ np.hstack([R4, t4])
    cameras.append(M4)
    
    return cameras

def generate_synthetic_data():
    """
    生成合成数据用于测试
    """
    # 创建一个3D点
    point_3d_true = np.array([2.0, 1.5, 5.0])
    
    # 创建相机投影矩阵
    projection_matrices = create_example_cameras()
    
    # 生成2D观测点（添加少量噪声）
    points_2d = []
    for M in projection_matrices:
        point_homogeneous = np.array([point_3d_true[0], point_3d_true[1], point_3d_true[2], 1.0])
        point_2d_homogeneous = M @ point_homogeneous
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        
        # 添加高斯噪声
        noise = np.random.normal(0, 0.5, 2)
        point_2d_noisy = point_2d + noise
        
        points_2d.append(point_2d_noisy)
    
    return points_2d, projection_matrices, point_3d_true

def visualize_results(point_3d_true, point_3d_estimated, cameras, points_2d, reprojected_points):
    """
    可视化结果
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 绘制真实3D点
    ax1.scatter(point_3d_true[0], point_3d_true[1], point_3d_true[2], 
               c='g', marker='o', s=100, label='真实3D点')
    
    # 绘制估计的3D点
    ax1.scatter(point_3d_estimated[0], point_3d_estimated[1], point_3d_estimated[2],
               c='r', marker='x', s=100, label='估计3D点')
    
    # 绘制相机位置
    camera_positions = []
    for M in cameras:
        # 从投影矩阵提取相机位置: C = -R^T * t
        R = M[:, :3]
        t = M[:, 3]
        camera_center = -R.T @ t
        camera_positions.append(camera_center)
        ax1.scatter(camera_center[0], camera_center[1], camera_center[2], 
                   c='b', marker='^', s=50)
    
    # 连接相机和3D点
    for cam_pos in camera_positions:
        ax1.plot([cam_pos[0], point_3d_estimated[0]], 
                [cam_pos[1], point_3d_estimated[1]], 
                [cam_pos[2], point_3d_estimated[2]], 'b--', alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D三角化结果')
    ax1.legend()
    
    # 重投影误差可视化
    ax2 = fig.add_subplot(132)
    errors = []
    for i, (orig, reproj) in enumerate(zip(points_2d, reprojected_points)):
        error = np.linalg.norm(orig - reproj)
        errors.append(error)
        ax2.plot([orig[0], reproj[0]], [orig[1], reproj[1]], 'r-', alpha=0.7)
        ax2.scatter(orig[0], orig[1], c='b', marker='o', s=50, label=f'观测点 {i+1}' if i == 0 else "")
        ax2.scatter(reproj[0], reproj[1], c='r', marker='x', s=50, label=f'重投影点 {i+1}' if i == 0 else "")
    
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_title('重投影误差')
    ax2.legend()
    ax2.grid(True)
    
    # 误差柱状图
    ax3 = fig.add_subplot(133)
    views = range(1, len(errors) + 1)
    ax3.bar(views, errors)
    ax3.set_xlabel('视图编号')
    ax3.set_ylabel('重投影误差 (像素)')
    ax3.set_title('各视图重投影误差')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示三角化过程
    """
    print("=== 3D点三角化演示 ===")
    
    # 生成合成数据
    points_2d, projection_matrices, point_3d_true = generate_synthetic_data()
    
    print(f"真实3D点: {point_3d_true}")
    print(f"生成的2D观测点数量: {len(points_2d)}")
    
    # 创建三角化器
    triangulator = Triangulation()
    
    # 执行三角化
    point_3d_estimated = triangulator.triangulate_points(points_2d, projection_matrices)
    
    print(f"估计的3D点: {point_3d_estimated}")
    
    # 计算重投影
    reprojected_points = triangulator.reproject_points(point_3d_estimated, projection_matrices)
    
    # 计算误差
    errors, total_error = triangulator.calculate_reprojection_error(points_2d, reprojected_points)
    
    print(f"\n重投影误差:")
    for i, error in enumerate(errors):
        print(f"  视图 {i+1}: {error:.4f} 像素")
    print(f"平均误差: {total_error:.4f} 像素")
    
    # 计算3D位置误差
    position_error = np.linalg.norm(point_3d_true - point_3d_estimated)
    print(f"\n3D位置误差: {position_error:.4f} 单位")
    
    # 可视化结果
    visualize_results(point_3d_true, point_3d_estimated, projection_matrices, points_2d, reprojected_points)

if __name__ == "__main__":
    main()