import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d

def initialize_realsense():
    """初始化RealSense摄像头"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动流
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"启动失败: {e}")
        config.disable_all_streams()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        profile = pipeline.start(config)
    
    # 获取深度传感器和内参
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # 创建对齐对象（深度对齐到彩色）
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 获取彩色流的内参
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    return pipeline, align, depth_scale, intrinsics

def get_3d_coordinates(depth_frame, depth_scale, intrinsics, pixel_x, pixel_y):
    """
    将2D像素坐标转换为3D世界坐标
    :param depth_frame: 深度帧
    :param depth_scale: 深度比例因子
    :param intrinsics: 相机内参
    :param pixel_x: 像素X坐标
    :param pixel_y: 像素Y坐标
    :return: (x, y, z) 3D坐标（单位：米）
    """
    # 确保坐标在图像范围内
    if (pixel_x < 0 or pixel_y < 0 or 
        pixel_x >= intrinsics.width or 
        pixel_y >= intrinsics.height):
        return None
    
    try:
        # 获取深度值（单位：米）
        depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
        if depth <= 0:  # 无效深度值
            return None
        
        # 将像素坐标转换为3D坐标
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth)
        return point  # (x, y, z) in meters
    except RuntimeError:
        return None

def depth_to_points(depth, intrinsic):
    """将深度图转换为点云"""
    K = intrinsic
    Kinv = np.linalg.inv(K)
    height, width = depth.shape
    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3
    D = depth[:, :, None, None]
    pts3D = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    return pts3D[0, :, :, :3, 0]

def main():
    # 初始化RealSense
    try:
        pipeline, align, depth_scale, intrinsics = initialize_realsense()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 加载YOLOv8模型
    try:
        model = YOLO('yolov8n.pt')  # 使用官方预训练模型
    except Exception as e:
        print(f"加载模型失败: {e}")
        pipeline.stop()
        return
    
    # 创建点云可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=1280, height=720)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # 创建显示窗口
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Detection', 800, 600)
    
    try:
        print("开始目标检测 (按 'q' 键退出)...")
        while True:
            # 等待下一组帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            # 转换为OpenCV格式
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 使用YOLOv8进行目标检测
            results = model(color_image)
            
            # 创建用于显示的图像副本
            display_image = color_image.copy()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 获取点云数据
            depth_meters = depth_image * depth_scale
            points_3d = depth_to_points(depth_meters, np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ]))
            
            # 更新点云
            points = points_3d.reshape(-1, 3)
            colors = color_image.reshape(-1, 3) / 255.0
            
            # 随机采样减少点数
            if len(points) > 50000:
                indices = np.random.choice(len(points), 50000, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            
            # 处理检测结果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    conf = box.conf[0]
                    
                    # 转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 计算中心点坐标
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 获取3D坐标
                    point_3d = get_3d_coordinates(
                        depth_frame, 
                        depth_scale, 
                        intrinsics, 
                        center_x, 
                        center_y
                    )
                    
                    # 绘制边界框
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制中心点
                    cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 显示类别和置信度
                    label = f"{result.names[class_id]} {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 显示3D坐标
                    if point_3d is not None:
                        x, y, z = point_3d
                        coord_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
                        cv2.putText(display_image, coord_text, (center_x - 70, center_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        print(f"检测到 {result.names[class_id]}, 3D坐标: ({x:.2f}, {y:.2f}, {z:.2f})m")
            
            # 显示图像
            cv2.imshow('Object Detection', display_image)
            cv2.imshow('Depth', depth_colormap)
            
            # 更新点云可视化
            vis.poll_events()
            vis.update_renderer()
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户请求退出...")
                break
                
    finally:
        # 停止流并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("程序已退出")

if __name__ == "__main__":
    main()