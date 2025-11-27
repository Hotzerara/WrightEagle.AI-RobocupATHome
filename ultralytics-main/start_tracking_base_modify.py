# 测试发现机器人从夹爪到base_link的TF矩阵有问题，但是机器人夹爪到手臂上的base的TF是没问题的，
# 我们暂时可以计算物体/人物在手臂base下的坐标，然后加/减一个偏移值就得到了物体在base_link下的正确坐标

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply

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

def get_median_depth_in_roi(depth_frame, depth_scale, x1, y1, x2, y2):
    """
    获取边界框内有效深度的中值
    :param depth_frame: 深度帧
    :param depth_scale: 深度比例因子
    :param x1, y1: 边界框左上角坐标
    :param x2, y2: 边界框右下角坐标
    :return: 有效深度的中值（单位：米），如果没有有效深度返回None
    """
    # 确保坐标在图像范围内
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))
    
    # 提取边界框内的深度数据
    depth_data = np.asanyarray(depth_frame.get_data())
    roi = depth_data[y1:y2, x1:x2]
    
    # 转换为实际深度值（米）
    roi_meters = roi.astype(float) * depth_scale
    
    # 过滤掉无效深度值（0表示无效）
    valid_depths = roi_meters[roi_meters > 0.1]  # 只考虑深度大于10cm的点
    
    if len(valid_depths) == 0:
        return None
    
    # 计算有效深度的中值
    median_depth = np.median(valid_depths)
    return median_depth

def get_3d_coordinates(depth_frame, depth_scale, intrinsics, pixel_x, pixel_y, depth_value=None):
    """
    将2D像素坐标转换为3D世界坐标
    :param depth_frame: 深度帧
    :param depth_scale: 深度比例因子
    :param intrinsics: 相机内参
    :param pixel_x: 像素X坐标
    :param pixel_y: 像素Y坐标
    :param depth_value: 可选的深度值（如果提供则使用）
    :return: (x, y, z) 3D坐标（单位：米）
    """
    # 确保坐标在图像范围内
    if (pixel_x < 0 or pixel_y < 0 or 
        pixel_x >= intrinsics.width or 
        pixel_y >= intrinsics.height):
        return None
    
    try:
        # 获取深度值（单位：米）
        if depth_value is None:
            depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
        else:
            depth = depth_value
            
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

def get_body_center_from_keypoints(keypoints):
    """
    从人体关键点计算更准确的身体中心位置
    :param keypoints: YOLOv8返回的关键点数据
    :return: (x, y) 身体中心坐标
    """
    # 关键点索引定义（COCO格式）
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    
    # 尝试使用肩膀和臀部关键点
    valid_points = []
    
    # 肩膀中心点
    if keypoints[LEFT_SHOULDER][2] > 0.1 and keypoints[RIGHT_SHOULDER][2] > 0.1:
        shoulder_center = (
            (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2,
            (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        )
        valid_points.append(shoulder_center)
    
    # 臀部中心点
    if keypoints[LEFT_HIP][2] > 0.1 and keypoints[RIGHT_HIP][2] > 0.1:
        hip_center = (
            (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
            (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
        )
        valid_points.append(hip_center)
    
    # 如果没有有效的肩膀或臀部点，使用边界框中心
    if not valid_points:
        return None
    
    # 计算所有有效点的平均值
    center_x = sum(p[0] for p in valid_points) / len(valid_points)
    center_y = sum(p[1] for p in valid_points) / len(valid_points)
    
    return (center_x, center_y)

def transform_point_with_matrix(point, transform_matrix):
    """
    使用4x4变换矩阵转换点
    """
    # 将点转换为齐次坐标
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    
    # 应用变换矩阵
    transformed_point = np.dot(transform_matrix, point_homogeneous)
    
    # 返回非齐次坐标
    return transformed_point[:3]

def transform_point_to_base_link(point_camera, transformation_matrix):
    """
    将相机坐标系下的点转换到base_link坐标系
    """
    # 初始化ROS节点（如果尚未初始化）
    try:
        rospy.init_node('camera_to_base_transform', anonymous=True)
    except:
        pass  # 节点已初始化
    
    # 创建TF缓冲区和监听器
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    try:
        # 获取从right_gripper_link到base_link的变换
        # modify by linrunfeng
        # 先计算人物到左臂base的坐标，然后修正偏移值得到基于base_link的坐标
        
        tf_buffer.can_transform("left_arm_base_link", "left_gripper_link", rospy.Time.now(), rospy.Duration(1.0))
        gripper_to_base_transform = tf_buffer.lookup_transform("left_arm_base_link", "left_gripper_link", rospy.Time(0))
        
        # 提取变换信息
        translation = np.array([
            gripper_to_base_transform.transform.translation.x,
            gripper_to_base_transform.transform.translation.y,
            gripper_to_base_transform.transform.translation.z
        ])
        
        rotation = np.array([
            gripper_to_base_transform.transform.rotation.x,
            gripper_to_base_transform.transform.rotation.y,
            gripper_to_base_transform.transform.rotation.z,
            gripper_to_base_transform.transform.rotation.w
        ])
        
        # 创建从right_gripper_link到base_link的变换矩阵
        rotation_matrix = quaternion_matrix(rotation)
        gripper_to_base_matrix = np.identity(4)
        gripper_to_base_matrix[:3, :3] = rotation_matrix[:3, :3]
        gripper_to_base_matrix[:3, 3] = translation

        # 组合变换：先应用相机到gripper的变换，再应用gripper到base的变换
        combined_matrix = np.dot(gripper_to_base_matrix, transformation_matrix)
        # camera_to_base
        
        # 转换点到base_link坐标系
        point_base = transform_point_with_matrix(point_camera, combined_matrix)
        
        # modify by linrunfeng

        point_base[1] += 0.3

        return point_base
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
        rospy.logerr("TF transformation failed: %s", e)
        return None

def main():
    # 初始化ROS节点和Publisher
    rospy.init_node('person_detection_3d', anonymous=True)
    position_pub = rospy.Publisher('/person/base_link_3d_position', PointStamped, queue_size=10)
    
    # 初始化RealSense
    try:
        pipeline, align, depth_scale, intrinsics = initialize_realsense()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 加载YOLOv8姿态估计模型（支持关键点检测）
    try:
        # 使用预训练的姿态估计模型
        model = YOLO('yolov8n-pose.pt')  # 专门用于姿态估计的模型
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
    cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Detection', 800, 600)
    
    # 用于存储检测到的人员信息
    detected_persons = []
    
    # 示例：相机到right_gripper_link的4x4转换矩阵
    # 注意：这是一个示例矩阵，您需要替换为实际的标定结果
    transformation_matrix = np.array([
        [-0.02937859, -0.1152559 ,  0.99290129, -0.02411462],
        [ 0.01197522,  0.99321818,  0.11564701, -0.06956071],
        [-0.99949662,  0.01528775, -0.02779914,  0.01524878],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
    try:
        print("开始人员检测 (按 'q' 键退出)...")
        while not rospy.is_shutdown():
            start_time = time.time()
            
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
            
            # 使用YOLOv8进行姿态估计
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
            
            # 清空前一帧检测到的人员
            detected_persons.clear()
            
            # 处理检测结果 - 只处理person类
            for result in results:
                if result.keypoints is None:
                    continue
                    
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    conf = box.conf[0]
                    
                    # 只处理person类
                    if class_id != 0:  # person类ID通常为0
                        continue
                    
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取当前检测的关键点
                    kpts = keypoints.xy[i].cpu().numpy()
                    confs = keypoints.conf[i].cpu().numpy()
                    
                    # 组合关键点坐标和置信度
                    person_keypoints = []
                    for j in range(len(kpts)):
                        x, y = kpts[j]
                        c = confs[j] if j < len(confs) else 0.0
                        person_keypoints.append((x, y, c))
                    
                    # 使用关键点计算更准确的身体中心
                    body_center = get_body_center_from_keypoints(person_keypoints)
                    
                    # 如果无法通过关键点确定身体中心，使用边界框中心
                    if body_center is None:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                    else:
                        center_x, center_y = int(body_center[0]), int(body_center[1])
                    
                    # 获取边界框内的中值深度
                    median_depth = get_median_depth_in_roi(
                        depth_frame, 
                        depth_scale, 
                        x1, y1, x2, y2
                    )
                    
                    # 获取3D坐标
                    if median_depth is not None:
                        point_3d = get_3d_coordinates(
                            depth_frame, 
                            depth_scale, 
                            intrinsics, 
                            center_x, 
                            center_y,
                            depth_value=median_depth
                        )
                    else:
                        # 如果中值深度无效，尝试直接获取中心点深度
                        point_3d = get_3d_coordinates(
                            depth_frame, 
                            depth_scale, 
                            intrinsics, 
                            center_x, 
                            center_y
                        )
                    
                    # 存储检测到的人员信息
                    if point_3d is not None:
                        detected_persons.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            '3d_position': point_3d,
                            'confidence': conf
                        })
                    
                    # 绘制边界框
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制中心点
                    cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 绘制关键点
                    for kpt in person_keypoints:
                        x, y, conf = kpt
                        if conf > 0.2:  # 只绘制置信度高于0.2的关键点
                            cv2.circle(display_image, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # 显示类别和置信度
                    label = f"Person {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 显示3D坐标
                    if point_3d is not None:
                        x, y, z = point_3d
                        coord_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
                        cv2.putText(display_image, coord_text, (center_x - 70, center_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # 转换到base_link坐标系并发布
                        point_camera = np.array([x, y, z])
                        point_base = transform_point_to_base_link(point_camera, transformation_matrix)
                        
                        if point_base is not None:
                            print(f"点在base_link坐标系下的坐标: {point_base}")
                            
                            # 发布到ROS topic
                            point_msg = PointStamped()
                            point_msg.header = Header()
                            point_msg.header.stamp = rospy.Time.now()
                            point_msg.header.frame_id = "base_link"
                            point_msg.point.x = point_base[0]
                            point_msg.point.y = point_base[1]
                            point_msg.point.z = point_base[2]
                            
                            position_pub.publish(point_msg)
                            print(f"已发布到 /person/base_link_3d_position: ({point_base[0]:.3f}, {point_base[1]:.3f}, {point_base[2]:.3f})")
            
            # 在图像左上角显示检测到的人数和FPS
            fps = 1.0 / (time.time() - start_time + 1e-9)
            cv2.putText(display_image, f"Persons: {len(detected_persons)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow('Person Detection', display_image)
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