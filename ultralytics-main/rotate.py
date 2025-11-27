import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Point, PointStamped, TwistStamped
from geometry_msgs.msg import TwistStamped
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
        # 获取从left_gripper_link到base_link的变换
        tf_buffer.can_transform("base_link", "left_gripper_link", rospy.Time.now(), rospy.Duration(1.0))
        gripper_to_base_transform = tf_buffer.lookup_transform("base_link", "left_gripper_link", rospy.Time(0))
        
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
        
        # 转换点到base_link坐标系
        point_base = transform_point_with_matrix(point_camera, combined_matrix)
        
        return point_base
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
        rospy.logerr("TF transformation failed: %s", e)
        return None
    

def rotate_in_place(vel_pub, direction='left', speed=0.3):
    """
    控制机器人原地旋转。
    direction: 'left' or 'right'
    speed: 角速度 (rad/s)
    """
    twist = TwistStamped()
    twist.header.stamp = rospy.Time.now()
    twist.header.frame_id = 'base_link'
    twist.twist.angular.z = speed if direction == 'left' else -speed
    vel_pub.publish(twist)

def stop_rotation(vel_pub):
    """
    停止旋转。
    """
    twist = TwistStamped()
    twist.header.stamp = rospy.Time.now()
    twist.header.frame_id = 'base_link'
    vel_pub.publish(twist)

def main():
    try:
        rospy.init_node('person_tracker_with_rotation', anonymous=True)
    except:
        pass

    # === ROS 话题 ===
    pub_pos = rospy.Publisher('/person/base_link_3d_position', PointStamped, queue_size=10)
    vel_pub = rospy.Publisher('/motion_target/target_speed_chassis', TwistStamped, queue_size=1)

    # === 初始化 RealSense ===
    pipeline, align, depth_scale, intrinsics = initialize_realsense()

    # === 加载 YOLO 模型 ===
    model = YOLO('yolov8n-pose.pt')

    # === 初始化 Open3D 可视化 ===
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud')
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    # === 相机→机械臂→base_link 变换矩阵 ===
    transformation_matrix = np.array([
        [ 0.01880272, -0.10014192,  0.99479548, -0.06809926],
        [ 0.08711035,  0.99135191,  0.09814878, -0.07856771],
        [-0.99602121, 0.08481152,  0.02736351, -0.00702536],
        [ 0.0,         0.0,          0.0,          1.0       ]
    ])

    # === OpenCV 窗口 ===
    cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
    print("开始人员检测 (按 'q' 键退出)...")

    # === 搜索状态变量 ===
    last_person_x = None
    frame_center_x = 640 // 2
    searching = False
    search_start_time = None
    max_search_time = 8.0  # 最长旋转时间（秒）

    try:
        while not rospy.is_shutdown():
            start = time.time()
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model(color_image)
            display = color_image.copy()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            found_person = False
            person_3d_camera = None

            for result in results:
                if result.keypoints is None:
                    continue
                boxes, keypoints = result.boxes, result.keypoints
                for i, box in enumerate(boxes):
                    if int(box.cls[0]) != 0:  # 仅person类
                        continue
                    found_person = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    kpts = keypoints.xy[i].cpu().numpy()
                    confs = keypoints.conf[i].cpu().numpy()
                    key_data = [(kpts[j][0], kpts[j][1], confs[j]) for j in range(len(kpts))]
                    center = get_body_center_from_keypoints(key_data)
                    cx, cy = ((x1 + x2) // 2, (y1 + y2) // 2) if center is None else map(int, center)
                    last_person_x = cx

                    median_depth = get_median_depth_in_roi(depth_frame, depth_scale, x1, y1, x2, y2)
                    point_3d = get_3d_coordinates(depth_frame, depth_scale, intrinsics, cx, cy, median_depth)
                    if point_3d is not None:
                        person_3d_camera = np.array(point_3d)

                        # --- 转换为 base_link 坐标并发布 ---
                        point_base = transform_point_to_base_link(person_3d_camera, transformation_matrix)
                        if point_base is not None:
                            msg = PointStamped()
                            msg.header.stamp = rospy.Time.now()
                            msg.header.frame_id = "base_link"
                            msg.point = Point(*point_base)
                            pub_pos.publish(msg)
                            rospy.loginfo_throttle(1, f"人物坐标(base_link): x={point_base[0]:.2f}, y={point_base[1]:.2f}, z={point_base[2]:.2f}")

                    # 绘制框与坐标
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
                    if point_3d is not None:
                        x, y, z = point_3d
                        cv2.putText(display, f"({x:.2f},{y:.2f},{z:.2f})m", (x1, y1-10), 0, 0.5, (0,255,0), 1)
                    cv2.circle(display, (cx, cy), 4, (0,0,255), -1)

            # === 如果未检测到人，执行旋转搜索 ===
            if not found_person:
                if last_person_x is not None:
                    twist = TwistStamped()
                    twist.header.stamp = rospy.Time.now()
                    twist.header.frame_id = 'base_link'

                    # 根据上次人物位置决定旋转方向
                    if last_person_x < frame_center_x - 60:
                        twist.twist.angular.z = 0.3   # 左旋
                        if not searching:
                            search_start_time = time.time()
                            searching = True
                        rospy.loginfo_throttle(3, "目标丢失，上次在左侧，向左旋转搜索...")
                    elif last_person_x > frame_center_x + 60:
                        twist.twist.angular.z = -0.3  # 右旋
                        if not searching:
                            search_start_time = time.time()
                            searching = True
                        rospy.loginfo_throttle(3, "目标丢失，上次在右侧，向右旋转搜索...")
                    else:
                        twist.twist.angular.z = 0.0
                        searching = False

                    vel_pub.publish(twist)

                    # 搜索超时则停止旋转
                    if searching and time.time() - search_start_time > max_search_time:
                        rospy.loginfo("搜索超时，停止旋转。")
                        twist.twist.angular.z = 0.0
                        vel_pub.publish(twist)
                        searching = False
                else:
                    # 无历史记录，不动
                    twist = TwistStamped()
                    twist.header.stamp = rospy.Time.now()
                    twist.header.frame_id = 'base_link'
                    vel_pub.publish(twist)
                    searching = False
            else:
                # 找到人后立即停止旋转
                if searching:
                    twist = TwistStamped()
                    twist.header.stamp = rospy.Time.now()
                    twist.header.frame_id = 'base_link'
                    vel_pub.publish(twist)
                    searching = False

            # === 可视化更新 ===
            fps = 1.0 / (time.time() - start + 1e-9)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), 0, 1, (0,0,255), 2)
            cv2.imshow("Person Detection", display)
            cv2.imshow("Depth", depth_colormap)
            vis.poll_events()
            vis.update_renderer()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("程序已退出")

if __name__ == "__main__":
    main()