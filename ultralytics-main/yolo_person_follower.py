#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
这是一个集成了YOLOv8姿态检测、RealSense 3D定位与ROS导航跟随功能的统一节点。
它将两个独立脚本的功能合并在一起，通过内部函数调用传递数据，提高了效率。
功能流程：
1. 使用RealSense相机获取彩色图和深度图。
2. 在彩色图上运行YOLOv8-pose模型，追踪并检测人体及其关键点。
3. 锁定一个目标人员（如距离最近的），并持续追踪。
4. 结合深度图，计算出目标人员在相机坐标系下的3D坐标。
5. 使用TF2将3D坐标从相机坐标系转换到机器人基座坐标系('base_link')。
6. 将'base_link'坐标送入跟随逻辑模块。
7. 跟随逻辑计算一个安全的导航目标点，并发布给move_base。
8. (可选) 提供OpenCV和Open3D进行2D和3D的可视化。
"""

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import cv2
import time
import math
import torch

# ROS消息类型
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from visualization_msgs.msg import Marker

# 第三方库
from ultralytics import YOLO
import pyrealsense2 as rs
import open3d as o3d

class YoloPersonFollower:
    def __init__(self):
        # === 1. 初始化ROS节点 ===
        rospy.init_node('yolo_person_follower_node')
        rospy.loginfo("启动YOLO人员检测与跟随集成节点...")

        # === 2. 加载ROS参数 (合并自两个脚本) ===
        # --- 跟随逻辑参数 ---
        self.goal_update_distance = rospy.get_param('~goal_update_distance', 0.5)
        self.follow_distance = rospy.get_param('~follow_distance', 1.0)
        self.global_frame = rospy.get_param('~global_frame', 'map')
        # --- 感知与通用参数 ---
        self.yolo_model_path = rospy.get_param('~yolo_model', 'yolov8n-pose.pt')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.enable_visualization = rospy.get_param('~enable_visualization', False) # 默认关闭以避免远程桌面问题
        self.min_keypoint_confidence = rospy.get_param('~min_keypoint_confidence', 0.2)
        self.min_detection_confidence = rospy.get_param('~min_detection_confidence', 0.5)
        self.target_lost_threshold = rospy.get_param('~target_lost_threshold', 30) # 目标丢失超过30帧后重新搜索

        # === 3. 初始化TF2 (只执行一次) ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === 4. 初始化ROS发布者 (合并自两个脚本) ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/person_marker', Marker, queue_size=1)

        # === 5. 初始化感知设备 ===
        try:
            self.pipeline, self.align, self.depth_scale, self.intrinsics = self._initialize_realsense()
        except Exception as e:
            rospy.logfatal("初始化RealSense失败: {}".format(e))
            return
        
        try:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            rospy.loginfo("正在使用设备: {}".format(device))
            self.model = YOLO(self.yolo_model_path).to(device)
            rospy.loginfo("YOLO模型 '{}' 加载成功。".format(self.yolo_model_path))
        except Exception as e:
            rospy.logfatal("加载YOLO模型失败: {}".format(e))
            self.pipeline.stop()
            return
        
        # === 6. 初始化可视化窗口 (如果启用) ===
        self.vis = None
        if self.enable_visualization:
            self._initialize_visualization()

        # === 7. 初始化状态变量 (合并自两个脚本) ===
        self.last_goal_position = None
        self.target_id = None # 当前锁定的追踪ID
        self.target_lost_frames = 0 # 目标连续丢失的帧数
        self.frame_count = 0 # 帧计数器，用于降低可视化更新频率

        rospy.loginfo("节点初始化完成，准备进入主循环。")

    # -------------------------------------------------------------------
    # 初始化与清理方法
    # -------------------------------------------------------------------

    def _initialize_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        rospy.loginfo("RealSense相机初始化成功。")
        return pipeline, align, depth_scale, intrinsics

    def _initialize_visualization(self):
        try:
            cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='3D Point Cloud')
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
            rospy.loginfo("可视化窗口初始化成功。")
        except Exception as e:
            rospy.logwarn("初始化可视化窗口失败 (可能是远程桌面环境): {}".format(e))
            self.enable_visualization = False

    def _cleanup(self):
        self.pipeline.stop()
        if self.vis:
            self.vis.destroy_window()
        cv2.destroyAllWindows()
        rospy.loginfo("节点已关闭。")

    # -------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------
    
    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            # 1. 获取图像帧
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                rate.sleep()
                continue
            color_image = np.asanyarray(color_frame.get_data())
            
            # 2. YOLOv8追踪
            results = self.model.track(color_image, persist=True, verbose=False)
            
            # 3. 处理检测结果并转换为可用数据结构
            persons_in_frame = self._process_yolo_results(results, depth_frame)

            # 4. 更新追踪逻辑
            target_person = self._update_tracking(persons_in_frame)

            # 5. 如果找到目标，则触发跟随逻辑
            if target_person:
                point_base_link = self._transform_point(
                    target_person['point_camera'], self.camera_frame, self.robot_base_frame
                )
                if point_base_link:
                    # 直接调用跟随逻辑，无需通过ROS话题
                    self._process_person_for_following(point_base_link)

            # 6. 更新可视化
            if self.enable_visualization:
                self.frame_count += 1
                self._update_visualization(color_image, depth_frame, persons_in_frame)

            rate.sleep()
            
        self._cleanup()

    # -------------------------------------------------------------------
    # 感知与计算的辅助方法 (来自原脚本1)
    # -------------------------------------------------------------------

    def _process_yolo_results(self, results, depth_frame):
        persons = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes, keypoints = results[0].boxes, results[0].keypoints
            for i in range(len(boxes)):
                if int(boxes.cls[i]) == 0 and boxes.conf[i] > self.min_detection_confidence:
                    bbox = boxes.xyxy[i].cpu().numpy()
                    center_2d = self._get_body_center_from_keypoints(
                        keypoints.xy[i].cpu().numpy(), keypoints.conf[i].cpu().numpy(), bbox
                    )
                    if center_2d is None: continue

                    median_depth = self._get_median_depth_in_roi(depth_frame, bbox)
                    if median_depth is None: continue

                    point_camera = self._get_3d_coordinates(depth_frame, center_2d[0], center_2d[1], median_depth)
                    if point_camera is None: continue
                    
                    persons.append({
                        'id': int(boxes.id[i]), 'point_camera': point_camera,
                        'bbox': bbox, 'center_2d': center_2d, 'distance': median_depth
                    })
        return persons

    def _get_body_center_from_keypoints(self, keypoints_xy, keypoints_conf, bbox):
        # (此处的具体实现与您的原代码相同)
        valid_points = []
        # 肩膀中心点
        if keypoints_conf[5] > self.min_keypoint_confidence and keypoints_conf[6] > self.min_keypoint_confidence:
            valid_points.append(((keypoints_xy[5][0] + keypoints_xy[6][0]) / 2, (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2))
        # 臀部中心点
        if keypoints_conf[11] > self.min_keypoint_confidence and keypoints_conf[12] > self.min_keypoint_confidence:
            valid_points.append(((keypoints_xy[11][0] + keypoints_xy[12][0]) / 2, (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2))
        
        if not valid_points: return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        
        return (int(sum(p[0] for p in valid_points) / len(valid_points)), int(sum(p[1] for p in valid_points) / len(valid_points)))

    def _get_median_depth_in_roi(self, depth_frame, bbox):
        # (此处的具体实现与您的原代码相同)
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_frame.get_width() - 1, x2), min(depth_frame.get_height() - 1, y2)
        if x1 >= x2 or y1 >= y2: return None
        roi_depth = np.asanyarray(depth_frame.get_data())[y1:y2, x1:x2].astype(float) * self.depth_scale
        valid_depths = roi_depth[roi_depth > 0.1]
        return np.median(valid_depths) if len(valid_depths) > 0 else None

    def _get_3d_coordinates(self, depth_frame, pixel_x, pixel_y, depth_value):
        # (此处的具体实现与您的原代码相同)
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth_value)

    def _transform_point(self, point_tuple, from_frame, to_frame):
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(0.2))
            point_stamped = PointStamped(header=Header(stamp=rospy.Time.now(), frame_id=from_frame), point=Point(*point_tuple))
            return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, "坐标变换从 {} 到 {} 失败: {}".format(from_frame, to_frame, e))
            return None

    # -------------------------------------------------------------------
    # 追踪与跟随的核心逻辑 (部分来自原脚本2)
    # -------------------------------------------------------------------

    def _update_tracking(self, persons_in_frame):
        target_person = None
        if self.target_id is not None:
            for p in persons_in_frame:
                if p['id'] == self.target_id:
                    target_person, self.target_lost_frames = p, 0
                    break
            if target_person is None: self.target_lost_frames += 1
        
        if self.target_id is None or self.target_lost_frames > self.target_lost_threshold:
            if persons_in_frame:
                persons_in_frame.sort(key=lambda p: p['distance'])
                target_person = persons_in_frame[0]
                if self.target_id != target_person['id']:
                    rospy.loginfo("锁定新目标, ID: {}".format(target_person['id']))
                    self.target_id = target_person['id']
                self.target_lost_frames = 0
            else:
                if self.target_id is not None:
                    rospy.loginfo("目标 {} 完全丢失。".format(self.target_id))
                    self.target_id = None
        return target_person

    def _process_person_for_following(self, person_point_base_link):
        person_point_map = self._transform_point(
            (person_point_base_link.point.x, person_point_base_link.point.y, person_point_base_link.point.z),
            person_point_base_link.header.frame_id, self.global_frame
        )
        if person_point_map is None: return

        if self.last_goal_position:
            dist = math.hypot(person_point_map.point.x - self.last_goal_position.x, 
                              person_point_map.point.y - self.last_goal_position.y)
            if dist < self.goal_update_distance:
                self._publish_person_marker(person_point_map.point)
                return

        goal_msg = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=self.global_frame))
        try:
            robot_transform = self.tf_buffer.lookup_transform(self.global_frame, self.robot_base_frame, rospy.Time(0), rospy.Duration(0.1))
            robot_pos = robot_transform.transform.translation
            
            angle_to_person = math.atan2(person_point_map.point.y - robot_pos.y, person_point_map.point.x - robot_pos.x)
            
            goal_msg.pose.position.x = person_point_map.point.x - self.follow_distance * math.cos(angle_to_person)
            goal_msg.pose.position.y = person_point_map.point.y - self.follow_distance * math.sin(angle_to_person)

            q = tf.transformations.quaternion_from_euler(0, 0, angle_to_person)
            goal_msg.pose.orientation.x, goal_msg.pose.orientation.y, goal_msg.pose.orientation.z, goal_msg.pose.orientation.w = q
        except Exception as e:
            rospy.logwarn("计算导航目标失败: {}".format(e))
            return

        self.goal_pub.publish(goal_msg)
        self.last_goal_position = goal_msg.pose.position
        rospy.loginfo("更新导航目标至: ({:.2f}, {:.2f})".format(goal_msg.pose.position.x, goal_msg.pose.position.y))
        self._publish_person_marker(person_point_map.point)

    # -------------------------------------------------------------------
    # 可视化方法
    # -------------------------------------------------------------------

    def _publish_person_marker(self, position):
        marker = Marker(
            header=Header(stamp=rospy.Time.now(), frame_id=self.global_frame),
            ns="person", id=0, type=Marker.CYLINDER, action=Marker.ADD,
            pose=PoseStamped(header=Header(), pose={'position': position, 'orientation': {'w': 1.0}}).pose,
            scale={'x': 0.4, 'y': 0.4, 'z': 1.5},
            color={'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0},
            lifetime=rospy.Duration(1.0)
        )
        self.marker_pub.publish(marker)

    def _update_visualization(self, color_image, depth_frame, persons_in_frame):
        display_image = color_image.copy()
        for person in persons_in_frame:
            x1, y1, x2, y2 = map(int, person['bbox'])
            color = (0, 0, 255) if person['id'] == self.target_id else (0, 255, 0)
            label = "TARGET {}".format(person['id']) if person['id'] == self.target_id else "ID {}".format(person['id'])
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Person Detection', display_image)
        if self.vis and self.frame_count % 5 == 0:
            depth_o3d = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
            color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0/self.depth_scale, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, o3d.camera.PinholeCameraIntrinsic(self.intrinsics))
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcd.points = pcd.points
            self.pcd.colors = pcd.colors
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("用户请求退出")


if __name__ == '__main__':
    try:
        follower_node = YoloPersonFollower()
        follower_node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("节点遇到未处理的异常: {}".format(e))