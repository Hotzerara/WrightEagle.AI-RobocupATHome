#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
这是一个集成了YOLOv8姿态检测、RealSense 3D定位与ROS导航跟随功能的统一节点。
它将两个独立脚本的功能合并在一起，通过内部函数调用传递数据，提高了效率。
并且加入了人员追踪逻辑，使跟随更稳定、更鲁棒。

功能流程：
1. 使用RealSense相机获取彩色图和深度图。
2. 在彩色图上运行YOLOv8-pose模型，追踪并检测人体及其关键点。
3. 锁定一个目标人员（如距离最近的），并持续追踪其ID。
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
from actionlib_msgs.msg import GoalID # 用于取消move_base目标

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
        self.goal_update_distance = rospy.get_param('~goal_update_distance', 0.5)
        self.follow_distance = rospy.get_param('~follow_distance', 1.0)
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.yolo_model_path = rospy.get_param('~yolo_model', 'yolov8n-pose.pt')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.enable_visualization = rospy.get_param('~enable_visualization', False)
        self.min_keypoint_confidence = rospy.get_param('~min_keypoint_confidence', 0.2)
        self.min_detection_confidence = rospy.get_param('~min_detection_confidence', 0.5)
        self.target_lost_threshold = rospy.get_param('~target_lost_threshold', 30) # 目标丢失超过30帧后重新搜索

        # === 3. 初始化TF2 (只执行一次) ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === 4. 初始化ROS发布者 ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/person_marker', Marker, queue_size=1)
        self.cancel_goal_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1) # 新增：用于取消目标

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

        # === 7. 初始化状态变量 ===
        self.last_goal_position = None
        self.target_id = None
        self.target_lost_frames = 0
        self.frame_count = 0

        rospy.loginfo("节点初始化完成，准备进入主循环。")

    # -------------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------------
    
    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            color_frame, depth_frame, color_image = self._get_frames()
            if color_frame is None:
                rate.sleep()
                continue
            
            results = self.model.track(color_image, persist=True, verbose=False)
            
            persons_in_frame = self._process_yolo_results(results, depth_frame)
            target_person = self._update_tracking(persons_in_frame)

            if target_person:
                point_base_link = self._transform_point(
                    target_person['point_camera'], self.camera_frame, self.robot_base_frame
                )
                if point_base_link:
                    self._process_person_for_following(point_base_link)
            else:
                # 如果没有目标，则取消当前的move_base目标，让机器人停下来
                if self.last_goal_position is not None:
                    self._cancel_current_goal()
                    rospy.loginfo("目标丢失，取消导航任务。")
                    self.last_goal_position = None

            if self.enable_visualization:
                self.frame_count += 1
                self._update_visualization(color_image, depth_frame, persons_in_frame)

            rate.sleep()
            
        self._cleanup()

    # -------------------------------------------------------------------
    # 核心逻辑方法 (感知、追踪、跟随)
    # -------------------------------------------------------------------
    
    def _get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        return color_frame, depth_frame, np.asanyarray(color_frame.get_data())

    def _process_yolo_results(self, results, depth_frame):
        persons = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes, keypoints = results[0].boxes, results[0].keypoints
            for i in range(len(boxes)):
                if int(boxes.cls[i]) == 0 and boxes.conf[i] > self.min_detection_confidence:
                    bbox = boxes.xyxy[i].cpu().numpy()
                    center_2d = self._get_body_center(keypoints.xy[i].cpu().numpy(), keypoints.conf[i].cpu().numpy(), bbox)
                    if center_2d is None: continue

                    median_depth = self._get_median_depth(depth_frame, bbox)
                    if median_depth is None: continue

                    point_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_2d[0], center_2d[1]], median_depth)
                    
                    persons.append({
                        'id': int(boxes.id[i]), 'point_camera': point_camera,
                        'bbox': bbox, 'center_2d': center_2d, 'distance': median_depth
                    })
        return persons

    def _update_tracking(self, persons_in_frame):
        target_person = None
        if self.target_id is not None:
            # 寻找已锁定的目标
            for p in persons_in_frame:
                if p['id'] == self.target_id:
                    target_person, self.target_lost_frames = p, 0
                    break
            if target_person is None: self.target_lost_frames += 1
        
        # 如果没有目标，或目标丢失太久
        if self.target_id is None or self.target_lost_frames > self.target_lost_threshold:
            if persons_in_frame:
                # 寻找最近的人作为新目标
                persons_in_frame.sort(key=lambda p: p['distance'])
                target_person = persons_in_frame[0]
                if self.target_id != target_person['id']:
                    rospy.loginfo("锁定新目标, ID: {}".format(target_person['id']))
                    self.target_id = target_person['id']
                self.target_lost_frames = 0
            else:
                # 画面中无人，且目标已丢失
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
                self._publish_person_marker(person_point_map.point) # 仍然更新marker
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

    def _cancel_current_goal(self):
        """发布一个空的GoalID来取消move_base的当前目标。"""
        cancel_msg = GoalID()
        self.cancel_goal_pub.publish(cancel_msg)

    # -------------------------------------------------------------------
    # 辅助与工具方法
    # -------------------------------------------------------------------

    def _get_body_center(self, kpts_xy, kpts_conf, bbox):
        # (此处的具体实现与您的原代码相同)
        valid_points = []
        if kpts_conf[5] > self.min_keypoint_confidence and kpts_conf[6] > self.min_keypoint_confidence:
            valid_points.append(((kpts_xy[5][0] + kpts_xy[6][0]) / 2, (kpts_xy[5][1] + kpts_xy[6][1]) / 2))
        if kpts_conf[11] > self.min_keypoint_confidence and kpts_conf[12] > self.min_keypoint_confidence:
            valid_points.append(((kpts_xy[11][0] + kpts_xy[12][0]) / 2, (kpts_xy[11][1] + kpts_xy[12][1]) / 2))
        if not valid_points: return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        return (int(sum(p[0] for p in valid_points) / len(valid_points)), int(sum(p[1] for p in valid_points) / len(valid_points)))

    def _get_median_depth(self, depth_frame, bbox):
        # (此处的具体实现与您的原代码相同)
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_frame.get_width() - 1, x2), min(depth_frame.get_height() - 1, y2)
        if x1 >= x2 or y1 >= y2: return None
        roi_depth = np.asanyarray(depth_frame.get_data())[y1:y2, x1:x2].astype(float) * self.depth_scale
        valid_depths = roi_depth[roi_depth > 0.1]
        return np.median(valid_depths) if len(valid_depths) > 0 else None

    def _transform_point(self, point_tuple, from_frame, to_frame):
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(0.2))
            point_stamped = PointStamped(header=Header(stamp=rospy.Time.now(), frame_id=from_frame), point=Point(*point_tuple))
            return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        except Exception as e:
            rospy.logwarn_throttle(5, "坐标变换从 {} 到 {} 失败: {}".format(from_frame, to_frame, e))
            return None

    def _publish_person_marker(self, position):
        marker = Marker(header=Header(stamp=rospy.Time.now(), frame_id=self.global_frame),
                        ns="person", id=0, type=Marker.CYLINDER, action=Marker.ADD,
                        pose=PoseStamped(header=Header(), pose={'position': position, 'orientation': {'w': 1.0}}).pose,
                        scale={'x': 0.4, 'y': 0.4, 'z': 1.5},
                        color={'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0},
                        lifetime=rospy.Duration(1.0))
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
            # (此处省略Open3D点云更新代码以保持简洁)
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("用户请求退出")

    # (此处省略 _initialize_realsense, _initialize_visualization, _cleanup 的具体实现)
    def _initialize_realsense(self, serial_number = 220422302842):
        # ... (和之前版本一样)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
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
        # ... (和之前版本一样)
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
        # ... (和之前版本一样)
        self.pipeline.stop()
        if self.vis:
            self.vis.destroy_window()
        cv2.destroyAllWindows()
        rospy.loginfo("节点已关闭。")


if __name__ == '__main__':
    try:
        follower_node = YoloPersonFollower()
        follower_node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("节点遇到未处理的异常: {}".format(e))