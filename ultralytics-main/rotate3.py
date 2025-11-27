#!/usr/bin/env python
# -*- coding: utf-8 -*-



#super original use this
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TwistStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply
import math

class IntegratedPersonFollower:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('integrated_person_follower')
        
        # === ROS Parameters ===
        self.goal_update_distance = rospy.get_param('~goal_update_distance', 1.2)
        self.follow_distance = rospy.get_param('~follow_distance', 0.8)
        self.global_frame = rospy.get_param('~global_frame', 'map')
        
        # === Tracking and Search Parameters ===
        self.frame_center_x = 640 // 2
        self.last_person_x = None
        self.searching = False
        self.search_start_time = None
        self.max_search_time = 8.0
        self.last_goal_position = None
        self.person_detected = False
        
        # === TF2 Setup ===
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # === ROS Publishers ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.vel_pub = rospy.Publisher('/motion_target/target_speed_chassis', TwistStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/person_marker', Marker, queue_size=1)
        self.person_pos_pub = rospy.Publisher('/person/base_link_position', PointStamped, queue_size=10)
        
        # === Initialize RealSense ===
        self.pipeline, self.align, self.depth_scale, self.intrinsics = self.initialize_realsense()
        
        # === Load YOLO Model ===
        self.model = YOLO('yolov8n-pose.pt')
        
        # === Camera to Gripper Transformation Matrix ===
        self.transformation_matrix = np.array([
            [ 0.01880272, -0.10014192,  0.99479548, -0.06809926],
            [ 0.08711035,  0.99135191,  0.09814878, -0.07856771],
            [-0.99602121, 0.08481152,  0.02736351, -0.00702536],
            [ 0.0,         0.0,          0.0,          1.0       ]
        ])
        
        rospy.loginfo("Integrated Person Follower started! Waiting for TF transforms...")
        rospy.sleep(2)  # Wait for TF tree to stabilize

    def initialize_realsense(self):
        """Initialize RealSense camera"""
        pipeline = rfs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            rospy.logerr(f"RealSense startup failed: {e}")
            config.disable_all_streams()
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)
            profile = pipeline.start(config)
        
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        
        return pipeline, align, depth_scale, intrinsics

    def get_median_depth_in_roi(self, depth_frame, x1, y1, x2, y2):
        """Get median depth in region of interest"""
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width - 1, int(x2))
        y2 = min(height - 1, int(y2))
        
        depth_data = np.asanyarray(depth_frame.get_data())
        roi = depth_data[y1:y2, x1:x2]
        roi_meters = roi.astype(float) * self.depth_scale
        valid_depths = roi_meters[roi_meters > 0.1]
        
        if len(valid_depths) == 0:
            return None
        return np.median(valid_depths)

    def get_3d_coordinates(self, depth_frame, pixel_x, pixel_y, depth_value=None):
        """Convert 2D pixel to 3D coordinates"""
        if (pixel_x < 0 or pixel_y < 0 or 
            pixel_x >= self.intrinsics.width or 
            pixel_y >= self.intrinsics.height):
            return None
        
        try:
            if depth_value is None:
                depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
            else:
                depth = depth_value
                
            if depth <= 0:
                return None
                
            point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth)
            return point
        except RuntimeError:
            return None

    def get_body_center_from_keypoints(self, keypoints):
        """Calculate body center from keypoints"""
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        
        valid_points = []
        
        if keypoints[LEFT_SHOULDER][2] > 0.1 and keypoints[RIGHT_SHOULDER][2] > 0.1:
            shoulder_center = (
                (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2,
                (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
            )
            valid_points.append(shoulder_center)
        
        if keypoints[LEFT_HIP][2] > 0.1 and keypoints[RIGHT_HIP][2] > 0.1:
            hip_center = (
                (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
                (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
            )
            valid_points.append(hip_center)
        
        if not valid_points:
            return None
        
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return (center_x, center_y)

    def transform_point_with_matrix(self, point, transform_matrix):
        """
        使用4x4变换矩阵转换点
        """
        # 将点转换为齐次坐标
        point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        
        # 应用变换矩阵
        transformed_point = np.dot(transform_matrix, point_homogeneous)
        
        # 返回非齐次坐标
        return transformed_point[:3]

    def transform_point_to_base_link(self, point_camera):
        """
        将相机坐标系下的点转换到base_link坐标系
        使用: Camera → Left Gripper → Base Link 变换链
        """
        try:
            # 获取从left_gripper_link到base_link的变换
            if not self.tf_buffer.can_transform("base_link", "left_gripper_link", rospy.Time.now(), rospy.Duration(1.0)):
                rospy.logwarn("Cannot transform from base_link to left_gripper_link")
                return None
                
            gripper_to_base_transform = self.tf_buffer.lookup_transform("base_link", "left_gripper_link", rospy.Time(0))
            
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
            
            # 创建从left_gripper_link到base_link的变换矩阵
            rotation_matrix = quaternion_matrix(rotation)
            gripper_to_base_matrix = np.identity(4)
            gripper_to_base_matrix[:3, :3] = rotation_matrix[:3, :3]
            gripper_to_base_matrix[:3, 3] = translation
            
            # 组合变换：先应用相机到gripper的变换，再应用gripper到base的变换
            combined_matrix = np.dot(gripper_to_base_matrix, self.transformation_matrix)
            
            # 转换点到base_link坐标系
            point_base = self.transform_point_with_matrix(point_camera, combined_matrix)
            
            return point_base
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            rospy.logwarn("TF transformation failed: %s", e)
            # Fallback: use simple transformation
            return [point_camera[2], -point_camera[0], -point_camera[1]]

    def publish_navigation_goal(self, person_point_base):
        """Publish navigation goal to follow the person"""
        try:
            # Transform base_link point to map frame
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "base_link"
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.point = Point(person_point_base[0], person_point_base[1], 0)  # Use 2D for navigation
            
            transform = self.tf_buffer.lookup_transform(self.global_frame, "base_link", rospy.Time(0))
            person_point_map = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            # Get robot position in map frame
            robot_point = PointStamped()
            robot_point.header.frame_id = "base_link"
            robot_point.header.stamp = rospy.Time.now()
            robot_point.point = Point(0, 0, 0)
            robot_point_map = tf2_geometry_msgs.do_transform_point(robot_point, transform)
            
            # Calculate direction from robot to person
            dx = person_point_map.point.x - robot_point_map.point.x
            dy = person_point_map.point.y - robot_point_map.point.y
            distance_to_person = math.sqrt(dx*dx + dy*dy)
            angle_to_person = math.atan2(dy, dx)
            
            # Calculate goal position (stop at follow_distance from person)
            if distance_to_person > self.follow_distance:
                # Move toward person but stop at follow_distance
                goal_distance = distance_to_person - self.follow_distance
                goal_x = robot_point_map.point.x + goal_distance * math.cos(angle_to_person)
                goal_y = robot_point_map.point.y + goal_distance * math.sin(angle_to_person)
            else:
                # Person is too close, don't move closer
                goal_x = robot_point_map.point.x
                goal_y = robot_point_map.point.y
            
            # Check if we need to update goal
            if self.last_goal_position:
                dist = math.sqrt((goal_x - self.last_goal_position[0])**2 +
                               (goal_y - self.last_goal_position[1])**2)
                if dist < self.goal_update_distance:
                    return  # Skip update
            
            # Create and publish goal
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = self.global_frame
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.pose.position.x = goal_x
            goal_msg.pose.position.y = goal_y
            
            # Orient robot toward person
            q = tf.transformations.quaternion_from_euler(0, 0, angle_to_person)
            goal_msg.pose.orientation.x = q[0]
            goal_msg.pose.orientation.y = q[1]
            goal_msg.pose.orientation.z = q[2]
            goal_msg.pose.orientation.w = q[3]
            
            self.goal_pub.publish(goal_msg)
            self.last_goal_position = (goal_x, goal_y)
            
            rospy.loginfo("Following: Person at (%.2f, %.2f), Goal at (%.2f, %.2f), Distance: %.2fm", 
                         person_point_base[0], person_point_base[1], goal_x, goal_y, distance_to_person)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Navigation goal creation failed: %s", e)

    def rotate_in_place(self, direction='left'):
        """Rotate robot to search for person"""
        twist = TwistStamped()
        twist.header.stamp = rospy.Time.now()
        twist.header.frame_id = 'base_link'
        twist.twist.angular.z = 0.3 if direction == 'left' else -0.3
        self.vel_pub.publish(twist)

    def stop_rotation(self):
        """Stop robot rotation"""
        twist = TwistStamped()
        twist.header.stamp = rospy.Time.now()
        twist.header.frame_id = 'base_link'
        twist.twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def publish_person_marker(self, position):
        """Publish RViz marker for visualization"""
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "person"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(0.5)
        self.marker_pub.publish(marker)

    def run(self):
        """Main detection and tracking loop"""
        rospy.loginfo("Starting person detection and tracking...")
        
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                # Get camera frames
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                display_image = color_image.copy()
                
                # Run YOLO detection
                results = self.model(color_image)
                
                found_person = False
                person_3d_base = None
                
                for result in results:
                    if result.keypoints is None:
                        continue
                    
                    boxes, keypoints = result.boxes, result.keypoints
                    
                    for i, box in enumerate(boxes):
                        if int(box.cls[0]) != 0:  # Person class
                            continue
                            
                        found_person = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Get keypoints and body center
                        kpts = keypoints.xy[i].cpu().numpy()
                        confs = keypoints.conf[i].cpu().numpy()
                        key_data = [(kpts[j][0], kpts[j][1], confs[j]) for j in range(len(kpts))]
                        center = self.get_body_center_from_keypoints(key_data)
                        cx, cy = ((x1 + x2) // 2, (y1 + y2) // 2) if center is None else map(int, center)
                        self.last_person_x = cx
                        
                        # Get 3D coordinates
                        median_depth = self.get_median_depth_in_roi(depth_frame, x1, y1, x2, y2)
                        point_3d_camera = self.get_3d_coordinates(depth_frame, cx, cy, median_depth)
                        
                        if point_3d_camera is not None:
                            # Transform to base_link frame
                            person_3d_base = self.transform_point_to_base_link(point_3d_camera)
                            
                            if person_3d_base is not None:
                                # Publish person position for debugging
                                pos_msg = PointStamped()
                                pos_msg.header.frame_id = "base_link"
                                pos_msg.header.stamp = rospy.Time.now()
                                pos_msg.point = Point(person_3d_base[0], person_3d_base[1], person_3d_base[2])
                                self.person_pos_pub.publish(pos_msg)
                                
                                # Publish navigation goal to follow person
                                self.publish_navigation_goal(person_3d_base)
                                
                                # Visualize in RViz
                                try:
                                    point_stamped = PointStamped()
                                    point_stamped.header.frame_id = "base_link"
                                    point_stamped.header.stamp = rospy.Time.now()
                                    point_stamped.point = Point(person_3d_base[0], person_3d_base[1], 0)
                                    transform = self.tf_buffer.lookup_transform(self.global_frame, "base_link", rospy.Time(0))
                                    person_point_map = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                                    self.publish_person_marker(person_point_map.point)
                                except Exception as e:
                                    rospy.logwarn("Marker publishing failed: %s", e)
                        
                        # Draw detection on display
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(display_image, (cx, cy), 4, (0, 0, 255), -1)
                        if point_3d_camera is not None:
                            cv2.putText(display_image, f"({point_3d_camera[2]:.2f}m)", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # === Search Behavior when Person is Lost ===
                if not found_person:
                    if self.last_person_x is not None:
                        # Determine rotation direction based on last seen position
                        if self.last_person_x < self.frame_center_x - 60:
                            self.rotate_in_place('left')
                            if not self.searching:
                                self.search_start_time = time.time()
                                self.searching = True
                            status = "SEARCHING LEFT"
                        elif self.last_person_x > self.frame_center_x + 60:
                            self.rotate_in_place('right') 
                            if not self.searching:
                                self.search_start_time = time.time()
                                self.searching = True
                            status = "SEARCHING RIGHT"
                        else:
                            self.stop_rotation()
                            self.searching = False
                            status = "PERSON LOST"
                        
                        # Stop searching after timeout
                        if self.searching and time.time() - self.search_start_time > self.max_search_time:
                            rospy.loginfo("Search timeout - stopping rotation")
                            self.stop_rotation()
                            self.searching = False
                            status = "SEARCH TIMEOUT"
                    else:
                        self.stop_rotation()
                        self.searching = False
                        status = "NO PERSON"
                else:
                    # Person found - stop any ongoing rotation
                    if self.searching:
                        self.stop_rotation()
                        self.searching = False
                        rospy.loginfo("Person found - stopping search rotation")
                    status = "FOLLOWING"
                
                # Display FPS and status
                fps = 1.0 / (time.time() - start_time + 1e-9)
                cv2.putText(display_image, f"FPS: {fps:.1f} | {status}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Person Detection and Following", display_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.stop_rotation()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        rospy.loginfo("Integrated Person Follower shutdown complete")

if __name__ == '__main__':
    try:
        follower = IntegratedPersonFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")