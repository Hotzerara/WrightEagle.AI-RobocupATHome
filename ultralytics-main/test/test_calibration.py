#!/usr/bin/env python

import numpy as np
import cv2
import pyrealsense2 as rs
import apriltag
import time
import rospy
import tf2_ros
from tf.transformations import quaternion_matrix

# 您的标定矩阵
transformation_matrix = np.array([
    [-0.0166314,  -0.04629356,  0.99878942,  0.01306665],
    [ 0.02129678,  0.99868456,  0.04664332, -0.04064238],
    [-0.99963486,  0.02204674, -0.01562362, -0.01117981],
    [ 0.,          0.,          0.,          1.        ]
])

def transform_point_with_matrix(point, transformation_matrix):
    """使用变换矩阵转换点"""
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    transformed_point = np.dot(transformation_matrix, point_homogeneous)
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
        
        # 创建从left_gripper_link到base_link的变换矩阵
        rotation_matrix = quaternion_matrix(rotation)
        gripper_to_base_matrix = np.identity(4)
        gripper_to_base_matrix[:3, :3] = rotation_matrix[:3, :3]
        gripper_to_base_matrix[:3, 3] = translation

        # 组合变换：先应用相机到gripper的变换，再应用gripper到base的变换
        combined_matrix = np.dot(gripper_to_base_matrix, transformation_matrix)
        
        # 转换点到base_link坐标系
        point_base = transform_point_with_matrix(point_camera, combined_matrix)
        print("*****")
        print(point_base)
        
        # [0.50017165 0.74310192 0.83441462]

        return point_base
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
        print(f"TF transformation failed: {e}")
        return None

class StandaloneAprilTagDetector:
    def __init__(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix
        self.detector = apriltag.Detector()
        
        # 初始化RealSense相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 开始流
        self.profile = self.pipeline.start(self.config)
        
        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_params = [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        print("=" * 50)
        print("相机内参:")
        print(f"fx: {intrinsics.fx:.2f}, fy: {intrinsics.fy:.2f}")
        print(f"cx: {intrinsics.ppx:.2f}, cy: {intrinsics.ppy:.2f}")
        print("=" * 50)
        
        # 用于验证的已知tag位置（您需要根据实际情况设置）
        self.expected_tag_positions = {
            0: np.array([0.5, 0.0, 0.3]),   # 示例：tag 0的预期位置
            1: np.array([0.5, 0.2, 0.3]),   # 示例：tag 1的预期位置
            2: np.array([0.5, -0.2, 0.3]),  # 示例：tag 2的预期位置
        }
        
        self.detection_count = 0

    def detect_apriltags(self, image):
        """检测图像中的AprilTag"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return detections
    
    def estimate_tag_pose(self, detection, tag_size=0.1):
        """估计AprilTag的3D位姿"""
        # 定义tag在3D空间中的角点（tag坐标系）
        obj_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ])
        
        # 检测到的2D角点
        img_points = detection.corners.astype(np.float32)
        
        # 畸变系数（假设没有畸变）
        dist_coeffs = np.zeros((4, 1))
        
        # 使用solvePnP计算位姿
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, dist_coeffs)
        
        if success:
            # tag在相机坐标系中的位置
            tag_position_camera = tvec.flatten()
            return tag_position_camera, True
        else:
            return None, False
    
    def validate_transformation(self, tag_id, calculated_position):
        """验证坐标变换的准确性"""
        if tag_id in self.expected_tag_positions:
            expected_pos = self.expected_tag_positions[tag_id]
            error = np.linalg.norm(calculated_position - expected_pos)
            
            print(f"验证结果 - Tag {tag_id}:")
            print(f"  预期位置: [{expected_pos[0]:.3f}, {expected_pos[1]:.3f}, {expected_pos[2]:.3f}]")
            print(f"  计算位置: [{calculated_position[0]:.3f}, {calculated_position[1]:.3f}, {calculated_position[2]:.3f}]")
            print(f"  误差: {error:.4f} 米")
            
            if error < 0.05:  # 5cm误差阈值
                print(f"  ✅ 变换准确！误差在可接受范围内")
            else:
                print(f"  ⚠️  误差较大，可能需要重新标定")
            
            return error
        else:
            print(f"Tag {tag_id} 没有预设的预期位置，无法验证准确性")
            return None
    
    def draw_detection(self, image, detection, position_camera, position_base, tag_id, error=None):
        """在图像上绘制检测结果和信息"""
        # 绘制边界框
        corners = detection.corners.astype(int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
        
        # 绘制中心点
        center = detection.center.astype(int)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)
        
        # 添加标签文本
        text1 = "Tag {} - Camera: ({:.3f}, {:.3f}, {:.3f})".format(
            tag_id, position_camera[0], position_camera[1], position_camera[2]
        )
        text2 = "Base: ({:.3f}, {:.3f}, {:.3f})".format(
            position_base[0], position_base[1], position_base[2]
        )
        
        # 根据误差显示不同颜色
        color1 = (0, 255, 255)  # 黄色
        color2 = (255, 255, 0)  # 青色
        
        if error is not None:
            if error < 0.05:
                color2 = (0, 255, 0)  # 绿色
            else:
                color2 = (0, 0, 255)  # 红色
            
            text3 = "Error: {:.3f}m".format(error)
            cv2.putText(image, text3, (10, 90 + tag_id * 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
        
        cv2.putText(image, text1, (10, 30 + tag_id * 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 2)
        cv2.putText(image, text2, (10, 60 + tag_id * 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
        
        # 在tag旁边显示ID
        cv2.putText(image, f"ID: {tag_id}", tuple(center + np.array([10, -10])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def run(self):
        """主运行循环"""
        print("开始AprilTag检测和坐标变换验证...")
        print("按 'q' 键退出")
        print("按 's' 键保存当前图像")
        print("按 'r' 键重置检测计数")
        print("=" * 50)
        
        try:
            while True:
                # 等待一帧
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 检测AprilTag
                detections = self.detect_apriltags(color_image)
                
                if len(detections) > 0:
                    self.detection_count += 1
                    print(f"\n=== 检测 #{self.detection_count} ===")
                    print(f"检测到 {len(detections)} 个AprilTag")
                    
                    for detection in detections:
                        tag_id = detection.tag_id
                        
                        # 估计tag位姿
                        position_camera, success = self.estimate_tag_pose(detection)
                        
                        if success:
                            print(f"\nTag {tag_id} 检测结果:")
                            print(f"  相机坐标系: [{position_camera[0]:.3f}, {position_camera[1]:.3f}, {position_camera[2]:.3f}] m")
                            
                            # 使用您的transform_point_to_base_link函数进行坐标变换
                            position_base = transform_point_to_base_link(position_camera, self.transformation_matrix)
                            
                            if position_base is not None:
                                print(f"  base_link坐标系: [{position_base[0]:.3f}, {position_base[1]:.3f}, {position_base[2]:.3f}] m")
                                
                                # 验证变换准确性
                                error = self.validate_transformation(tag_id, position_base)
                                
                                # 在图像上绘制
                                self.draw_detection(color_image, detection, position_camera, position_base, tag_id, error)
                            else:
                                print(f"  ❌ 坐标变换失败")
                                self.draw_detection(color_image, detection, position_camera, np.array([0, 0, 0]), tag_id)
                        else:
                            print(f"Tag {tag_id} 位姿估计失败")
                
                # 显示图像
                cv2.imshow('AprilTag Detection - 坐标变换验证', color_image)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前图像
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"apriltag_validation_{timestamp}.jpg"
                    cv2.imwrite(filename, color_image)
                    print(f"图像已保存: {filename}")
                elif key == ord('r'):
                    self.detection_count = 0
                    print("检测计数已重置")
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("相机已关闭")
            print(f"总检测次数: {self.detection_count}")

def main():
    # 创建检测器实例
    detector = StandaloneAprilTagDetector(transformation_matrix)
    
    # 运行检测
    detector.run()

if __name__ == '__main__':
    main()