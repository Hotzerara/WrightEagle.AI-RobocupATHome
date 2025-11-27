# yolo_seg_infer_3d.py
from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from tf.transformations import quaternion_matrix

class YOLO3DDetector:
    def __init__(self, args):
        self.args = args
        
        # 初始化ROS节点
        try:
            rospy.init_node('yolo_3d_detector', anonymous=True)
        except:
            pass
        
        # 发布camera3D位置信息
        self.object_3d_pub = rospy.Publisher(
            '/object_3d_position', 
            PointStamped, 
            queue_size=10
        )
        
        # 发布apple 3D位置信息
        self.object_base_pub = rospy.Publisher(
            '/object/base_link_3d_position',
            PointStamped,
            queue_size=10
        )
        
        # 初始化RealSense
        self.pipeline, self.align, self.depth_scale, self.intrinsics = self.initialize_realsense()
        
        # 相机到gripper的变换矩阵
        self.transformation_matrix = np.array([
            [-0.02937859, -0.1152559 ,  0.99290129, -0.02411462],
            [ 0.01197522,  0.99321818,  0.11564701, -0.06956071],
            [-0.99949662,  0.01528775, -0.02779914,  0.01524878],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])
        
        # 加载YOLO模型
        self.model = YOLO(args.model)
        
        # 创建显示窗口
        cv2.namedWindow('YOLO Apple Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Apple Detection', 800, 600)

    def initialize_realsense(self, serial_number = "220422302842"):
        """初始化RealSense摄像头"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 启动流
        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"RealSense启动失败: {e}")
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
        
        print(f"RealSense初始化完成 - 深度比例: {depth_scale}")
        return pipeline, align, depth_scale, intrinsics

    def get_median_depth_in_roi(self, depth_frame, x, y, roi_size=20):
        """
        获取以指定点为中心的ROI区域内的中值深度
        """
        # 确保坐标在图像范围内
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # 计算ROI边界
        x1 = max(0, int(x - roi_size//2))
        y1 = max(0, int(y - roi_size//2))
        x2 = min(width - 1, int(x + roi_size//2))
        y2 = min(height - 1, int(y + roi_size//2))
        
        # 提取ROI内的深度数据
        depth_data = np.asanyarray(depth_frame.get_data())
        roi = depth_data[y1:y2, x1:x2]
        
        # 转换为实际深度值（米）
        roi_meters = roi.astype(float) * self.depth_scale
        
        # 过滤掉无效深度值（0表示无效）
        valid_depths = roi_meters[roi_meters > 0.1]  # 只考虑深度大于10cm的点
        valid_depths = valid_depths[valid_depths < 2.0] #过滤掉太远的点

        if len(valid_depths) == 0:
            return None
        
        # 计算有效深度的中值
        median_depth = np.median(valid_depths)
        return median_depth

    def get_3d_coordinates(self, depth_frame, pixel_x, pixel_y, depth_value=None):
        """
        将2D像素坐标转换为3D世界坐标
        """
        # 确保坐标在图像范围内
        if (pixel_x < 0 or pixel_y < 0 or 
            pixel_x >= self.intrinsics.width or 
            pixel_y >= self.intrinsics.height):
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
            point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth)
            return point  # (x, y, z) in meters
        except RuntimeError:
            return None

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

    
    # def transform_point_to_base_link(self, point_camera):
    #     """
    #     将相机坐标系下的点转换到base_link坐标系
    #     """
    #     # 创建TF缓冲区和监听器
    #     tf_buffer = tf2_ros.Buffer()
    #     tf_listener = tf2_ros.TransformListener(tf_buffer)
        
    #     try:
    #         # 获取从right_gripper_link到base_link的变换
    #         tf_buffer.can_transform("base_link", "left_gripper_link", rospy.Time.now(), rospy.Duration(1.0))
    #         gripper_to_base_transform = tf_buffer.lookup_transform("base_link", "left_gripper_link", rospy.Time(0))
            
    #         # 提取变换信息
    #         translation = np.array([
    #             gripper_to_base_transform.transform.translation.x,
    #             gripper_to_base_transform.transform.translation.y,
    #             gripper_to_base_transform.transform.translation.z
    #         ])
            
    #         rotation = np.array([
    #             gripper_to_base_transform.transform.rotation.x,
    #             gripper_to_base_transform.transform.rotation.y,
    #             gripper_to_base_transform.transform.rotation.z,
    #             gripper_to_base_transform.transform.rotation.w
    #         ])
            
    #         # 创建从right_gripper_link到base_link的变换矩阵
    #         rotation_matrix = quaternion_matrix(rotation)
    #         gripper_to_base_matrix = np.identity(4)
    #         gripper_to_base_matrix[:3, :3] = rotation_matrix[:3, :3]
    #         gripper_to_base_matrix[:3, 3] = translation
            
    #         # 组合变换：先应用相机到gripper的变换，再应用gripper到base的变换
    #         combined_matrix = np.dot(gripper_to_base_matrix, self.transformation_matrix)
            
    #         # 转换点到base_link坐标系
    #         point_base = self.transform_point_with_matrix(point_camera, combined_matrix)
            
    #         return point_base
            
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
    #             tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
    #         rospy.logerr(f"TF transformation failed: {e}")
    #         return None

    def transform_point_to_right_arm_base(self, point_camera):
        """
        将相机坐标系下的点转换到 right_arm_base_link 坐标系
        """
        # 创建TF缓冲区和监听器
        # 建议：为了性能，最好在 __init__ 中初始化 self.tf_buffer 和 self.tf_listener
        # 这里为了改动最小，保持局部变量
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        
        try:
            # ==================================================
            # 关键修改区域
            # ==================================================
            target_frame = "right_arm_base_link"  # 目标：右臂基座
            source_frame = "left_gripper_link"    # 源：左手抓手（假设相机固定在左手上）
            
            # 检查是否可以变换
            tf_buffer.can_transform(target_frame, source_frame, rospy.Time.now(), rospy.Duration(1.0))
            
            # 获取变换关系：从 left_gripper_link 到 right_arm_base_link
            # TF 会自动计算路径：left_gripper -> left_arm_base -> torso -> right_arm_base
            transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
            
            # 提取平移
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # 提取旋转
            rotation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # 构建矩阵 T_right_base_to_left_gripper
            rotation_matrix = quaternion_matrix(rotation)
            gripper_to_right_base_matrix = np.identity(4)
            gripper_to_right_base_matrix[:3, :3] = rotation_matrix[:3, :3]
            gripper_to_right_base_matrix[:3, 3] = translation
            
            # 组合矩阵： T_total = T_(right_base<-gripper) * T_(gripper<-camera)
            combined_matrix = np.dot(gripper_to_right_base_matrix, self.transformation_matrix)
            
            # 执行坐标变换
            point_in_right_base = self.transform_point_with_matrix(point_camera, combined_matrix)
            
            return point_in_right_base
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            rospy.logerr(f"TF transformation failed: {e}")
            return None



    def process_frame(self):
        """处理单帧图像"""
        try:
            # 等待下一组帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return False
                
            # 转换为OpenCV格式
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 使用YOLO进行分割推理 - 只检测苹果 (ID:47)
            results = self.model.predict(
                source=color_image,
                conf=self.args.conf,
                iou=self.args.iou,
                device=self.args.device,
                classes=[47],  # 只检测苹果 (ID:47)
                verbose=False
            )
            
            # 创建用于显示的图像副本
            display_image = color_image.copy()
            
            apple_detected = False
            
            # 处理检测结果
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = box.conf[0]
                    
                    # 只处理苹果 (ID:47)
                    if class_id != 47:
                        continue
                    
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    print(f"检测到苹果 - 2D中心坐标: ({center_x:.1f}, {center_y:.1f}), 置信度: {conf:.2f}")
                    
                    # 获取ROI区域的中值深度
                    median_depth = self.get_median_depth_in_roi(depth_frame, center_x, center_y)
                    
                    # 获取3D坐标
                    if median_depth is not None:
                        point_3d = self.get_3d_coordinates(
                            depth_frame, 
                            center_x, 
                            center_y,
                            depth_value=median_depth
                        )
                    else:
                        # 如果中值深度无效，尝试直接获取中心点深度
                        point_3d = self.get_3d_coordinates(depth_frame, center_x, center_y)
                    
                    # 绘制边界框和中心点
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 显示类别和置信度
                    label = f"Apple {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if point_3d is not None:
                        x, y, z = point_3d
                        
                        # 在图像上显示3D坐标
                        coord_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
                        cv2.putText(display_image, coord_text, (center_x - 70, center_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        print(f"苹果3D位置 - 相机坐标系: ({x:.2f}, {y:.2f}, {z:.2f})m")
                        
                        
                        point_camera = np.array([x, y, z])
                        # 转换到base_link坐标系
                        # point_base = self.transform_point_to_base_link(point_camera)

                        # 转换到right_arm_base坐标系
                        point_right_base = self.transform_point_to_right_arm_base(point_camera)

                        # if point_base is not None:
                        #     print(f"苹果3D位置 - base_link坐标系: ({point_base[0]:.2f}, {point_base[1]:.2f}, {point_base[2]:.2f})m")
                        if point_right_base is not None:
                            print(f"苹果3D位置 - 右臂基座坐标系: ({point_right_base[0]:.2f}, {point_right_base[1]:.2f}, {point_right_base[2]:.2f})m")
                            
                            # 发布3D位置信息
                            # 相机坐标系
                            camera_point_msg = PointStamped()
                            camera_point_msg.header = Header()
                            camera_point_msg.header.stamp = rospy.Time.now()
                            camera_point_msg.header.frame_id = "camera_color_optical_frame"
                            camera_point_msg.point = Point(x, y, z)
                            self.object_3d_pub.publish(camera_point_msg)
                            
                            # base_link坐标系
                            # base_point_msg = PointStamped()
                            # base_point_msg.header = Header()
                            # base_point_msg.header.stamp = rospy.Time.now()
                            # base_point_msg.header.frame_id = "base_link"
                            # base_point_msg.point = Point(point_base[0], point_base[1], point_base[2])
                            # self.object_base_pub.publish(base_point_msg)

                            # 2. 发布右臂基座坐标系下的点 (修改这里)
                            base_point_msg = PointStamped()
                            base_point_msg.header = Header()
                            base_point_msg.header.stamp = rospy.Time.now()
                            base_point_msg.header.frame_id = "right_arm_base_link"
                            base_point_msg.point = Point(point_right_base[0], point_right_base[1], point_right_base[2])
                            self.object_base_pub.publish(base_point_msg)
                    
                    apple_detected = True
                    break  # 只处理第一个检测到的苹果
                
                if apple_detected:
                    break
            
            # 如果没有检测到苹果，显示提示信息
            if not apple_detected:
                cv2.putText(display_image, "No Apple Detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("未检测到苹果")
            
            # 显示图像
            cv2.imshow('YOLO Apple Detection', display_image)
            
            return True
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return False

    def run_realtime(self):
        """实时运行模式"""
        print("开始实时苹果3D检测 (按 'q' 键退出)...")
        print("只检测苹果 (ID: 47)")
        
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                # 处理当前帧
                success = self.process_frame()
                
                if not success:
                    continue
                
                # 计算并显示FPS
                fps = 1.0 / (time.time() - start_time + 1e-9)
                print(f"FPS: {fps:.1f}")
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户请求退出...")
                    break
                    
        except Exception as e:
            print(f"运行过程中出错: {e}")
        finally:
            # 清理资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")

def main(args):
    # 创建3D检测器
    detector = YOLO3DDetector(args)
    
    # 运行实时检测
    detector.run_realtime()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation with Apple 3D Position Detection")
    parser.add_argument("--model", type=str, default="yolo11x-seg.pt", help="Path to YOLOv8 segmentation model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda', 'cuda:0', or 'cpu'")

    args = parser.parse_args()
    main(args)