import cv2
import numpy as np
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import time
import pyrealsense2 as rs

import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from tf.transformations import quaternion_matrix

class YOLOPlacementDetector:
    def __init__(self, target_item="orange", exclude_classes=None, conf_threshold=0.25, iou_threshold=0.5):
        # 初始化ROS节点
        try:
            rospy.init_node('yolo_placement_detector', anonymous=True)
        except:
            pass
        
        # 发布放置点位置信息
        self.placement_3d_pub = rospy.Publisher(
            '/placement_3d_position', 
            PointStamped, 
            queue_size=10
        )
        
        # 发布机器人坐标系下的放置点位置
        self.placement_base_pub = rospy.Publisher(
            '/placement/base_link_3d_position',
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
        self.model = YOLO("yolo11x-seg.pt")
        
        # 初始化SBERT模型
        self.sbert_model = SentenceTransformer("./all-MiniLM-L6-v2-local")
        self.embedding_cache = {}
        
        # 检测参数
        self.target_item = target_item
        self.exclude_classes = exclude_classes if exclude_classes else ["potted plant"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 创建显示窗口
        cv2.namedWindow('YOLO Placement Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Placement Detection', 800, 600)

    def initialize_realsense(self, serial_number="220422302842"):
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

    def text_similarity(self, a, b):
        """计算文本相似度"""
        if a not in self.embedding_cache:
            self.embedding_cache[a] = self.sbert_model.encode(a, convert_to_numpy=True)
        if b not in self.embedding_cache:
            self.embedding_cache[b] = self.sbert_model.encode(b, convert_to_numpy=True)
        emb_a = self.embedding_cache[a]
        emb_b = self.embedding_cache[b]
        sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a)*np.linalg.norm(emb_b)))
        return max(0.0, min(1.0, sim))

    def extract_objects(self, result):
        """提取物体信息"""
        objs = []
        names = result.names
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            name = names[cls_id] if isinstance(names, (list, dict)) else str(cls_id)
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            objs.append({
                "name": name,
                "xmin": int(x1),
                "xmax": int(x2),
                "center": (cx, cy),
                "y": cy,
                "width": x2 - x1
            })
        return objs

    def assign_layers(self, objects, eps=100):
        """分层处理"""
        sorted_objs = sorted(objects, key=lambda o: o["y"])
        layers = []
        layer_id = 0
        if not sorted_objs:
            return []
        layers.append(layer_id)
        last_y = sorted_objs[0]["y"]
        for obj in sorted_objs[1:]:
            if abs(obj["y"] - last_y) > eps:
                layer_id += 1
                last_y = obj["y"]
            layers.append(layer_id)
        for i, obj in enumerate(sorted_objs):
            obj["layer"] = layers[i]
        return sorted_objs

    def estimate_width(self, objects_on_layer, target_name):
        """估计目标宽度"""
        sims = [self.text_similarity(o["name"], target_name) for o in objects_on_layer]
        similar_objs = [o for o, s in zip(objects_on_layer, sims) if s > 0.5]
        if similar_objs:
            widths = [o["width"] for o in similar_objs]
            return np.median(widths)
        widths = [o["width"] for o in objects_on_layer]
        return np.median(widths) if widths else 100

    def find_best_gap(self, objects_on_layer, target_name, img_width=None):
        """寻找最佳放置间隙"""
        if not objects_on_layer:
            return None
        objs = sorted(objects_on_layer, key=lambda o: o["center"][0])
        target_width = self.estimate_width(objects_on_layer, target_name)

        # 1. 尝试放在两个物体中间
        gaps = []
        for i in range(len(objs) - 1):
            left, right = objs[i], objs[i + 1]
            gap_width = right["xmin"] - left["xmax"]
            if gap_width >= target_width:
                cx = int((left["xmax"] + right["xmin"]) / 2)
                cy = int((left["center"][1] + right["center"][1]) / 2)
                gaps.append((gap_width, cx, cy))
        if gaps:
            gaps.sort(reverse=True)
            _, cx, cy = gaps[0]
            return cx, cy

        # 2. 没有 gap 时，放最左或最右
        leftmost = objs[0]
        rightmost = objs[-1]
        cy = int(np.median([o["center"][1] for o in objs]))

        if img_width:
            left_space = leftmost["xmin"]
            right_space = img_width - rightmost["xmax"]

            if left_space >= target_width:
                cx = int(leftmost["xmin"] - target_width / 2)
                return cx, cy
            elif right_space >= target_width:
                cx = int(rightmost["xmax"] + target_width / 2)
                return cx, cy
        else:
            cx = int(rightmost["xmax"] + target_width / 2)
            return cx, cy

        cx = int(leftmost["xmin"] - target_width / 2)
        return cx, cy

    def compute_placement(self, result, img_width=None):
        """计算放置点"""
        objects = self.extract_objects(result)
        objects = [o for o in objects if o["name"] not in self.exclude_classes]
        if not objects:
            return None, None

        objects = self.assign_layers(objects, eps=100)

        # 选择层：语义相似度加权
        layer_scores = {}
        for o in objects:
            sim = self.text_similarity(o["name"], self.target_item)
            layer_scores[o["layer"]] = layer_scores.get(o["layer"], 0) + sim
        if not layer_scores:
            return None, None
        best_layer = max(layer_scores.items(), key=lambda x: x[1])[0]

        layer_objs = [o for o in objects if o["layer"] == best_layer]

        return self.find_best_gap(layer_objs, self.target_item, img_width), best_layer

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

    def transform_point_to_right_arm_base(self, point_camera):
        """
        将相机坐标系下的点转换到 right_arm_base_link 坐标系
        """
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        
        try:
            target_frame = "right_arm_base_link"  # 目标：右臂基座
            source_frame = "left_gripper_link"    # 源：左手抓手
            
            # 检查是否可以变换
            tf_buffer.can_transform(target_frame, source_frame, rospy.Time.now(), rospy.Duration(1.0))
            
            # 获取变换关系
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
            
            img_height, img_width = color_image.shape[:2]
            
            # 使用YOLO进行推理
            results = self.model.predict(
                source=color_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 创建用于显示的图像副本
            display_image = color_image.copy()
            
            placement_found = False
            
            # 计算放置点
            placement_pos, layer = self.compute_placement(results[0], img_width)
            
            if placement_pos:
                cx, cy = placement_pos
                
                print(f"找到放置点 - 2D坐标: ({cx:.1f}, {cy:.1f}), 层级: {layer}")
                
                # 获取ROI区域的中值深度
                median_depth = self.get_median_depth_in_roi(depth_frame, cx, cy)
                
                # 获取3D坐标
                if median_depth is not None:
                    point_3d = self.get_3d_coordinates(
                        depth_frame, 
                        cx, 
                        cy,
                        depth_value=median_depth
                    )
                else:
                    # 如果中值深度无效，尝试直接获取中心点深度
                    point_3d = self.get_3d_coordinates(depth_frame, cx, cy)
                
                # 绘制放置点
                cv2.circle(display_image, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(display_image, f"Place Here: ({cx:.0f}, {cy:.0f})", 
                           (int(cx)-50, int(cy)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if point_3d is not None:
                    x, y, z = point_3d
                    
                    # 在图像上显示3D坐标
                    coord_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
                    cv2.putText(display_image, coord_text, (int(cx) - 70, int(cy) + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    print(f"放置点3D位置 - 相机坐标系: ({x:.2f}, {y:.2f}, {z:.2f})m")
                    
                    point_camera = np.array([x, y, z])
                    
                    # 转换到right_arm_base坐标系
                    point_right_base = self.transform_point_to_right_arm_base(point_camera)

                    if point_right_base is not None:
                        print(f"放置点3D位置 - 右臂基座坐标系: ({point_right_base[0]:.2f}, {point_right_base[1]:.2f}, {point_right_base[2]:.2f})m")
                        
                        # 发布3D位置信息
                        # 相机坐标系
                        camera_point_msg = PointStamped()
                        camera_point_msg.header = Header()
                        camera_point_msg.header.stamp = rospy.Time.now()
                        camera_point_msg.header.frame_id = "camera_color_optical_frame"
                        camera_point_msg.point = Point(x, y, z)
                        self.placement_3d_pub.publish(camera_point_msg)
                        
                        # 右臂基座坐标系
                        base_point_msg = PointStamped()
                        base_point_msg.header = Header()
                        base_point_msg.header.stamp = rospy.Time.now()
                        base_point_msg.header.frame_id = "right_arm_base_link"
                        base_point_msg.point = Point(point_right_base[0], point_right_base[1], point_right_base[2])
                        self.placement_base_pub.publish(base_point_msg)
                
                placement_found = True
            
            # 如果没有找到放置点，显示提示信息
            if not placement_found:
                cv2.putText(display_image, "No Placement Found", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("未找到合适的放置点")
            
            # 显示检测信息
            # info_text = f"Target: {self.target_item}"
            info_text = f"Target: Fruit"
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('YOLO Placement Detection', display_image)
            
            return True
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return False

    def run_realtime(self):
        """实时运行模式"""
        print(f"开始实时放置点检测 (目标: {self.target_item}, 按 'q' 键退出)...")
        
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

def main():
    # 创建放置点检测器
    detector = YOLOPlacementDetector(
        target_item="orange",
        exclude_classes=["potted plant"],
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    
    # 运行实时检测
    detector.run_realtime()

if __name__ == "__main__":
    main()