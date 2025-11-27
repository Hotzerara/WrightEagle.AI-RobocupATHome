import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

def initialize_realsense():
    """初始化RealSense摄像头，使用更兼容的配置"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    print("尝试查找RealSense设备...")
    # 尝试查找所有连接的设备
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("未找到RealSense设备! 请检查连接。")
    
    print(f"找到 {len(devices)} 个设备")
    
    # 获取第一个设备支持的配置
    device = devices[0]
    print(f"设备名称: {device.get_info(rs.camera_info.name)}")
    print(f"设备序列号: {device.get_info(rs.camera_info.serial_number)}")
    
    # 尝试查找支持的配置
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动流
    try:
        print("尝试启动流...")
        profile = pipeline.start(config)
        print("RealSense启动成功!")
    except RuntimeError as e:
        print(f"启动失败: {e}")
        print("尝试兼容模式...")
        config.disable_all_streams()
        config.enable_stream(rs.stream.color)  # 使用默认配置
        config.enable_stream(rs.stream.depth)  # 使用默认配置
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
    
    # 打印相机信息
    print(f"深度比例因子: {depth_scale}")
    print(f"相机内参: fx={intrinsics.fx}, fy={intrinsics.fy}, ppx={intrinsics.ppx}, ppy={intrinsics.ppy}")
    print(f"图像尺寸: {intrinsics.width}x{intrinsics.height}")
    
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

def draw_box_coordinates(image, boxes, points_3d=None, track_ids=None, class_names=None):
    """在图像上绘制边界框坐标信息和3D位置"""
    for i, box in enumerate(boxes):
        # 解包边界框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box[:4]
        
        # 转换为整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 计算中心点坐标
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制中心点
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 准备坐标文本
        coords_text = f"({x1},{y1})-({x2},{y2})"
        center_text = f"Center: ({center_x},{center_y})"
        
        # 如果有三维坐标，添加显示
        point_3d_text = ""
        if points_3d is not None and i < len(points_3d) and points_3d[i] is not None:
            x, y, z = points_3d[i]
            point_3d_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
        
        # 如果有跟踪ID，添加到文本
        id_text = ""
        if track_ids is not None and i < len(track_ids):
            id_text = f"ID: {track_ids[i]} "
        
        # 如果有类别名称，添加到文本
        class_text = ""
        if class_names is not None and len(box) > 5:
            class_id = int(box[5])
            class_text = f"{class_names[class_id]} " if class_id in class_names else ""
        
        # 组合所有文本
        full_text = f"{id_text}{class_text}{coords_text}"
        
        # 在边界框上方绘制坐标信息
        cv2.putText(image, full_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 在中心点下方绘制中心坐标
        cv2.putText(image, center_text, (center_x - 70, center_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 在中心点下方绘制3D坐标
        if point_3d_text:
            cv2.putText(image, point_3d_text, (center_x - 70, center_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def main():
    # 初始化RealSense
    try:
        print("初始化RealSense摄像头...")
        pipeline, align, depth_scale, intrinsics = initialize_realsense()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查摄像头连接和权限设置。")
        print("尝试运行: sudo chmod a+rw /dev/video*")
        return
    
    # 加载YOLOv8模型
    print("加载YOLOv8模型...")
    try:
        model = YOLO('yolov8n.pt')  # 使用官方预训练模型
    except Exception as e:
        print(f"加载模型失败: {e}")
        pipeline.stop()
        return
    
    # 获取类别名称
    try:
        class_names = model.model.names if hasattr(model, 'model') else model.names
        print(f"加载的模型支持 {len(class_names)} 个类别")
    except:
        class_names = {0: 'person'}  # 默认类别
        print("使用默认类别: person")
    
    # 创建显示窗口
    cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Tracking', 1280, 720)
    
    # 只跟踪person类（ID 0）
    target_class_id = 0
    
    try:
        print("开始人员跟踪 (按 'q' 键退出)...")
        print("左上角坐标 (x1,y1) 和右下角坐标 (x2,y2) 将显示在图像上")
        print("3D坐标 (X,Y,Z) 显示在中心点下方")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 等待下一组帧
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as e:
                print(f"获取帧失败: {e}")
                break
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("获取帧不完整，跳过...")
                continue
                
            # 转换为OpenCV格式
            color_image = np.asanyarray(color_frame.get_data())
            
            # 使用YOLOv8进行目标跟踪
            try:
                results = model.track(
                    source=color_image,
                    persist=True,  # 保持跟踪ID
                    classes=[target_class_id],  # 只检测person类
                    conf=0.5,  # 置信度阈值
                    verbose=False,  # 减少控制台输出
                    device='cpu'  # 强制使用CPU，如果GPU有问题
                )
            except Exception as e:
                print(f"目标检测失败: {e}")
                continue
            
            # 创建用于显示的图像副本
            display_image = color_image.copy()
            
            # 在图像上绘制结果
            if results is not None and len(results) > 0 and results[0].boxes is not None:
                # 获取检测结果
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # 获取跟踪ID（如果可用）
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # 计算每个检测框中心点的3D坐标
                points_3d = []
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    point_3d = get_3d_coordinates(
                        depth_frame, 
                        depth_scale, 
                        intrinsics, 
                        center_x, 
                        center_y
                    )
                    points_3d.append(point_3d)
                
                # 在图像上绘制边界框和坐标
                display_image = draw_box_coordinates(
                    display_image, 
                    boxes, 
                    points_3d=points_3d,
                    track_ids=track_ids,
                    class_names=class_names
                )
                
                # 在控制台打印坐标信息
                if frame_count % 10 == 0:  # 每10帧打印一次
                    print(f"\n帧 {frame_count}:")
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box[:4]
                        track_id = track_ids[i] if track_ids is not None and i < len(track_ids) else "N/A"
                        point_3d = points_3d[i]
                        
                        if point_3d is not None:
                            x, y, z = point_3d
                            print(f"ID: {track_id} | 2D: ({x1:.1f}, {y1:.1f})-({x2:.1f}, {y2:.1f}) | 3D: ({x:.2f}, {y:.2f}, {z:.2f})m")
                        else:
                            print(f"ID: {track_id} | 2D: ({x1:.1f}, {y1:.1f})-({x2:.1f}, {y2:.1f}) | 3D: 无效深度")
                
                # 显示跟踪结果
                cv2.imshow('Person Tracking', display_image)
            else:
                # 没有检测到人员时显示原始图像
                cv2.imshow('Person Tracking', display_image)
            
            frame_count += 1
            
            # 计算并显示FPS
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户请求退出...")
                break
                
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止流并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()