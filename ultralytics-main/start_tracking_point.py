import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def initialize_realsense():
    """初始化RealSense摄像头"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置彩色流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 启动流
    pipeline.start(config)
    return pipeline

def draw_box_coordinates(image, boxes, track_ids=None, class_names=None):
    """在图像上绘制边界框坐标信息"""
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
    
    return image

def main():
    # 初始化RealSense
    pipeline = initialize_realsense()
    
    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')  # 使用官方预训练模型

    # 创建更大的显示窗口
    cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Tracking', 1280, 720)  # 设置窗口初始大小
    
    # 获取类别名称
    class_names = model.model.names if hasattr(model, 'model') else model.names
    
    # 只跟踪person类（ID 0）
    target_class_id = 0
    
    try:
        print("开始人员跟踪 (按 'q' 键退出)...")
        print("左上角坐标 (x1,y1) 和右下角坐标 (x2,y2) 将显示在图像上")
        
        while True:
            # 等待下一组帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # 转换为OpenCV格式
            color_image = np.asanyarray(color_frame.get_data())
            
            # 使用YOLOv8进行目标跟踪
            results = model.track(
                source=color_image,
                persist=True,  # 保持跟踪ID
                classes=[target_class_id],  # 只检测person类
                conf=0.5,  # 置信度阈值
                verbose=False,  # 减少控制台输出
                device='cpu'  # 强制使用CPU，如果GPU有问题
            )
            
            # 创建用于显示的图像副本
            display_image = color_image.copy()
            
            # 在图像上绘制结果
            if results is not None and len(results) > 0:
                # 获取检测结果
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # 获取跟踪ID（如果可用）
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # 在图像上绘制边界框和坐标
                display_image = draw_box_coordinates(
                    display_image, 
                    boxes, 
                    track_ids=track_ids,
                    class_names=class_names
                )
                
                # 在控制台打印坐标信息
                print("\n检测到的人员:")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box[:4]
                    track_id = track_ids[i] if track_ids is not None and i < len(track_ids) else "N/A"
                    print(f"ID: {track_id} | 左上角: ({x1:.1f}, {y1:.1f}) | 右下角: ({x2:.1f}, {y2:.1f})")
                
                # 显示跟踪结果
                cv2.imshow('Person Tracking', display_image)
            else:
                # 没有检测到人员时显示原始图像
                cv2.imshow('Person Tracking', display_image)
            
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