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

def main():
    # 初始化RealSense
    pipeline = initialize_realsense()
    
    # 加载YOLOv8模型 - 使用完整模型路径避免歧义
    model = YOLO('yolov8n.pt')  # 或使用绝对路径

    # 创建更大的显示窗口
    cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Tracking', 1280, 720)  # 设置窗口初始大小
    
    # 获取类别名称
    class_names = model.model.names if hasattr(model, 'model') else model.names
    
    # 只跟踪person类（ID 0）
    target_class_id = 0
    
    try:
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
            
            # 在图像上绘制结果
            if results is not None and len(results) > 0:
                annotated_frame = results[0].plot()
                # 显示跟踪结果
                cv2.imshow('Person Tracking', annotated_frame)
            else:
                cv2.imshow('Person Tracking', color_image)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 停止流并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()