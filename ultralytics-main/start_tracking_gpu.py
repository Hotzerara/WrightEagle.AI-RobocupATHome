import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def initialize_realsense():
    """初始化RealSense摄像头并配置更高分辨率"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置更高分辨率的彩色流 (1280x720)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 启动流
    pipeline.start(config)
    return pipeline

def main():
    # 初始化RealSense
    pipeline = initialize_realsense()
    
    # 加载YOLOv8模型 - 使用GPU加速
    model = YOLO('yolov8n.pt').to('cuda')  # 使用GPU加速
    
    # 创建更大的显示窗口
    cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Tracking', 1280, 720)  # 设置窗口初始大小
    
    # 只跟踪person类（ID 0）
    target_class_id = 0
    
    try:
        print("开始人员跟踪 (按 'q' 键退出)...")
        while True:
            # 等待下一组帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # 转换为OpenCV格式
            color_image = np.asanyarray(color_frame.get_data())
            
            # 使用YOLOv8进行目标跟踪 (GPU加速)
            results = model.track(
                source=color_image,
                persist=True,  # 保持跟踪ID
                classes=[target_class_id],  # 只检测person类
                conf=0.5,  # 置信度阈值
                verbose=False,  # 减少控制台输出
                device='cuda:0'  # 使用GPU加速
            )
            
            # 在图像上绘制结果
            if results is not None and len(results) > 0:
                annotated_frame = results[0].plot(line_width=2, font_size=1.2)
                
                # 显示跟踪结果
                cv2.imshow('Person Tracking', annotated_frame)
            else:
                cv2.imshow('Person Tracking', color_image)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止流并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置环境变量确保兼容性
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    main()