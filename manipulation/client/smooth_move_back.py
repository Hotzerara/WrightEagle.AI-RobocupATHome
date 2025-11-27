import time
import numpy as np
from robots import R1Robot

def smooth_return_to_start(start_pose, duration=5.0, steps=10):
    """
    从当前位置平滑返回到初始位置，保持orientation不变
    
    Args:
        start_pose: 初始位置坐标 [x, y, z, qx, qy, qz, qw]
        duration: 平滑运动的总时间
        steps: 平滑运动的步数
    """
    robot = R1Robot('r1')
    
    # 获取当前位置
    current_pose = robot.read_current_pose()
    
    print(f"当前位置: {current_pose[:3]}")
    print(f"目标位置(初始位置): {start_pose[:3]}")
    print(f"保持姿态: {start_pose[3:]}")
    
    # 使用smoother_move函数平滑返回
    smoother_move(start_pose, duration, steps)

def smoother_move(target_pose, duration=2.0, steps=10):
    """
    使用轨迹规划进行平滑移动
    Args:
        target_pose: 目标位姿 [x, y, z, qx, qy, qz, qw]
        duration: 运动总时间（秒）
        steps: 分解的步数
    """
    robot = R1Robot('r1')
    cur_pose = robot.read_current_pose()
    start_position = cur_pose[:3]
    start_position[2] += 0.05
    start_pose = np.concatenate((start_position, target_pose[3:]))
    
    # 先让机械臂抬起来一点
    robot.set_endpose_quick(start_pose)

    start_time = time.time()
    interval = duration / steps

    for i in range(steps + 1):
        current_time = i * interval
        fraction = cubic_polynomial(current_time, duration)

        # 计算插值位姿
        interpolated_position = [
            start_pose[0] + fraction * (target_pose[0] - start_pose[0]),
            start_pose[1] + fraction * (target_pose[1] - start_pose[1]),
            start_pose[2] + fraction * (target_pose[2] - start_pose[2])
        ]
        
        # 保持目标姿态不变
        interpolated_orientation = target_pose[3:]
        
        print(f"步数 {i}/{steps}, 位置: {[f'{p:.3f}' for p in interpolated_position]}")
        
        interpolated_pose = np.concatenate((interpolated_position, interpolated_orientation))
        robot.set_endpose_quick(interpolated_pose)

        # 精确时间控制
        expected_time = start_time + (i + 1) * interval
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 确保最后到达目标点
    robot.set_endpose(target_pose)
    print("平滑返回完成")

def cubic_polynomial(t, T):
    """三次多项式轨迹规划，t是当前时间，T是总时间。返回插值系数s (0->1)"""
    if T == 0:
        return 1.0
    tau = t / T
    s = 3 * (tau ** 2) - 2 * (tau ** 3)
    return s

# 使用示例
if __name__ == "__main__":
    # 假设您已经保存了初始位置
    initial_pose = [0.0, 0.0, 0.3, 0.7, 0, 0, 0.7]  # 请替换为您的实际初始位置
    
    # 平滑返回到初始位置
    smooth_return_to_start(
        start_pose=initial_pose,
        duration=2,
        steps=10
    )