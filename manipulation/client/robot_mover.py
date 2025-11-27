#!/usr/bin/env python
import sys
import time
import numpy as np
from robots import R1Robot

def cubic_polynomial(t, T):
    """三次多项式轨迹规划"""
    tau = t / T
    s = 3 * (tau ** 2) - 2 * (tau ** 3)
    return s

def main():
    if len(sys.argv) != 4:
        print("Usage: python robot_mover.py x y z")
        return
    
    try:
        # 解析目标位置
        target_position = [
            float(sys.argv[1]),
            float(sys.argv[2]), 
            float(sys.argv[3])
        ]
        
        # 防止机械臂撞上
        target_position[0] -= 0.1
        target_position[2] += 0.13
        
        
        # 初始化机器人（在新的ROS节点中）
        robot = R1Robot('r1')
        print("Connected to robot successfully")
        
        # 调整目标位置
        first_position = target_position[:]

        first_position[0] = first_position[0] - 0.4
        print(f"First position: {first_position}")
        # print(target_position)
        target_orientation = [0.7, 0, 0, 0.7]
        first_pose = np.concatenate((first_position, target_orientation))
        robot.set_endpose(first_pose)

        target_pose = np.concatenate((target_position, target_orientation))
        print("Target Pose: ", target_pose)
        
        # 获取起始位姿
        start_pose = robot.read_current_pose()
        print(f"Start pose: {start_pose}")
        
        # 轨迹执行代码（与之前相同）
        duration = 2.0
        steps = 10
        interval = duration / steps
        start_time = time.time()

        print("Starting smooth trajectory...")
        
        for i in range(steps + 1):
            current_time = i * interval
            fraction = cubic_polynomial(current_time, duration)

            interpolated_position = [
                start_pose[0] + fraction * (target_position[0] - start_pose[0]),
                start_pose[1] + fraction * (target_position[1] - start_pose[1]),
                start_pose[2] + fraction * (target_position[2] - start_pose[2])
            ]
            interpolated_orientation = [0.7, 0, 0, 0.7]

            interpolated_pose = np.concatenate((interpolated_position, interpolated_orientation))
            robot.set_endpose_quick(interpolated_pose)

            expected_time = start_time + (i + 1) * interval
            sleep_time = expected_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        robot.set_endpose_quick(target_pose)
        print("Trajectory completed successfully")
        
    except Exception as e:
        print(f"Error in robot movement: {str(e)}")

if __name__ == "__main__":
    main()