#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading
import argparse
import time

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

class GripperSmoothController(object):
    def __init__(
        self,
        target_pos,
        step_size=2.0,
        control_rate=1,
        tolerance=30,
        state_topic="/hdas/feedback_gripper_right",
        command_topic="/motion_control/position_control_gripper_right",
        joint_name="right_gripper",
    ):
        """
        :param target_pos: 目标位置
        :param step_size: 每一步移动的距离
        :param control_rate: 控制频率 Hz
        :param tolerance: 认为到达目标的误差阈值
        :param state_topic: 夹爪当前位置的 JointState 话题
        :param command_topic: 控制夹爪的 Float32 话题
        :param joint_name: JointState 中对应夹爪的关节名称
        """
        rospy.init_node("gripper_slow_move_once", anonymous=True)

        self.target_pos = float(target_pos)
        self.step_size = abs(float(step_size))
        self.tolerance = abs(float(tolerance))
        self.rate = rospy.Rate(control_rate)

        self.state_topic = state_topic
        self.command_topic = command_topic
        self.joint_name = joint_name

        # 当前夹爪位置，由 JointState 回调更新
        self.current_pos = None
        self.state_lock = threading.Lock()

        # 反馈“停止”的检测参数
        self.last_pos = None
        self.stable_count = 0
        self.stable_threshold = 0.5  # 认为“没动”的最小变化量
        self.stable_max_count = int(control_rate * 2)  # 连续大约 2 秒没动就认为停止

        # 发布器：控制夹爪位置
        self.cmd_pub = rospy.Publisher(self.command_topic, Float32, queue_size=10)

        # 订阅器：读取夹爪实际位置
        self.state_sub = rospy.Subscriber(self.state_topic, JointState, self.state_callback)

        rospy.loginfo("启动夹爪平滑控制节点")
        rospy.loginfo(
            f"目标位置: {self.target_pos:.3f}, 步长: {self.step_size:.3f}, "
            f"频率: {control_rate:.1f} Hz, 容差: {self.tolerance:.3f}"
        )
        rospy.loginfo(
            f"状态话题: {self.state_topic}, 控制话题: {self.command_topic}, 关节名: {self.joint_name}"
        )

    def state_callback(self, msg):
        """
        JointState 回调，从中提取 joint_name 对应的 position。
        """
        try:
            if self.joint_name in msg.name:
                idx = msg.name.index(self.joint_name)
                pos = msg.position[idx]
                with self.state_lock:
                    self.current_pos = pos
        except Exception as e:
            rospy.logwarn(f"解析 JointState 失败: {e}")

    def wait_for_first_state(self, timeout=5.0):
        """
        等待第一次收到夹爪状态。
        """
        start_time = rospy.Time.now().to_sec()
        rospy.loginfo("等待夹爪当前状态消息...")
        while not rospy.is_shutdown():
            with self.state_lock:
                if self.current_pos is not None:
                    rospy.loginfo(f"收到初始夹爪位置: {self.current_pos:.3f}")
                    # 初始化 last_pos，避免一开始就触发“停止”判断
                    self.last_pos = self.current_pos
                    return True

            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logerr("在指定时间内未收到夹爪状态消息，退出。")
                return False

            rospy.sleep(0.1)

    def control_loop(self):
        """
        控制循环：根据当前实际位置平滑逼近目标位置。
        当当前值与目标值误差小于 tolerance 时结束。
        如果反馈值长时间几乎不变，则发送最终目标值并结束。
        """
        if not self.wait_for_first_state():
            return

        while not rospy.is_shutdown():
            with self.state_lock:
                current = self.current_pos

            if current is None:
                rospy.logwarn("当前夹爪位置未知，等待下一帧状态...")
                self.rate.sleep()
                continue

            error = self.target_pos - current

            # 到达判定：误差在容差范围内，直接结束
            if abs(error) <= self.tolerance:
                rospy.loginfo(
                    f"已到达目标附近: 当前={current:.3f}, "
                    f"目标={self.target_pos:.3f}, 误差={error:.3f}, 容差={self.tolerance:.3f}"
                )
                break

            # 检查反馈是否“停止”
            if self.last_pos is not None:
                delta = abs(current - self.last_pos)
                if delta < self.stable_threshold:
                    self.stable_count += 1
                else:
                    self.stable_count = 0

                if self.stable_count >= self.stable_max_count:
                    rospy.logwarn(
                        f"检测到反馈基本停止 (连续 {self.stable_count} 步 Δ={delta:.5f})，"
                        f"直接发送最终目标值 {self.target_pos:.3f} 并退出控制循环。"
                    )
                    self.cmd_pub.publish(Float32(self.target_pos))
                    time.sleep(1)
                    break

            self.last_pos = current

            # 按步长逼近目标
            direction = 1.0 if error > 0.0 else -1.0
            next_pos = current + direction * self.step_size

            # 防止朝错误方向走过头：如果下一步已经越过目标，则直接发目标值
            if (direction > 0.0 and next_pos > self.target_pos) or (
                direction < 0.0 and next_pos < self.target_pos
            ):
                next_pos = int(self.target_pos)

            # 发布控制命令
            self.cmd_pub.publish(Float32(next_pos))
            rospy.loginfo(
                f"当前: {current:.3f} -> 发送目标: {next_pos:.3f} (总目标: {self.target_pos:.3f})"
            )

            self.rate.sleep()

        rospy.loginfo("控制循环结束，脚本即将退出。")

def parse_args():
    """
    命令行参数：
      python slow.py 0
      python slow.py 50 --step 1.0 --rate 20 --tol 5.0
    """
    parser = argparse.ArgumentParser(
        description="控制机械臂夹爪平滑运动到目标位置（单位与 JointState 一致）。"
    )
    parser.add_argument(
        "target", type=float, help="目标位置（与 JointState.position 单位一致）"
    )
    parser.add_argument(
        "--step", type=float, default=6, help="每一步移动的距离，默认 5.0"
    )
    parser.add_argument(
        "--rate", type=float, default=5, help="控制循环频率 Hz，默认 10.0"
    )
    parser.add_argument(
        "--tol", type=float, default=5, help="认为到达目标位置的误差阈值，默认 15.0"
    )
    parser.add_argument(
        "--state_topic",
        type=str,
        default="/hdas/feedback_gripper_right",
        help="夹爪 JointState 话题名，默认 /hdas/feedback_gripper_right",
    )
    parser.add_argument(
        "--command_topic",
        type=str,
        default="/motion_control/position_control_gripper_right",
        help="控制夹爪的 Float32 话题名，默认 /motion_control/position_control_gripper_right",
    )
    parser.add_argument(
        "--joint_name",
        type=str,
        default="right_gripper",
        help="JointState 中对应夹爪的关节名，默认 right_gripper",
    )

    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = parse_args()

    try:
        controller = GripperSmoothController(
            target_pos=args.target,
            step_size=args.step,
            control_rate=args.rate,
            tolerance=args.tol,
            state_topic=args.state_topic,
            command_topic=args.command_topic,
            joint_name=args.joint_name,
        )

        controller.control_loop()
    except rospy.ROSInterruptException:
        pass