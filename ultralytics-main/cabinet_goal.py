#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


# 固定导航目标（只用于第二步导航）
TABLES = {
    'cabinet': {
        'position': {'x': -4.4230454748563925, 'y': 10.095434083421907},
        'orientation': {'x': 0, 'y': 0, 'z': 0.6849788032178179, 'w': 0.7006945012882259}
    },
    'table': {
        'position': {'x': -5.0037099, 'y': 0.868781},
        'orientation': {'x': 0.1556566, 'y': 0.1422990, 'z': -0.7060529, 'w': 0.676026104}
    },
    'bed': {
        'position': {'x': -8.92316723, 'y': 6.5385472477},
        'orientation': {'x': -0.1466802644, 'y': 0.14669437831, 'z': 0.701188290, 'w': 0.6821294}
    }
}


class SmartNavigator:
    def __init__(self):

        self.current_pose = None

        # 发布导航目标
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # 发布底盘速度
        self.vel_pub = rospy.Publisher('/motion_target/target_speed_chassis', TwistStamped, queue_size=1)

        rospy.loginfo("等待 move_base_simple/goal 的订阅者...")
        while self.goal_pub.get_num_connections() == 0:
            rospy.sleep(0.1)
        rospy.loginfo("move_base 已连接。")

        # 订阅 odom
        self.odom_sub = rospy.Subscriber('/local_odom', Odometry, self.odom_callback)
        rospy.loginfo("等待当前位姿...")
        while self.current_pose is None:
            rospy.sleep(0.1)
        rospy.loginfo("已收到 odom 位姿。")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def backup(self, speed=-0.2, duration=2.5):
        """
        后退：速度 * 时间 = 距离
        0.2 m/s * 2.5 s = 0.5m
        """
        rospy.loginfo(f"后退中：速度 {speed} m/s，持续 {duration} 秒")

        move_cmd = TwistStamped()
        move_cmd.header.frame_id = "base_link"
        move_cmd.twist.linear.x = speed

        # 连续发布确保控制有效
        rate = rospy.Rate(10)
        end_time = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < end_time:
            move_cmd.header.stamp = rospy.Time.now()
            self.vel_pub.publish(move_cmd)
            rate.sleep()

        # 停止
        stop_cmd = TwistStamped()
        stop_cmd.header.frame_id = "base_link"
        self.vel_pub.publish(stop_cmd)
        rospy.loginfo("后退完成。")
        rospy.sleep(0.5)

    def go_to_target(self, target_name):
        if target_name not in TABLES:
            rospy.logerr("目标 %s 不存在！", target_name)
            return

        # ====================
        #  第 1 步：先后退 0.5m
        # ====================
        rospy.loginfo("步骤 1：后退 0.5 米...")
        self.backup(speed=-0.2, duration=2.5)

        # ====================
        #  第 2 步：导航到目标
        # ====================
        rospy.loginfo(f"步骤 2：导航到 {target_name}")

        target = TABLES[target_name]
        x = target['position']['x']
        y = target['position']['y']
        q = target['orientation']

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = q['x']
        goal.pose.orientation.y = q['y']
        goal.pose.orientation.z = q['z']
        goal.pose.orientation.w = q['w']

        self.goal_pub.publish(goal)

        rospy.loginfo("已发布 cabinet 目标点：x=%.3f y=%.3f" % (x, y))


if __name__ == '__main__':
    rospy.init_node("smart_table_navigator")

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, help="目标名称 (cabinet/table/bed)", default="cabinet")
    args = parser.parse_args()

    try:
        nav = SmartNavigator()
        nav.go_to_target(args.target)
    except rospy.ROSInterruptException:
        pass
