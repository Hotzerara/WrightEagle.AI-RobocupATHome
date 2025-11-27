#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
import math

class PersonFollower:
    def __init__(self):
        rospy.init_node('person_follower_node')

        self.goal_update_distance = rospy.get_param('~goal_update_distance', 0.6)
        self.follow_distance = rospy.get_param('~follow_distance', 1.5)
        self.global_frame = rospy.get_param('~global_frame', 'map')

        # === TF2 监听器 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === 发布与订阅 ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/person_marker', Marker, queue_size=1)
        rospy.Subscriber('/person/base_link_3d_position', PointStamped, self.person_callback)

        # === 变量 ===
        self.last_goal_position = None
        rospy.loginfo("人物跟随节点已启动。")

    def person_callback(self, point_stamped_msg):
        # 1. 坐标变换：将 base_link 下的点转换到 map 坐标系
        try:
            # 等待并获取最新的 transform
            transform = self.tf_buffer.lookup_transform(self.global_frame,
                                                        point_stamped_msg.header.frame_id,
                                                        rospy.Time(0),
                                                        rospy.Duration(1.0))
            person_point_map = tf2_geometry_msgs.do_transform_point(point_stamped_msg, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("坐标变换失败: %s", e)
            return

        # 2. 计算与上一个目标的距离，决定是否更新目标
        if self.last_goal_position:
            dist = math.sqrt((person_point_map.point.x - self.last_goal_position.x)**2 +
                             (person_point_map.point.y - self.last_goal_position.y)**2)
            if dist < self.goal_update_distance:
                # 移动距离太小，不更新目标
                self.publish_person_marker(person_point_map.point) # 但仍然更新Rviz显示
                return

        # 3. 生成新的导航目标 (PoseStamped)
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.global_frame
        goal_msg.header.stamp = rospy.Time.now()

        # 目标点设置在人后方一定距离，避免碰撞
        # 首先获取机器人的当前位置
        try:
            robot_transform = self.tf_buffer.lookup_transform(self.global_frame, 'base_link', rospy.Time(0))
            robot_pos = robot_transform.transform.translation
            
            # 计算人到机器人的方向向量
            dx = person_point_map.point.x - robot_pos.x
            dy = person_point_map.point.y - robot_pos.y
            angle_to_person = math.atan2(dy, dx)
            
            # 在人的位置基础上，沿着“人-机器人”反方向后退 a follow_distance 的距离
            # 这样目标点总是在机器人和人之间
            goal_x = person_point_map.point.x - self.follow_distance * math.cos(angle_to_person)
            goal_y = person_point_map.point.y - self.follow_distance * math.sin(angle_to_person)

            goal_msg.pose.position.x = goal_x
            goal_msg.pose.position.y = goal_y

            # 让机器人朝向人的方向
            q = tf.transformations.quaternion_from_euler(0, 0, angle_to_person)
            goal_msg.pose.orientation.x = q[0]
            goal_msg.pose.orientation.y = q[1]
            goal_msg.pose.orientation.z = q[2]
            goal_msg.pose.orientation.w = q[3]

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("获取机器人位置失败: %s, 将直接使用人的位置作为目标", e)
            goal_msg.pose.position = person_point_map.point
            # 简单的让机器人朝向目标点
            # ... (此处省略朝向计算)

        # 4. 发布目标并更新状态
        self.goal_pub.publish(goal_msg)
        self.last_goal_position = goal_msg.pose.position
        rospy.loginfo("已更新导航目标至: (%.2f, %.2f)", goal_msg.pose.position.x, goal_msg.pose.position.y)
        
        # 5. 发布可视化标记
        self.publish_person_marker(person_point_map.point)


    def publish_person_marker(self, position):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "person"
        marker.id = 0
        marker.type = Marker.CYLINDER # 使用圆柱体模拟人
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4 # 直径
        marker.scale.y = 0.4
        marker.scale.z = 1.5 # 高度
        marker.color.a = 1.0 # 不透明
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0 # 绿色
        marker.lifetime = rospy.Duration(1.0) # 标记持续1秒
        self.marker_pub.publish(marker)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        follower = PersonFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass