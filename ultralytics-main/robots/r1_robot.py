import tf
import tf.transformations as tft
from .base_robot import Robot
import argparse
import numpy as np
import rospy
from sensor_msgs.msg import JointState  # 导入 JointState 消息类型
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

parser = argparse.ArgumentParser()

class R1Robot(Robot):    
    def __init__(self,name):
        super().__init__(name)
        rospy.init_node('r1_robot', anonymous=True)  # 初始化 ROS 节点
        self.base = "right_arm_base_link"   #标定的坐标的基准坐标系

    def get_tf_transform(self,target_frame, source_frame):
        """
        获取 ROS TF 坐标变换信息，仅返回 (x, y, z) 和 (Roll, Pitch, Yaw) 角度制。
        
        :param target_frame: 目标坐标系 (child frame)
        :param source_frame: 源坐标系 (parent frame)
        :return: (xyz, rpy_degrees) 或 None
        """
        listener = tf.TransformListener()

        try:
            # 等待 TF 变换信息
            listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0))

            # 获取 TF 变换
            (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))

            # 提取 x, y, z
            xyz = trans  # (x, y, z)

            # 将四元数转换为欧拉角 (弧度)
            rpy = tft.euler_from_quaternion(rot)
            rpy = [r for r in rpy]

            return xyz + rpy  # 仅返回平移 (xyz) 和旋转角度 (rpy)
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed: %s -> %s" % (source_frame, target_frame))
            return None

    def check_ready(self):
        return True

    def read_current_pose(self):
        '''
        return: [x,y,z,roll,pitch,yaw]
        弧度制
        '''
        target = "right_gripper_link"
        source = self.base
        pose = self.get_tf_transform(target, source)
        # print("**************************************************")
        # print(pose)
        
        return pose
        

    def get_control_angle(self):
        pass

    def set_pose(self,pose):
        '''
        pose: [x1, x2, x3, x4, x5, x6]
        '''
        pub = rospy.Publisher('/motion_target/target_joint_state_arm_right', JointState, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz 发送
        for i in range(10):
            msg = JointState()
            msg.header.seq = 0
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = ''
            msg.name = ['']
            msg.position = pose
            msg.velocity = [0]
            msg.effort = [0]

            # rospy.loginfo("Publishing JointState: %s", msg)
            pub.publish(msg)
            rate.sleep()
    
    def set_endpose(self,pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # 设置 Header 信息
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # 当前时间戳
        # 设置目标位置
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # 设置目标姿态（四元数）
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # 创建一个 ROS 发布者，发布到指定话题
        rate = rospy.Rate(10)  # 10Hz 发送
        pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)

        # 发布消息
        for i in range(50):
            pub.publish(target_pose)
            rate.sleep()

    def set_gripper(self):
        pass

    def set_angle(self,angle):
        pass
        
    def close(self):
        pass


class R1Robot_left(Robot):    
    def __init__(self,name):
        super().__init__(name)
        rospy.init_node('r1_robot_left', anonymous=True)  # 初始化 ROS 节点
        self.base = "left_arm_base_link"

    def get_tf_transform(self,target_frame, source_frame):
        """
        获取 ROS TF 坐标变换信息，仅返回 (x, y, z) 和 (Roll, Pitch, Yaw) 角度制。
        
        :param target_frame: 目标坐标系 (child frame)
        :param source_frame: 源坐标系 (parent frame)
        :return: (xyz, rpy_degrees) 或 None
        """
        listener = tf.TransformListener()

        try:
            # 等待 TF 变换信息
            listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0))

            # 获取 TF 变换
            (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))

            # 提取 x, y, z
            xyz = trans  # (x, y, z)

            # 将四元数转换为欧拉角 (弧度)
            rpy = tft.euler_from_quaternion(rot)
            rpy = [r for r in rpy]

            return xyz + rpy  # 仅返回平移 (xyz) 和旋转角度 (rpy)
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed: %s -> %s" % (source_frame, target_frame))
            return None

    def check_ready(self):
        return True

    def read_current_pose(self):
        '''
        return: [x,y,z,roll,pitch,yaw]
        弧度制
        '''
        target = "left_gripper_link"
        source = self.base
        pose = self.get_tf_transform(target, source)
        print("**************************************************")
        print(pose)
        
        return pose
        

    def get_control_angle(self):
        pass

    def set_pose(self,pose):
        '''
        pose: [x1, x2, x3, x4, x5, x6]
        '''
        pub = rospy.Publisher('/motion_target/target_joint_state_arm_left', JointState, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz 发送
        for i in range(10):
            msg = JointState()
            msg.header.seq = 0
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = ''
            msg.name = ['']
            msg.position = pose
            msg.velocity = [0]
            msg.effort = [0]

            # rospy.loginfo("Publishing JointState: %s", msg)
            pub.publish(msg)
            rate.sleep()

    def set_endpose(self,pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # 设置 Header 信息
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # 当前时间戳
        # 设置目标位置
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # 设置目标姿态（四元数）
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # 创建一个 ROS 发布者，发布到指定话题
        rate = rospy.Rate(10)  # 10Hz 发送
        pub = rospy.Publisher('/motion_target/target_pose_arm_left', PoseStamped, queue_size=10)

        # 发布消息
        for i in range(10):
            pub.publish(target_pose)
            rate.sleep()
        
    def set_angle(self,angle):
        pass
        
    def close(self):
        pass