#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
import math

class ManualGoalTester:
    def __init__(self):
        rospy.init_node('manual_goal_tester_node')

        # === Parameters ===
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')

        # === TF2 Buffer and Listener ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === Publishers ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=1)

        # === Variables ===
        self.last_goal = None
        
        rospy.loginfo("Manual Goal Tester Node Started")
        rospy.loginfo("Testing with hardcoded goal: 1.0 meters forward, 0.0 left/right, 0.0 rad orientation")

    def send_goal_relative_to_base_link(self, x, y, theta=0.0):
        """
        Send a goal relative to the robot's base_link frame
        x, y: coordinates in meters relative to base_link
        theta: orientation in radians (0 = facing forward)
        """
        try:
            # Create a point in base_link frame
            point_base = PointStamped()
            point_base.header.frame_id = self.robot_frame
            point_base.header.stamp = rospy.Time.now()
            point_base.point.x = x
            point_base.point.y = y
            point_base.point.z = 0.0

            # Transform point to map frame
            transform = self.tf_buffer.lookup_transform(self.global_frame, 
                                                       self.robot_frame,
                                                       rospy.Time(0),
                                                       rospy.Duration(1.0))
            point_map = tf2_geometry_msgs.do_transform_point(point_base, transform)


            # Create goal pose in map frame
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = self.global_frame
            goal_msg.header.stamp = rospy.Time.now()

            # Set position
            goal_msg.pose.position = point_map.point

            # Set orientation - face the direction specified by theta
            # This is relative to the robot's current orientation
            try:
                # Get robot's current orientation in map frame
                robot_transform = self.tf_buffer.lookup_transform(self.global_frame, 
                                                                self.robot_frame, 
                                                                rospy.Time(0))
                robot_orientation = robot_transform.transform.rotation
                
                # Convert to Euler angles
                robot_euler = tf.transformations.euler_from_quaternion([
                    robot_orientation.x,
                    robot_orientation.y, 
                    robot_orientation.z,
                    robot_orientation.w
                ])
                
                # Add the relative theta to robot's current yaw
                new_yaw = robot_euler[2] + theta
                
                # Create new quaternion
                q = tf.transformations.quaternion_from_euler(0, 0, new_yaw)
                goal_msg.pose.orientation.x = q[0]
                goal_msg.pose.orientation.y = q[1]
                goal_msg.pose.orientation.z = q[2]
                goal_msg.pose.orientation.w = q[3]
                
            except Exception as e:
                rospy.logwarn("Could not set orientation: %s, using default", e)
                goal_msg.pose.orientation.w = 1.0  # Default orientation

            # Publish the goal
            self.goal_pub.publish(goal_msg)
            self.last_goal = goal_msg
            
            # Publish visualization marker
            self.publish_goal_marker(goal_msg.pose.position)
            
            rospy.loginfo("Sent goal to: (%.2f, %.2f) in %s frame", 
                         goal_msg.pose.position.x, goal_msg.pose.position.y, self.global_frame)
            rospy.loginfo("Relative to base_link: (%.2f, %.2f, %.2f rad)", x, y, theta)
            
            return True
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("TF transformation failed: %s", e)
            return False

    def publish_goal_marker(self, position):
        """Publish a marker to visualize the goal in Rviz"""
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "manual_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0  # Red color
        marker.lifetime = rospy.Duration(30.0)  # Longer lifetime for manual testing
        
        self.marker_pub.publish(marker)

    def run(self):
        """Main loop - sends hardcoded goals"""
        # Wait for TF buffer to fill and nodes to initialize
        rospy.sleep(2.0)
        
        # Test sequence with different goals
        test_goals = [
            (3.7, 1.9, 0.0),    # 1 meter forward
        ]
        
        for i, (x, y, theta) in enumerate(test_goals):
            rospy.loginfo("=== Sending Test Goal %d ===", i+1)
            rospy.loginfo("Relative coordinates: (%.1f, %.1f, %.1f rad)", x, y, theta)
            
            success = self.send_goal_relative_to_base_link(x, y, theta)
            if success:
                rospy.loginfo("Goal %d sent successfully! Waiting 10 seconds...", i+1)
                rospy.sleep(10.0)  # Wait for robot to reach goal
            else:
                rospy.logerr("Failed to send goal %d", i+1)
                
            rospy.sleep(1.0)  # Brief pause between goals

        rospy.loginfo("All test goals completed!")

if __name__ == '__main__':
    try:
        tester = ManualGoalTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass