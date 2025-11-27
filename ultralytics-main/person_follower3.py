#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
import math

class PersonGoalSender:
    def __init__(self):
        rospy.init_node('person_goal_sender_node')

        # === Parameters ===
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')

        # === TF2 Buffer and Listener ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === Publishers ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/person_goal_marker', Marker, queue_size=1)

        # === Variables ===
        self.last_goal = None
        
        rospy.loginfo("Person Goal Sender Node Started")
        rospy.loginfo("Will send goals to specified map coordinates")

    def send_goal_to_map_coordinates(self, x, y, theta=None):
        """
        Send a goal to specific map coordinates
        x, y: coordinates in map frame
        theta: orientation in radians (None = face towards goal from current position)
        """
        try:
            # Create goal pose in map frame
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = self.global_frame
            goal_msg.header.stamp = rospy.Time.now()

            # Set position
            goal_msg.pose.position.x = x
            goal_msg.pose.position.y = y
            goal_msg.pose.position.z = 0.0

            # Set orientation
            if theta is not None:
                # Use specified orientation
                q = tf.transformations.quaternion_from_euler(0, 0, theta)
                goal_msg.pose.orientation.x = q[0]
                goal_msg.pose.orientation.y = q[1]
                goal_msg.pose.orientation.z = q[2]
                goal_msg.pose.orientation.w = q[3]
            else:
                # Face towards goal from current robot position
                try:
                    # Get robot's current position in map frame
                    robot_transform = self.tf_buffer.lookup_transform(self.global_frame, 
                                                                    self.robot_frame, 
                                                                    rospy.Time(0))
                    robot_x = robot_transform.transform.translation.x
                    robot_y = robot_transform.transform.translation.y
                    
                    # Calculate angle from robot to goal
                    dx = x - robot_x
                    dy = y - robot_y
                    goal_yaw = math.atan2(dy, dx)
                    
                    # Create quaternion
                    q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
                    goal_msg.pose.orientation.x = q[0]
                    goal_msg.pose.orientation.y = q[1]
                    goal_msg.pose.orientation.z = q[2]
                    goal_msg.pose.orientation.w = q[3]
                    
                    rospy.loginfo("Auto-calculated orientation: %.2f rad (facing goal)", goal_yaw)
                    
                except Exception as e:
                    rospy.logwarn("Could not calculate orientation: %s, using default", e)
                    goal_msg.pose.orientation.w = 1.0  # Default orientation

            # Publish the goal
            self.goal_pub.publish(goal_msg)
            self.last_goal = goal_msg
            
            # Publish visualization marker
            self.publish_goal_marker(goal_msg.pose.position, goal_msg.pose.orientation)
            
            rospy.loginfo("Sent goal to map coordinates: (%.2f, %.2f)", x, y)
            
            return True
            
        except Exception as e:
            rospy.logerr("Failed to send goal: %s", e)
            return False

    def publish_goal_marker(self, position, orientation):
        """Publish a marker to visualize the goal in Rviz"""
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "person_goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation = orientation
        marker.scale.x = 0.5  # Length of arrow
        marker.scale.y = 0.1  # Width of arrow
        marker.scale.z = 0.1  # Height of arrow
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # Green color for person goal
        marker.lifetime = rospy.Duration(60.0)  # Longer lifetime
        
        self.marker_pub.publish(marker)

    def send_person_goal(self, person_x, person_y, distance=1.0, orientation=None):
        """
        Send goal to approach a person at specific map coordinates
        person_x, person_y: Person's position in map frame
        distance: Distance to stop from person (meters)
        orientation: None = face person, or specific angle in radians
        """
        try:
            # Get robot's current position
            robot_transform = self.tf_buffer.lookup_transform(self.global_frame, 
                                                            self.robot_frame, 
                                                            rospy.Time(0))
            robot_x = robot_transform.transform.translation.x
            robot_y = robot_transform.transform.translation.y
            
            # Calculate direction from robot to person
            dx = person_x - robot_x
            dy = person_y - robot_y
            dist_to_person = math.sqrt(dx*dx + dy*dy)
            
            if dist_to_person <= distance:
                rospy.logwarn("Robot is already within %.1f meters of person", distance)
                return False
            
            # Calculate goal position (stop at specified distance from person)
            goal_x = person_x - (dx / dist_to_person) * distance
            goal_y = person_y - (dy / dist_to_person) * distance
            
            rospy.loginfo("Person at: (%.2f, %.2f)", person_x, person_y)
            rospy.loginfo("Approach goal at: (%.2f, %.2f) - %.1f meters from person", 
                         goal_x, goal_y, distance)
            
            return self.send_goal_to_map_coordinates(goal_x, goal_y, orientation)
            
        except Exception as e:
            rospy.logerr("Failed to calculate person approach goal: %s", e)
            return False

    def run_demo(self):
        """Demo with hardcoded person positions"""
        # Wait for TF and nodes to initialize
        rospy.sleep(2.0)
        
        # Example person positions in map coordinates
        person_positions = [
            (1.0, 0.0, 1.0),    # Person at (1, 0), stop 1.0m away
            (2.0, 1.0, 1.5),    # Person at (2, 1), stop 1.5m away
            (-1.0, -1.0, 1.0),  # Person at (-1, -1), stop 1.0m away
        ]
        
        for i, (person_x, person_y, distance) in enumerate(person_positions):
            rospy.loginfo("=== Approaching Person %d at (%.1f, %.1f) ===", 
                         i+1, person_x, person_y)
            
            success = self.send_person_goal(person_x, person_y, distance)
            if success:
                rospy.loginfo("Goal %d sent! Waiting 15 seconds for robot to reach...", i+1)
                rospy.sleep(15.0)
            else:
                rospy.logerr("Failed to send goal %d", i+1)
                
            rospy.sleep(2.0)

        rospy.loginfo("All person approach goals completed!")

    def run_single_goal(self, x, y, theta=None):
        """Send a single goal to specific map coordinates"""
        rospy.sleep(2.0)  # Wait for initialization
        
        rospy.loginfo("=== Sending goal to (%.2f, %.2f) ===", x, y)
        success = self.send_goal_to_map_coordinates(x, y, theta)
        
        if success:
            rospy.loginfo("Goal sent successfully!")
        else:
            rospy.logerr("Failed to send goal")
        
        return success

if __name__ == '__main__':
    try:
        sender = PersonGoalSender()
        
        # === CHOOSE ONE OPTION ===
        
        # Option 1: Run demo with multiple person positions
        # sender.run_demo()
        
        # Option 2: Send to specific coordinates (modify these values)
        sender.run_single_goal(x=-2.4, y=-1.2, theta=0.0)  # Go to (2.0, 1.5) facing 0 rad
        
        # Option 3: Approach a person at specific coordinates
        # sender.send_person_goal(person_x=3.0, person_y=2.0, distance=1.0)
        
    except rospy.ROSInterruptException:
        pass