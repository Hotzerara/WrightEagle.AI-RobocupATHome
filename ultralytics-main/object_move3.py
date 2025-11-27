#!/usr/bin/env python3
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import math

class SimpleGripperOrientationController:
    def __init__(self):
        rospy.init_node('simple_gripper_orientation_controller')
        
        # Create publisher for target pose
        self.target_pose_pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)
        
        # Yaw-Pitch-Roll values (in radians) - MODIFY THESE VALUES AS NEEDED
        self.target_yaw = math.radians(90.0)    # Rotation around Z axis   but around x
        self.target_pitch = math.radians(0.0)  # Rotation around Y axis  
        self.target_roll = math.radians(90.0)   # Rotation around X axis
        
        # Position values - MODIFY THESE IF YOU WANT TO CHANGE POSITION TOO
        self.target_x = 0.06
        self.target_y = 0.0
        self.target_z = 0.3
        
        rospy.loginfo("Simple Gripper Orientation Controller ready!")
        rospy.loginfo(f"Target YPR: Yaw={math.degrees(self.target_yaw):.1f}°, "
                     f"Pitch={math.degrees(self.target_pitch):.1f}°, "
                     f"Roll={math.degrees(self.target_roll):.1f}°")

    def set_orientation_pose(self):
        '''
        Set target pose with specific Yaw-Pitch-Roll orientation
        '''
        target_pose = PoseStamped()
        
        # Set Header information
        target_pose.header = Header()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = "right_arm_base_link"
        
        # Set target position
        target_pose.pose.position.x = self.target_x
        target_pose.pose.position.y = self.target_y
        target_pose.pose.position.z = self.target_z

        # Convert Yaw-Pitch-Roll to quaternion
        # Rotation order: 'sxyz' (static frame, roll around X, pitch around Y, yaw around Z)
        quaternion = tft.quaternion_from_euler(self.target_roll, self.target_pitch, self.target_yaw, 'rzyx')
        
        # Set target orientation (quaternion)
        target_pose.pose.orientation.x = quaternion[0]
        target_pose.pose.orientation.y = quaternion[1]
        target_pose.pose.orientation.z = quaternion[2]
        target_pose.pose.orientation.w = quaternion[3]

        # Publish message
        self.target_pose_pub.publish(target_pose)
        
        rospy.loginfo(f"Published target pose:")
        rospy.loginfo(f"Position: [{self.target_x:.3f}, {self.target_y:.3f}, {self.target_z:.3f}] m")
        rospy.loginfo(f"Orientation - Yaw: {math.degrees(self.target_yaw):.1f}°, "
                     f"Pitch: {math.degrees(self.target_pitch):.1f}°, "
                     f"Roll: {math.degrees(self.target_roll):.1f}°")
        rospy.loginfo(f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")

    def run(self):
        """Main loop - continuously publishes the target orientation"""
        rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("Starting to publish target orientation...")
        
        while not rospy.is_shutdown():
            self.set_orientation_pose()
            rate.sleep()

def main():
    try:
        controller = SimpleGripperOrientationController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Simple Gripper Orientation Controller shutdown")
    except Exception as e:
        rospy.logerr(f"Error in Orientation Controller: {e}")

if __name__ == "__main__":
    main()