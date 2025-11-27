#!/usr/bin/env python3
import tf
import tf.transformations as tft
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PointStamped, Point, PoseStamped, Pose
from std_msgs.msg import Header
from tf.transformations import quaternion_matrix, quaternion_from_euler
import math

class AdvancedAppleGripperController:
    def __init__(self):
        rospy.init_node('advanced_apple_gripper_controller')
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Wait for TF to be ready
        rospy.sleep(2.0)
        
        # Create publisher for target pose
        self.target_pose_pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)
        
        # Subscribe to apple position in base link frame
        self.apple_base_sub = rospy.Subscriber(
            '/object/base_link_3d_position', 
            PointStamped, 
            self.apple_base_position_callback
        )
        
        # State variables
        self.current_apple_position_base = None  # Position 1: Apple in base_link frame
        self.current_gripper_position = None     # Position 2: Gripper in base_link frame
        self.is_moving = False
        self.message_count = 0
        
        # Movement parameters
        self.approach_distance = 0.10  # 10cm from apple
        self.safe_height = 0.20  # 20cm above table/surface
        
        rospy.loginfo("Advanced Apple Gripper Controller ready!")
        rospy.loginfo("Waiting for apple positions...")

    def get_tf_transform(self, target_frame, source_frame):
        """
        Get TF transform between frames, returns (x, y, z, roll, pitch, yaw)
        """
        listener = tf.TransformListener()
        try:
            listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
            
            # Extract position (x, y, z)
            x, y, z = trans
            
            # Convert quaternion to Euler angles (radians)
            roll, pitch, yaw = tft.euler_from_quaternion(rot)
            
            # Return as a list instead of tuple concatenation
            return [x, y, z, roll, pitch, yaw]
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed: %s -> %s" % (source_frame, target_frame))
            return None

    def get_gripper_position_base_frame(self):
        """Get current right gripper position in right_arm_base_link frame using TF"""
        try:
            # Get transform from right_gripper_link to base_link
            gripper_tf = self.get_tf_transform("right_gripper_link", "right_arm_base_link")
            
            if gripper_tf is not None:
                # Extract position (x, y, z) - first three elements
                gripper_position = np.array([gripper_tf[0], gripper_tf[1], gripper_tf[2]])
                rospy.loginfo(f"Gripper position in right_arm_base_link: [{gripper_position[0]:.3f}, {gripper_position[1]:.3f}, {gripper_position[2]:.3f}] m")
                return gripper_position
            else:
                rospy.logwarn("Failed to get gripper position from TF")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error getting gripper position: {e}")
            return None

    def set_endpose(self, pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # Set Header information
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # Current timestamp
        target_pose.header.frame_id = "right_arm_base_link"  # Important: specify the frame
        
        # Set target position
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # Set target orientation (quaternion)
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # Publish message multiple times for reliability
        rate = rospy.Rate(10)  # 10Hz
        for i in range(50):
            self.target_pose_pub.publish(target_pose)
            rate.sleep()
        
        rospy.loginfo(f"Published target pose: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")

    def calculate_target_orientation(self, direction_vector):
        """Calculate orientation for the gripper to point along the movement direction"""
        # Calculate yaw and pitch from the direction vector
        # Yaw: rotation around Z axis (pointing horizontally toward apple)
        yaw = math.atan2(direction_vector[1], direction_vector[0])
        
        # Pitch: rotation around Y axis (pointing up/down toward apple)
        pitch = math.atan2(-direction_vector[2], math.sqrt(direction_vector[0]**2 + direction_vector[1]**2))
        
        # Roll: keep it zero (no sideways tilt)
        roll = 0.0
        
        # Convert Euler angles to quaternion
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        
        return quaternion

    def apple_base_position_callback(self, msg):
        """Process apple position in right_arm_base_link frame - Position 1"""
        self.message_count += 1
        
        rospy.loginfo(f"\n=== Received Apple Position #{self.message_count} ===")
        rospy.loginfo(f"Frame: {msg.header.frame_id}")
        rospy.loginfo(f"Apple Position (right_arm_base_link): [{msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}] m")
        
        # Store the apple position (Position 1)
        self.current_apple_position_base = np.array([
            msg.point.x,
            msg.point.y, 
            max(msg.point.z, self.safe_height)  # Ensure safe height
        ])
        
        # Get current gripper position (Position 2)
        self.current_gripper_position = self.get_gripper_position_base_frame()
        
        if self.current_gripper_position is not None:
            rospy.loginfo(f"Gripper Position (right_arm_base_link): [{self.current_gripper_position[0]:.3f}, {self.current_gripper_position[1]:.3f}, {self.current_gripper_position[2]:.3f}] m")
            
            # Calculate distance between gripper and apple
            distance = np.linalg.norm(self.current_apple_position_base - self.current_gripper_position)
            rospy.loginfo(f"Distance between gripper and apple: {distance:.3f} m")
        
        if not self.is_moving and self.current_apple_position_base is not None and self.current_gripper_position is not None:
            self.move_to_apple()

    def calculate_movement_vector(self):
        """Calculate movement vector from gripper to apple"""
        if self.current_apple_position_base is None or self.current_gripper_position is None:
            rospy.logerr("Cannot calculate movement vector: Missing positions")
            return None, None
        
        # Vector from gripper (Position 2) to apple (Position 1)
        direction_vector = self.current_apple_position_base - self.current_gripper_position
        distance = np.linalg.norm(direction_vector)
        
        rospy.loginfo(f"Raw distance to apple: {distance:.3f} m")
        
        if distance <= 0:
            rospy.logwarn("Apple is at gripper position or invalid distance")
            return self.current_gripper_position, direction_vector
        
        # Normalize direction vector
        normalized_direction = direction_vector / distance
        
        # Calculate how far to move along the vector
        if distance <= self.approach_distance:
            # If already close, move half the remaining distance
            move_distance = distance * 0.5
            rospy.loginfo("Apple is close, moving 50% of remaining distance")
        else:
            # Move most of the way but stop at approach distance
            move_distance = distance * 0.5
            rospy.loginfo(f"Moving {move_distance:.3f}m (stopping {self.approach_distance}m from apple)")
        
        # Ensure minimum movement
        move_distance = max(0.02, move_distance)  # At least 2cm
        
        # Calculate target position
        target_position = self.current_gripper_position + normalized_direction * move_distance
        
        # Ensure we don't go below safe height
        target_position[2] = max(target_position[2], self.safe_height)
        
        rospy.loginfo(f"Movement direction: {normalized_direction}")
        rospy.loginfo(f"Movement distance: {move_distance:.3f} m")
        rospy.loginfo(f"Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] m")
        
        return target_position, normalized_direction

    def move_to_apple(self):
        """Execute movement towards apple along the vector"""
        if self.current_apple_position_base is None or self.current_gripper_position is None:
            rospy.logwarn("No apple or gripper position available for movement")
            return
            
        if self.is_moving:
            rospy.logwarn("Already moving, skipping new movement command")
            return
            
        self.is_moving = True
        
        try:
            rospy.loginfo("Starting movement towards apple...")
            
            # Calculate target position and movement vector
            target_position, movement_vector = self.calculate_movement_vector()
            if target_position is None:
                rospy.logerr("Failed to calculate movement vector")
                return
            
            # Calculate orientation for the target pose
            orientation = self.calculate_target_orientation(movement_vector)
            
            # Create pose array: [x, y, z, qx, qy, qz, qw]
            target_pose = [
                target_position[0],  # x
                target_position[1],  # y
                target_position[2],  # z
                orientation[0],      # qx
                orientation[1],      # qy
                orientation[2],      # qz
                orientation[3]       # qw
            ]
            
            # Send target pose using set_endpose method
            rospy.loginfo("Sending target pose to motion controller...")
            self.set_endpose(target_pose)
            
            rospy.loginfo("✓ Successfully sent target pose!")
                
        except Exception as e:
            rospy.logerr(f"Movement failed: {e}")
        finally:
            self.is_moving = False
            rospy.loginfo("Movement execution completed")

    def run(self):
        """Main loop - sends one position then sleeps forever"""
        rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("Controller main loop started - will execute ONE movement only")
        
        # Wait for apple detection with timeout
        wait_start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            
            if self.current_apple_position_base is not None and self.current_gripper_position is not None:
                rospy.loginfo("Apple detected! Executing ONE-TIME movement...")
                self.move_to_apple()
                rospy.loginfo("ONE-TIME movement completed. No further movements will be executed.")
                break
            elif (current_time - wait_start_time).to_sec() > 30.0:
                rospy.logwarn("No apple detected within 30 seconds. Shutting down.")
                return
            
            rate.sleep()
        
        # After sending one position, sleep forever and ignore all future apple detections
        rospy.loginfo("Node now inactive - ignoring all future apple positions")
        while not rospy.is_shutdown():
            rospy.sleep(1.0)


def main():
    try:
        controller = AdvancedAppleGripperController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Apple Gripper Controller shutdown")
    except Exception as e:
        rospy.logerr(f"Error in Apple Gripper Controller: {e}")

if __name__ == "__main__":
    main()