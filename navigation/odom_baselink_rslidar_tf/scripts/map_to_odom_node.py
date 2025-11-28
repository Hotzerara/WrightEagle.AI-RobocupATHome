#!/usr/bin/env python  

import rospy  
import tf  
import tf2_ros  
import geometry_msgs.msg  
from nav_msgs.msg import Odometry  

class MapToOdomPublisher:  
    def __init__(self):  
        rospy.init_node('map_to_odom_broadcaster', anonymous=True)  

        self.br = tf2_ros.TransformBroadcaster()  
        self.listener = tf.TransformListener()  
        
        # Subscribe to the odometry topic  
        self.odom_sub = rospy.Subscriber('odom_slam', Odometry, self.odometry_callback)  
        
        self.map_to_rslidar_trans = None  
        self.map_to_rslidar_rot = None  

        self.rate = rospy.Rate(10.0)  
        self.run()  

    def odometry_callback(self, msg):
        # Extract the position and orientation from the Odometry message  
        position = msg.pose.pose.position  
        orientation = msg.pose.pose.orientation  
        
        # Store the translation and rotation for map -> rslidar  
        self.map_to_rslidar_trans = (position.x, position.y, position.z)  
        self.map_to_rslidar_rot = (orientation.x, orientation.y, orientation.z, orientation.w)  

    def run(self):  
        while not rospy.is_shutdown():
            try:  
                # Only proceed if we have received the map -> rslidar transformation  
                if self.map_to_rslidar_trans is None or self.map_to_rslidar_rot is None:  
                    rospy.loginfo("Waiting for map -> rslidar transform...")
                    continue
                
                # Obtain existing transformations  
                (trans_odom_base, rot_odom_base) = self.listener.lookupTransform('odom_fusion', 'base_link_fusion', rospy.Time(0))  
                (trans_base_rslidar, rot_base_rslidar) = self.listener.lookupTransform('base_link_fusion', 'livox_frame', rospy.Time(0))  

                # Invert the baselink -> rslidar transformation  
                inv_trans_base_rslidar = tf.transformations.translation_matrix(trans_base_rslidar)  
                inv_rot_base_rslidar = tf.transformations.quaternion_matrix(rot_base_rslidar)  
                inv_odom_base = tf.transformations.concatenate_matrices(inv_trans_base_rslidar, inv_rot_base_rslidar)  
                inv_odom_base = tf.transformations.inverse_matrix(inv_odom_base)  
                
                # Combine transformations to get map -> odom  
                trans_map_odom = tf.transformations.translation_matrix(self.map_to_rslidar_trans)  
                rot_map_odom = tf.transformations.quaternion_matrix(self.map_to_rslidar_rot)  
                map_rslidar = tf.transformations.concatenate_matrices(trans_map_odom, rot_map_odom)  
                
                map_odom = tf.transformations.concatenate_matrices(map_rslidar, inv_odom_base)  

                # Extract translation and rotation  
                final_trans = tf.transformations.translation_from_matrix(map_odom)  
                final_rot = tf.transformations.quaternion_from_matrix(map_odom)  

                # Create and publish the Transform  
                t = geometry_msgs.msg.TransformStamped()  
                
                t.header.stamp = rospy.Time.now()  
                t.header.frame_id = "map"  
                t.child_frame_id = "odom_fusion"  
                t.transform.translation.x = final_trans[0]  
                t.transform.translation.y = final_trans[1]  
                t.transform.translation.z = final_trans[2]  
                t.transform.rotation.x = final_rot[0]  
                t.transform.rotation.y = final_rot[1]  
                t.transform.rotation.z = final_rot[2]  
                t.transform.rotation.w = final_rot[3]  
                
                self.br.sendTransform(t)  

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):  
                rospy.logwarn("Transformation lookup failed")  
                continue  
            
             

if __name__ == '__main__':  
    try:  
        MapToOdomPublisher()
    except rospy.ROSInterruptException:  
        rospy.loginfo("failed to start MapToOdomPublisher node!")