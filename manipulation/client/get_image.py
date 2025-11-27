#!/usr/bin/env python
import rospy
import cv2
import os
import numpy as np
import json
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import argparse

class ImageSaver:
    def __init__(self, save_path):
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None

        # self.save_path = rospy.get_param('~data_path', '/tmp/rs_capture/')
        self.save_path =save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        print("init finish")

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                "camera_matrix": [
                    [msg.K[0], 0.0, msg.K[2]],
                    [0.0, msg.K[4], msg.K[5]],
                    [0.0, 0.0, 1.0]
                ],
                "distortion_coefficients": [msg.D]
            }

    def save_all(self):
        # print(self.color_image)
        # print(self.depth_image)
        # print(self.intrinsics)
        if self.color_image is not None and self.depth_image is not None and self.intrinsics is not None:
            print(self.save_path)
            # print(self.color_image)
            cv2.imwrite(os.path.join(self.save_path, 'color.png'), self.color_image)
            cv2.imwrite(os.path.join(self.save_path, 'depth.png'), self.depth_image)
            with open(os.path.join(self.save_path, 'camera.json'), 'w') as f:
                json.dump(self.intrinsics, f, indent=4)
            print("Saved color.png, depth.png, camera.json")

            # workspace_mask.png
            h, w = self.color_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            top_left = (w // 2 - h // 2, 0)
            bottom_right = (w // 2 + h // 2, h)
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            cv2.imwrite(os.path.join(self.save_path, 'workspace_mask.png'), mask)
            print("Saved workspace_mask.png")
            rospy.signal_shutdown("Data saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Output folder to save color.png, depth.png, and camera.json')
    cfgs = parser.parse_args()
    rospy.init_node('realsense_ros_image_saver', anonymous=True)
    saver = ImageSaver(cfgs.data_path)
    
    # 等待所有图像都收到再保存
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        saver.save_all()
        rate.sleep()
