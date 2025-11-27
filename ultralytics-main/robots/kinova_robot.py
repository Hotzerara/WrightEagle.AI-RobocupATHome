from .kinova_tools import kinova_api as kinova_api
from .base_robot import Robot
import argparse
import numpy as np
parser = argparse.ArgumentParser()

class KinovaRobot(Robot):
    def __init__(self,name):
        super().__init__(name)

    def check_ready(self):
        try:
            self.cnnt, self.base, self.base_cyclic = kinova_api.connection(parser)
            return True
        except:
            return False

    def read_current_pose(self):
        '''
        return: [x,y,z,roll,pitch,yaw]
        弧度制
        '''

        tmp = kinova_api.Get_cartesian_pose(
            base_cyclic=self.base_cyclic)
        tmp[3] = tmp[3] / 180.0 * np.pi
        tmp[4] = tmp[4] / 180.0 * np.pi
        tmp[5] = tmp[5] / 180.0 * np.pi
        return tmp
    
    def get_control_angle(self):
        '''
        return 7DoF angle of actor
        '''
        return kinova_api.Get_joint_positions(
            self.base,self.base_cyclic)


    def set_pose(self,pose):
        '''
        pose: [x,y,z,roll,pitch,yaw]
        '''
        kinova_api.Send_relative_cartesian_pose(
            self.base, self.base_cyclic, pose)
        
    def set_angle(self,angle):
        '''
        angle: 7DoF angle of actor
        '''
        kinova_api.Send_joint_angles(
            self.base, angle)
        
    def close(self):
        kinova_api.disconnect()


