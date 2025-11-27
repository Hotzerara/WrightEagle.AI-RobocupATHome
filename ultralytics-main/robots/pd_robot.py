from .base_robot import Robot
import subprocess
from scipy.spatial.transform import Rotation as R
import os
class PdRobot(Robot):
    def __init__(self, name) -> None:
        super().__init__(name)
    
    def check_ready(self):
        output = subprocess.check_output(['ros2', 'topic', 'echo', '/pd_leftArm/end_pose', '--once'])
        if len(output)>10:
            return True
        else:
            return False
        
    def read_current_pose(self):
        '''
            output:
                pose: xyz rxryrz
        '''
        xyzrxryrz = []
        while True:
            output = subprocess.check_output(['ros2', 'topic', 'echo', '/pd_leftArm/end_pose', '--once'])
            '''
                DATA:
                    pos_xyz:
                    - -0.2332434105314392
                    - 0.7078862105384243
                    - 0.5037974290653124
                    pose_quat:
                    - 0.9263626000716239
                    - 0.11677077392146264
                    - -0.16600052828311618
                    - 0.3172707741914622
                    ---
            '''
            # read pose
            result = output.decode()
            pose = result.replace('- ','').split('\n')
            #if data error will repeat
            if len(pose) >11 :
                print("pose data error, repeat read pose!")
                continue
            point_set = [pose[1],pose[2],pose[3],pose[5],pose[6],pose[7],pose[8]]
            for idx,item in enumerate(point_set):
                point_set[idx] = float(item.replace(' ',''))

            #xyz+quat to xyz+rxryrz
            xyz = point_set[0:3]
            quaternion = point_set[3:]
            rxryrz = R.from_quat(quaternion).as_euler('xyz',degrees=False).tolist()
            xyzrxryrz = xyz+rxryrz
            break

        return xyzrxryrz
    
    def set_pose(self,point):
        '''
            set robot pose
            :param point xyz+rxryrz
        '''
        xyz = point[0:3]
        rxryrz = point[3:]
        quaternion = -R.from_euler('xyz',rxryrz,degrees=False).as_quat()
        quaternion = quaternion.tolist()
        point = xyz+quaternion
        result = os.system('ros2 service call /moveJ_globalIK pd_arm_interface/srv/EndTarget "{end_pose_xyz_quat: ['+f'{point[0]},{point[1]},{point[2]},{point[3]},{point[4]},{point[5]},{point[6]}'+'], time: 4.0}"')
        if result ==0:
            return True
        else:
            return False

    def close(self):
        return super().close()