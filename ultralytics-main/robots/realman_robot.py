from .base_robot import Robot
from .robotic_arm_package.robotic_arm import *

class RealmanRobot(Robot):
    def __init__(self, name,host) -> None:
        super().__init__(name)
        self.robot = Arm(RM75, host)

    def check_ready(self):
        state = self.robot.Arm_Socket_State()
        if state!=0:
            print(f"error state:{state}")
            return False
        else:
            return True
    
    def read_current_pose(self):
        '''
            output:
                pose: xyz rxryrz
        '''
        e,joints,xyzrxryrz,_,_ = self.robot.Get_Current_Arm_State()#单位m+弧度
        return xyzrxryrz
    
    def set_pose(self,pose):
        '''
            input:
                pose: xyz rxryrz
        '''
        ret = self.robot.Movel_Cmd(pose=pose,v=20,trajectory_connect=0)
        
        return ret
    
    def close(self):
        self.robot.Arm_Socket_Close()
        
if __name__ == '__main__':
    robot = RealmanRobot('realman','192.168.1.18')
    result = robot.check_ready()
    print(result)
    result = robot.read_current_pose()
    print(result)
    pose = [0.2752929925918579, -0.24337700009346008, 0.49572598934173584, -2.997999906539917, 0.6449999809265137, 2.4830000400543213]
    result = robot.set_pose(pose)
    print(result)