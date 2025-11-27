import socket
import struct
import json
import cv2
import os
import numpy as np
import sys
import argparse
from robots import R1Robot, R1Robot_left
from scipy.spatial.transform import Rotation as R
import pandas as pd

# --- 配置参数 ---
parser = argparse.ArgumentParser(description="Grasp Socket Client")
parser.add_argument('--server_host', default = '192.168.31.44', required=False, help='Server host address')
parser.add_argument('--server_port', default = 9090, type=int, required=False, help='Server port')
parser.add_argument('--rgb_path', default = './images/color.png', required=False, help='Path to RGB image')
parser.add_argument('--depth_path', default = './images/depth.png',required=False, help='Path to Depth image')
parser.add_argument('--text_prompt', required=False, help='Text prompt for grasping')
parser.add_argument('--mode', default=0, required=False, help='Mode of operation -- 0: graspnet+groundesam, 1:graspnet+groundedsam+vggt')
cfgs = parser.parse_args()

# --- 消息辅助函数 ---
def send_msg(sock, msg_bytes):
    try:
        len_header = struct.pack('>Q', len(msg_bytes))
        sock.sendall(len_header)
        sock.sendall(msg_bytes)
    except Exception as e:
        print(f"发送消息时出错: {e}")
        raise

def recvall(sock, n):
    """
    确保能从socket中接收n个字节的数据
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    try:
        # 接收消息长度
        len_header_bytes = sock.recv(8)

        # 检查连接是否关闭
        # 如果没有接收到数据，说明连接已关闭
        if not len_header_bytes:
            print("Socket连接已关闭")
            return None
        msg_len = struct.unpack('>Q', len_header_bytes)[0]

        # 循环接收，直到接收到完整的消息
        msg_bytes = b''
        while len(msg_bytes) < msg_len:
            remaining_bytes = msg_len - len(msg_bytes)
            bytes_to_recv = min(4096, remaining_bytes)
            chunk = sock.recv(bytes_to_recv)
            if not chunk:
                raise OSError("Socket连接已关闭,接收消息中断")
            msg_bytes += chunk
        return msg_bytes
    
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"Socket接收消息失败: {e}")
        raise
        

def run_grasp_client(server_host, server_port, rgb_path, depth_path, text_prompt, mode):
    print(f"连接到服务器 {server_host}:{server_port}...")

    # --- 数据准备 ---
    print("正在准备数据...")
    # 1.准备JSON
    request_data = {
        'text_prompt': text_prompt,
        'mode': mode
    }
    json_bytes = json.dumps(request_data).encode('utf-8')
    print(f'JSON数据: {request_data}')

    # 2.准备彩色图像
    try:
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_image is None:
            print(f"无法读取RGB图像: {rgb_path}")
            return
        success, rgb_bytes_encoded = cv2.imencode('.png', rgb_image)
        rgb_bytes = rgb_bytes_encoded.tobytes()
        print(f'RGB图像 {rgb_path} 大小: {len(rgb_bytes)} 字节')
    except Exception as e:
        print(f"读取RGB图像时出错: {e}")
        return
    
    # 3.准备深度图像
    try:
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"无法读取深度图像: {depth_path}")
            return
        success, depth_bytes_encoded = cv2.imencode('.png', depth_image)
        depth_bytes = depth_bytes_encoded.tobytes()
        print(f'深度图像 {depth_path} 大小: {len(depth_bytes)} 字节')
    except Exception as e:
        print(f"读取深度图像时出错: {e}")
        return
    
    # --- 建立连接并发送数据 ---
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"正在连接到服务器 {server_host}:{server_port}...")
            s.connect((server_host, server_port))
            print("连接成功！")

            # --- 发送数据 ---
            # 发送JSON
            print("正在发送JSON数据...")
            send_msg(s, json_bytes)
            print("JSON数据发送完成。")
            # 发送RGB图像
            print("正在发送RGB图像数据...")
            send_msg(s, rgb_bytes)
            print("RGB图像数据发送完成。")
            # 发送深度图像
            print("正在发送深度图像数据...")
            send_msg(s, depth_bytes)
            print("深度图像数据发送完成。")
            print("所有数据发送完成，等待服务器响应...")

            # --- 接收响应 ---
            response_bytes = recv_msg(s)
            if response_bytes is None:
                print("未收到服务器响应。")
                return
            print(f"收到服务器响应")
            response_data = json.loads(response_bytes.decode('utf-8'))
            print(json.dumps(response_data, indent=2))
            if response_data.get('success'):
                print("抓取点计算成功！")
            else:
                print("抓取点计算失败。")
            
            pose_response = response_data.get('grasp_poses', {})

            grasp_pose = pose_response

            return grasp_pose
    except ConnectionAbortedError:
        print("连接被服务器中止。")
    except Exception as e:
        print(f"发生意外错误: {e}")

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = r.as_matrix()
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix

def transform_pose(pose, cam2base):

    x, y, z, qx, qy, qz, qw = pose

    # 创建旋转四元数对象
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()  # 3x3 旋转矩阵
    
    # 构造 4x4 齐次变换矩阵
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = [x, y, z]
    
    # 2. 使用 cam2base 变换
    new_pose_matrix = cam2base @ pose_matrix
    
    # 3. 从新的 4x4 矩阵中提取结果
    new_rotation_matrix = new_pose_matrix[:3, :3]
    new_translation = new_pose_matrix[:3, 3]
    
    # 转90度以适应机械臂问题
    r = np.array([[1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]])
    new_rotation_matrix = new_rotation_matrix @ r

    # 从旋转矩阵得到新的四元数
    new_rotation = R.from_matrix(new_rotation_matrix)
    new_quat = new_rotation.as_quat()  # 得到四元数 (qx, qy, qz, qw)
    
    # print(new_translation)
    # exit()
    # 返回新的 pose
    return np.hstack([new_translation, new_quat])

def get_pose(gg, retreat_distance=0.03):
    position = gg.translation
    rotation_matrix = gg.rotation_matrix

    # 法向方向是 rotation_matrix 的第三列（z 轴）
    approach_vector = rotation_matrix[:, 0]  # shape: (3,)
    
    # 沿法向方向后退 retreat_distance
    adjusted_position = position - approach_vector * retreat_distance

    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]

    pose = np.concatenate((adjusted_position, quaternion))
    return pose

class gg_data:
    def __init__(self):
        self.translation = None
        self.rotation_matrix = None

def grasp(cam2base_path, grasp_pose):
    print("--------------------------",grasp_pose)

    qx, qy, qz, qw = grasp_pose['orientation']
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()



    gg = gg_data()
    gg.translation = grasp_pose['position']
    gg.rotation_matrix = rotation_matrix

    print("--------------------------",gg.rotation_matrix)

    robot = R1Robot('r1')
    pose = robot.read_current_pose()
    x, y, z, roll, pitch, yaw = pose
    end2base = xyzrpy_to_matrix(x, y, z, roll, pitch, yaw)
    # exit()
    data = pd.read_csv(cam2base_path, header=None)
    cam2end = data.to_numpy()    
    cam2base = end2base @ cam2end
    # print(end2base)
    print(cam2base)
    print(end2base)
    # exit()
    # print(get_pose(gg[0]))
    # ipdb.set_trace()

    
    # pose_base_0 = transform_pose(get_pose(gg,retreat_distance=0.50), cam2base)
    pose_base_1 = transform_pose(get_pose(gg,retreat_distance=0.08), cam2base)
    pose_base_2 = transform_pose(get_pose(gg,retreat_distance=0.04), cam2base)
    print(pose)    
    print(pose_base_2)
    # pose_base[2] = pose_base[2] + 0.04      #防撞
    # exit()+
    print("*********************")
    print(robot.read_current_pose())
    # exit()
    # robot.set_endpose(pose_base_0)
    robot.set_endpose(pose_base_1)
    robot.set_endpose(pose_base_2)
    print("after grasp ee pose: \n", robot.read_current_pose())
    exit()
    # print(pose_base)`


if __name__ == "__main__":
    print("参数解析完成")
    if not os.path.isfile(cfgs.rgb_path):
        print(f"RGB图像文件不存在: {cfgs.rgb_path}")
        sys.exit(1)
    if not os.path.isfile(cfgs.depth_path):
        print(f"深度图像文件不存在: {cfgs.depth_path}")
        sys.exit(1)
    grasp_pose = run_grasp_client(cfgs.server_host, 
                     cfgs.server_port, 
                     cfgs.rgb_path, 
                     cfgs.depth_path, 
                     cfgs.text_prompt, 
                     cfgs.mode)
    cam2base_path = './cam2end_H.csv'
    grasp(cam2base_path, grasp_pose)