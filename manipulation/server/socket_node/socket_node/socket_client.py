import socket
import struct
import json
import cv2
import os
import numpy as np
import sys
import argparse

# --- 配置参数 ---
parser = argparse.ArgumentParser(description="Grasp Socket Client")
parser.add_argument('--server_host', default = '192.168.31.44', required=False, help='Server host address')
parser.add_argument('--server_port', default = 9090, type=int, required=False, help='Server port')
parser.add_argument('--rgb_path', required=True, help='Path to RGB image')
parser.add_argument('--depth_path', required=True, help='Path to Depth image')
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
        len_header_bytes = sock.recvall(sock,8)

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
            
            save_path = './grasp_result.json'        # 保存响应的文件路径
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)
                print(f"\n[状态] 响应已成功保存到: {save_path}")

            except IOError as e:
                print(f"!!! 错误: 无法将 JSON 写入文件 {save_path}: {e}")
            except Exception as e:
                print(f"!!! 发生意外错误: {e}")
            # pose_dict = response_data.get('grasp_poses')
            # grasp_pose = [
            #     pose_dict.get('x'),
            #     pose_dict.get('y'),
            #     pose_dict.get('z'),
            #     pose_dict.get('roll'),
            #     pose_dict.get('pitch'),
            #     pose_dict.get('yaw')
            # ]

            return grasp_pose
    except ConnectionAbortedError:
        print("连接被服务器中止。")
    except Exception as e:
        print(f"发生意外错误: {e}")

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