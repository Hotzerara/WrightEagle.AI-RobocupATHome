import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import time
from scipy.spatial.transform import Rotation as R
# --- SokcetBridgeNode 导入 ---
import socket 
import threading
import struct
import json
import os
import uuid

# --- 导入图像的处理 ---
import cv2
import numpy as np

# --- 导入服务接口 ---
from grasp_srv_interface.srv import TriggerGrasp

# --- 辅助函数 ---
def send_msg(sock, msg_bytes):
    try:
        # 发送消息长度
        msg_length = struct.pack('>Q', len(msg_bytes))
        sock.sendall(msg_length)
        # 发送消息内容
        sock.sendall(msg_bytes)
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"Socket发送消息失败: {e}")
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
        # len_header_bytes = recvall(sock,8)
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

class SocketBridgeNode(Node):
    def __init__(self):
        super().__init__('socket_bridge_node')
        self.get_logger().info('Socket Bridge 节点启动中')

        self.reentrant_group = ReentrantCallbackGroup()

        # 声明参数
        self.declare_parameter('image_save_path', '/root/host_home/ros2_ws/data/images')
        self.declare_parameter('socket_host', '0.0.0.0')
        self.declare_parameter('socket_port', 9090)

        # 获取参数
        self.image_save_path = self.get_parameter('image_save_path').get_parameter_value().string_value
        self.socket_host = self.get_parameter('socket_host').get_parameter_value().string_value
        self.socket_port = self.get_parameter('socket_port').get_parameter_value().integer_value

        try:
            if not os.path.exists(self.image_save_path):
                os.makedirs(self.image_save_path)
                self.get_logger().info(f'创建图像保存路径: {self.image_save_path}')
        except Exception as e:
            self.get_logger().error(f'创建图像保存路径失败: {e}')
            raise

        # 初始化 Ros2 服务客户端
        self.trigger_cli = self.create_client(TriggerGrasp, 
                                            'trigger_grasp_pipeline',
                                            callback_group=self.reentrant_group)
        while not self.trigger_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('TriggerGrasp 服务不可用，正在等待...')
        self.get_logger().info('TriggerGrasp 客户端已启动.')

        # 初始化 Socket 服务器 
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.socket_host, self.socket_port))
            self.server_socket.listen(5)
            self.get_logger().info(f'Socket 服务器已启动，监听 {self.socket_host}:{self.socket_port}')
        except Exception as e:
            self.get_logger().error(f'Socket 服务器绑定到 {self.socket_host}:{self.socket_port} 失败: {e}')
            raise
        self.socket_thread = threading.Thread(target=self._socket_server_loop)
        self.socket_thread.daemon = True
        self.socket_thread.start()
    
    def _socket_server_loop(self):
        """
        Socket 服务器主循环，接受客户端连接并处理图像接收和服务调用
        """
        # 无限循环接受客户端连接
        # 只要该ROS2节点在运行(rclpy.ok())，就继续接受连接
        while rclpy.ok():
            try:
                conn, addr = self.server_socket.accept()
                self.get_logger().info(f'接受到来自 {addr} 的连接')
                # 启动一个新线程处理该连接，以便原线程继续接受其他连接
                client_thread = threading.Thread(target=self._handle_client_connection, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
            except OSError as e:
                if rclpy.ok():
                    self.get_logger().error(f'Socket accept 连接失败: {e}')
                else:
                    pass
        self.get_logger().info('Socket 侦听线程正在关闭...')
        self.server_socket.close()

    def _handle_client_connection(self, conn, addr):
        """
        处理单个客户端连接
        1.（Socket）接收请求（JSON：text_prompt,mode; 图像数据：RGB+Depth）
        2. 创建唯一目录保存图像
        3. 调用 TriggerGrasp 服务
        4. 发送响应给客户端
        5. 关闭连接
        """
        self.get_logger().info(f'[客户端{addr}] 处理线程已启动')
        try:
            # --- 创建唯一的请求目录 ---
            # request_dir_name = f'request_{uuid.uuid4().hex}'
            # request_dir_path = os.path.join(self.image_save_path, request_dir_name)
            request_dir_path = self.image_save_path
            try:
                os.makedirs(request_dir_path, exist_ok=True)
                self.get_logger().info(f'[客户端{addr}] 创建请求目录: {request_dir_path}')
            except Exception as e:
                self.get_logger().error(f'[客户端{addr}] 创建请求目录失败: {e}')
                return
            
            # --- 接收JSON和图像数据 ---
            # 1.接收JSON请求
            self.get_logger().info(f'[客户端{addr}] 等待接收JSON数据...')
            request_json_bytes = recv_msg(conn)
            if request_json_bytes is None:
                self.get_logger().error(f'[客户端{addr}] 未接收到请求数据，连接可能已关闭')
                return
            # 2.接收RGB图像
            self.get_logger().info(f'[客户端{addr}] 等待接收RGB图像数据...')
            rgb_image_bytes = recv_msg(conn)
            if rgb_image_bytes is None:
                self.get_logger().error(f'[客户端{addr}] 未接收到RGB图像数据，连接可能已关闭')
                return
            # 3.接收Depth图像
            self.get_logger().info(f'[客户端{addr}] 等待接收Depth图像数据...')
            depth_image_bytes = recv_msg(conn)
            if depth_image_bytes is None:
                self.get_logger().error(f'[客户端{addr}] 未接收到Depth图像数据，连接可能已关闭')
                return
            
            # --- 处理和保存数据 ---
            # 1.解析JSON
            try:
                request_data = json.loads(request_json_bytes.decode('utf-8'))
                text_prompt = request_data.get('text_prompt', '')
                mode = request_data.get('mode', 0)
                self.get_logger().info(f'[客户端{addr}] JSON数据解析成功: text_prompt="{text_prompt}", mode={mode}')
            except Exception as e:
                self.get_logger().error(f'[客户端{addr}] 解析JSON数据失败: {e}')
                return
            # 2.保存RGB图像
            try:
                rgb_array = np.frombuffer(rgb_image_bytes, dtype=np.uint8)
                rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
                if rgb_image is None:
                    raise ValueError("解码RGB图像失败")
                rgb_image_path = os.path.join(request_dir_path, 'color.png')
                cv2.imwrite(rgb_image_path, rgb_image)
                self.get_logger().info(f'[客户端{addr}] RGB图像保存成功: {rgb_image_path}')
            except Exception as e:
                self.get_logger().error(f'[客户端{addr}] 保存RGB图像失败: {e}')
                return
            # 3.保存Depth图像
            try:
                depth_array = np.frombuffer(depth_image_bytes, dtype=np.uint8)
                depth_image = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    raise ValueError("解码Depth图像失败")
                depth_image_path = os.path.join(request_dir_path, 'depth.png')
                cv2.imwrite(depth_image_path, depth_image)
                self.get_logger().info(f'[客户端{addr}] Depth图像保存成功: {depth_image_path}')
            except Exception as e:
                self.get_logger().error(f'[客户端{addr}] 保存Depth图像失败: {e}')
                return
            
            # --- 调用 TriggerGrasp 服务 ---
            self.get_logger().info(f'[客户端{addr}] 准备调用 TriggerGrasp 服务...')
            trigger_request = TriggerGrasp.Request()
            trigger_request.input_path = request_dir_path
            trigger_request.text_prompt = text_prompt
            trigger_request.mode = mode

            try:
                ros_response = self.trigger_cli.call(trigger_request)
                self.get_logger().info(f'[客户端{addr}] 服务响应： success={ros_response.success}, message="{ros_response.message}"')
            except Exception as e:
                self.get_logger().error(f'[客户端{addr}] 调用 TriggerGrasp 服务失败: {e}')
                response_json_dict = {
                    'success': False,
                    'message': f'服务调用失败: {str(e)}'
                }
                response_bytes = json.dumps(response_json_dict).encode('utf-8')
                send_msg(conn, response_bytes)
                return
            # ros_future = self.trigger_cli.call_async(trigger_request)
            # while rclpy.ok() and not ros_future.done():
            #     time.sleep(0.1)
            #     print(ros_future.done())
            # if not ros_future.done():
            #     self.get_logger().error(f'[客户端{addr}] ROS2被关闭，TriggerGrasp 服务调用超时或失败')
            #     return
            # ros_response = ros_future.result()
            # self.get_logger().info(f'[客户端{addr}] 服务响应： success={ros_response.success}, message="{ros_response.message}"')

            # --- 构建并发送 Socket 响应 ---
            response_json_dict = {}
            if ros_response.success:
                p = ros_response.grasp_poses[0].position
                o = ros_response.grasp_poses[0].orientation
                
                pos = [p.x, p.y, p.z]
                quat = [o.x, o.y, o.z, o.w]
                # 构建成功的JSON字典
                response_json_dict = {
                    'success': True,
                    'message': ros_response.message,
                    'grasp_poses': {
                        'position': pos,
                        'orientation': quat
                    }
                }
            else:
                # 构建失败的JSON字典
                response_json_dict = {
                    'success': False,
                    'message': ros_response.message
                }
            response_bytes = json.dumps(response_json_dict).encode('utf-8')
            self.get_logger().info(f'[客户端{addr}] 发送响应数据...')
            send_msg(conn, response_bytes)
            self.get_logger().info(f'[客户端{addr}] 响应数据发送完成')

        except (ConnectionAbortedError, BrokenPipeError, OSError) as e:
            self.get_logger().warn(f'[客户端{addr}] 处理连接时发生异常: {e}')
        
        except Exception as e:
            self.get_logger().error(f'[客户端{addr}] 处理连接时发生意外错误: {e}')
        
        finally:
            self.get_logger().info(f'[客户端{addr}] 关闭连接')
            conn.close()
            self.get_logger().info(f'[客户端{addr}] 处理线程已结束')
    
def main(args=None):
    rclpy.init(args=args)
    socket_bridge_node = None
    executor = None
    try:
        socket_bridge_node = SocketBridgeNode()
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(socket_bridge_node)
        
        try:
            socket_bridge_node.get_logger().info('SocketBridgeNode 已启动并开始自旋 (多线程)...')
            # 使用多线程执行器 "自旋"
            executor.spin()
        finally:
            # (这个 finally 确保了 executor 在节点关闭前被关闭)
            socket_bridge_node.get_logger().info('正在关闭 Executor...')
            executor.shutdown()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'节点运行时发生错误: {e}')
    finally:
        socket_bridge_node.get_logger().info('正在关闭 Socket Bridge 节点...')
        socket_bridge_node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
        
            
