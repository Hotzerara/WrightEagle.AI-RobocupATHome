import sys
from grasp_srv_interface.srv import Graspnet, GroundedSam, Vggt, Move, TriggerGrasp                                            
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
import time 

class GraspServerNode(Node):
    def __init__(self):
        super().__init__('grasp_client')
        self.get_logger().info('Grasp 客户端节点启动中')

        self.reentrant_group = ReentrantCallbackGroup()

        ############################################################
        # --- 内部服务客户端，负责调用graspnet、groundedsam、vggt服务 ---
        ############################################################

        #创建groundedsam_cli服务
        self.gsam_cli = self.create_client(GroundedSam, 'grounded_sam_service', 
                                           callback_group=self.reentrant_group)
        while not self.gsam_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('服务不可用，正在等待...')
        self.get_logger().info('GroundedSam 客户端已启动.')

        #创建graphnet_cli服务
        self.gspnet_cli = self.create_client(Graspnet, 'graspnet_service', 
                                             callback_group=self.reentrant_group)
        while not self.gspnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('服务不可用，正在等待...')
        self.get_logger().info('Graspnet 客户端已启动.')

        # #创建vggt_cli服务
        # self.vggt_cli = self.create_client(Vggt, 'vggt_service')
        # while not self.vggt_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('服务不可用，正在等待...')
        # self.get_logger().info('Vggt 客户端已启动.')

        # #创建 move_cli服务
        # self.move_cli = self.create_client(Move, 'move')
        # while not self.move_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Move 服务不可用，正在等待...')
        # self.get_logger().info('Move 客户端已启动.')

        #############################################
        # --- 外部服务服务端，负责接收指令开始grasp流程 ---
        #############################################
        self.trigger_service = self.create_service(TriggerGrasp, 
                                                   'trigger_grasp_pipeline', 
                                                   self.trigger_grasp_callback,
                                                   callback_group=self.reentrant_group)
        self.get_logger().info('TriggerGrasp 服务已启动，等待请求...')

    def call_grounded_sam(self,input_path=None, text_prompt=None, mode=0):
        self.get_logger().info('准备调用 GroundedSam 服务...')
        #创建请求
        gsam_req = GroundedSam.Request()
        #设置请求参数
        self.get_logger().info('开始设置请求参数')
        gsam_req.input_path = input_path
        gsam_req.text_prompt = text_prompt
        gsam_req.mode = mode
        self.get_logger().info(f'请求参数已设置: input_path={gsam_req.input_path}, text_prompt={gsam_req.text_prompt}')
        self.get_logger().info('正在发送请求...')
        try:
            response = self.gsam_cli.call(gsam_req)
            self.get_logger().info('收到GroundedSam响应')
            return response
        except Exception as e:
            self.get_logger().error(f'调用 GroundedSam 服务时出错: {e}')
            return None
    
    def call_vggt(self,input_path=None):
        self.get_logger().info("准备调用 Vggt 服务...")
        #创建请求
        vggt_req = Vggt.Request()
        #设置请求参数
        self.get_logger().info("开始设置请求参数")
        vggt_req.input_path = input_path
        self.get_logger().info(f"请求参数已设置: input_path={vggt_req.input_path}")
        self.get_logger().info("正在发送请求...")
        try:
            response = self.vggt_cli.call(vggt_req)
            self.get_logger().info("收到Vggt响应")
            return response
        except Exception as e:
            self.get_logger().error(f"调用 Vggt 服务时出错: {e}")
            return None
    
    def call_graspnet(self, input_path=None, mode=0):
        self.get_logger().info('准备调用 Graspnet 服务...')
        #创建请求
        gspnet_req = Graspnet.Request()
        #设置请求参数
        self.get_logger().info('开始设置请求参数')
        gspnet_req.input_path = input_path
        gspnet_req.mode = mode     
        self.get_logger().info(f'请求参数已设置: input_path={gspnet_req.input_path}')
        self.get_logger().info('正在发送请求...')
        try:
            response = self.gspnet_cli.call(gspnet_req)
            self.get_logger().info('收到Graspnet响应')
            return response
        except Exception as e:
            self.get_logger().error(f'调用 Graspnet 服务时出错: {e}')
            return None
    
    def call_move(self, target_pose=None):
        self.get_logger().info('准备调用 Move 服务...')
        # 创建请求
        move_req = Move.Request()
        # 设置请求参数
        move_req.mode = 0  # 设置为GSAM图片输入模式
        self.get_logger().info('正在发送异步请求...')
        move_req_future = self.move_cli.call_async(move_req)
        self.get_logger().info('等待服务响应...')
        rclpy.spin_until_future_complete(self, move_req_future)
        self.get_logger().info('收到move响应')
        return move_req_future.result()
        
    def send_request(self,input_path=None, text_prompt=None, mode=0):
        # mode = self.mode
        print("------")
        # 根据模式调用不同的服务组合
        # --- mode 0: GroundedSam + Graspnet ---
        # --- mode 1: GroundedSam + Vggt + Graspnet ---

        #########################################
        # --- mode 0: GroundedSam + Graspnet ---
        #########################################
        if mode == 0:
             #调用groundedsam服务
            self.get_logger().info(f'准备调用 GroundedSam 服务 with input_path={input_path}, text_prompt={text_prompt}, mode={mode}')
            gsam_response = self.call_grounded_sam(input_path=input_path,
                                                    text_prompt=text_prompt,
                                                    mode=mode)
            if not gsam_response or not gsam_response.success:
                self.get_logger().error('GroundedSam 服务调用失败或未成功执行.')
                return None
            self.get_logger().info('GroundedSam 服务调用成功')

            #调用graspnet服务
            self.get_logger().info(f'准备调用 Graspnet 服务 with input_path={gsam_response.output_path}, mode={mode}')
            gspnet_response = self.call_graspnet(gsam_response.output_path, mode=mode)
            if not gspnet_response:
                self.get_logger().error('Graspnet 服务调用失败或未成功执行.')
                return None
            self.get_logger().info('Graspnet 服务调用成功.') 
            return gspnet_response
        
        ###############################################
        # --- mode 1: GroundedSam + Vggt + Graspnet ---
        ###############################################
        elif mode ==1:
            #调用groundedsam服务
            self.get_logger().info(f'准备调用 GroundedSam 服务 with input_path={input_path}, text_prompt={text_prompt}, mode={mode}')
            gsam_response = self.call_grounded_sam(input_path=input_path,
                                                    text_prompt=text_prompt,
                                                    mode=mode)
            if not gsam_response or not gsam_response.success:
                self.get_logger().error('GroundedSam 服务调用失败或未成功执行.')
                return None
            self.get_logger().info('GroundedSam 服务调用成功')
            
            #调用vggt服务
            self.get_logger().info(f"准备调用 Vggt 服务 with input_path={gsam_response.output_path}")
            vggt_response = self.call_vggt(gsam_response.output_path)
            if not vggt_response or not vggt_response.success:
                self.get_logger().error('Vggt 服务调用失败或未成功执行.')
                return None
            self.get_logger().info('Vggt 服务调用成功.')

            #调用graspnet服务
            self.get_logger().info(f'准备调用 Graspnet 服务 with input_path={vggt_response.output_path}, mode={mode}')
            gspnet_response = self.call_graspnet(input_path=gsam_response.output_path, 
            mode=mode)
            if not gspnet_response:
                self.get_logger().error('Graspnet 服务调用失败或未成功执行.')
                return None
            self.get_logger().info('Graspnet 服务调用成功.') 
            return gspnet_response
        
    def trigger_grasp_callback(self, request, response):
        """
        当'trigger_grasp_pipeline'服务被调用时触发此回调函数
        
        Args:
            request (grasp_srv_interface.srv.TriggerGrasp.Request): 
                服务请求，包含触发抓取流程的参数
                - input_path (str): 输入图像路径
                - text_prompt (str): 文本提示
                - mode (uint8): 抓取模式选择
            

            response (grasp_srv_interface.srv.TriggerGrasp.Response): 
                服务响应，包含抓取结果
                - success (bool): 抓取是否成功
                - message (str): 抓取结果信息
                - grasp_poses (list of Pose): 抓取位姿列表

        Returns:
            grasp_srv_interface.srv.TriggerGrasp.Response:
            - success (bool): 抓取是否成功
            - message (str): 抓取结果信息
            - grasp_poses (list of Pose): 抓取位姿列表
        """
        self.get_logger().info(f'收到 TriggerGrasp 服务请求: path={request.input_path}, '
                               f'prompt={request.text_prompt}, '
                               f'mode={request.mode}, '
                               '\n开始执行抓取流程...')
        # 有待完善，现在暂时使用外部请求地址，后面继承move capture后应当调整为使用capture的图像路径
        response_req = self.send_request(input_path=request.input_path,
                                     text_prompt=request.text_prompt,
                                     mode=request.mode)
        # 填充并返回响应
        # print(response)
        if response_req and response_req.success:
            self.get_logger().info('视觉处理执行成功,正在返回位姿...')
            response.success = True
            response.message = "视觉Pipeline执行成功"
            response.grasp_poses = response_req.grasp_poses
            # print(response)
        else:
            self.get_logger().error('视觉处理执行失败,无法返回位姿.')
            response.success = False
            response.message = "视觉Pipeline执行失败,未有抓取位姿"
        print("返回response")
        return response
        
def main(args=None):
    rclpy.init(args=args)
    grasp_server_node = GraspServerNode()
    #创建多线程执行器
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(grasp_server_node)

    try:
        grasp_server_node.get_logger().info('GraspSergverNode 正在运行，等待服务请求...')
        executor.spin()
    except KeyboardInterrupt:
        grasp_server_node.get_logger().info('GraspServerNode 被用户手动关闭...')
    except Exception as e:
        grasp_server_node.get_logger().error(f'GraspServerNode 运行时发生错误: {e}')
    finally:
        grasp_server_node.get_logger().info('GraspServerNode 正在关闭...')
        executor.shutdown()
        grasp_server_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()