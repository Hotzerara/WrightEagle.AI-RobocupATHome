import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import open3d as o3d
import glob
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion
from grasp_srv_interface.srv import Graspnet             
from pathlib import Path
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 设置grapnetAPI地址
GraspNetDIR = '/root/host_home/ros2_ws/graspnet-baseline'
sys.path.append(os.path.join(GraspNetDIR, 'models'))
sys.path.append(os.path.join(GraspNetDIR, 'dataset'))
sys.path.append(os.path.join(GraspNetDIR, 'utils'))


#导入graspnet module
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image



#定义服务器节点
class GraspnetServer(Node):
    def __init__(self):
        super().__init__('graspnet_server')
        self.get_logger().info('Graspnet 服务节点启动中')

        #graspnet模型参数
        # 注册参数值
        self.declare_parameter('checkpoint_path', 
                               '/root/host_home/ros2_ws/graspnet-baseline/logs/log_kn/checkpoint.tar')
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('collision_thresh', -0.01)
        self.declare_parameter('voxel_size', 0.01)

        # 获取参数值
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.num_point = self.get_parameter('num_point').get_parameter_value().integer_value
        self.num_view = self.get_parameter('num_view').get_parameter_value().integer_value
        self.collision_thresh = self.get_parameter('collision_thresh').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
     
        #创建srv服务
        self.srv = self.create_service(
            Graspnet,
            'graspnet_service',
            self.handle_graspnet_request
        )
        self.get_logger().info('Graspnet 服务已启动，等待请求...')

        #加载计算设备
        self.get_logger().info(f"计算设备：{device}")
        print("CUDA版本:", torch.version.cuda)

        #graspnet模型加载#
        self.get_logger().info("正在加载Graspnet模型...")
        self.graspnet_net = self.graspnet_get_net()
        self.get_logger().info("Graspnet模型加载完成.")

    #grapsnet模型加载函数
    def graspnet_get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    
    def get_and_process_data(self,data_dir):
    # load data
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        with open(os.path.join(data_dir, 'camera.json')) as f:
            params = json.load(f)
        intrinsic = np.array(params['camera_matrix'])
        print(intrinsic)
        # factor_depth = meta['factor_depth']
        factor_depth = [[1000.]]
        print(factor_depth)

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        mask = mask.astype(bool)
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        print(len(cloud_masked))
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud_ = o3d.geometry.PointCloud()
        # cloud_.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))  #cloud_masked -> cloud_sampled
        # cloud_.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))  #cloud_masked -> cloud_sampled
        cloud_.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32)) 
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud_
    
    def find_ply_file(self, input_path):
        """
        在指定目录中查找.ply文件
        """
        ply_file = glob.glob(os.path.join(input_path, '*.ply'))
        if len(ply_file) == 0:
            self.get_logger().error(f"在目录 {input_path} 中未找到 .ply 文件.")
            return None
        return ply_file[0]
    
    def load_ply_as_np(self, ply_file):
        """
        加载.ply文件并转换为numpy数组
        """
        self.get_logger().info(f"正在加载点云文件: {ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        return points
        
    def process_point_cloud(self, points):
        """
        处理输入的numpy点云数据，进行采样并打包成模型所需的输入格式
        """
        # 采样点云
        self.get_logger().info("开始处理点云数据...")
        if len(points) >= self.num_point:
            idxs = np.random.choice(len(points), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(len(points), self.num_point-len(points), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = points[idxs]

        # 将数据类型转化为模型所需格式(end_points字典)
        end_points = dict()
        cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        end_points['point_clouds'] = cloud_sampled_torch.to(device)
        end_points['cloud_colors'] = None  # 如果有颜色信息，可以在这里添加

        #创建用于碰撞检测的open3d点云对象
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        self.get_logger().info("点云数据处理完成.")
        return end_points, cloud_o3d
    
    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self,gg, cloud):
        #print(gg)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0, 0, 0])
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers, coordinate_frame])

    def handle_graspnet_request(self, request, response):
        mode = request.mode
        # === gsam+vggt+graspnet模式 ===
        if mode == 1:
            self.get_logger().info('收到请求，正在处理(vggt点云输入)...')
            input_path = request.input_path
            ply_file = self.find_ply_file(input_path)
            input_cloud = self.load_ply_as_np(ply_file)
            end_points, cloud = self.process_point_cloud(input_cloud)
        # === gsam+graspnet模式 ===
        elif mode == 0:
            self.get_logger().info('收到请求，正在处理(gsam图片输入)...')
            input_path = request.input_path
            end_points, cloud = self.get_and_process_data(input_path)
        gg = self.get_grasps(self.graspnet_net, end_points)
        o3d.visualization.draw_geometries([cloud])
        #记录原始抓取姿态数量
        self.get_logger().info(f"原始抓取姿态数量: {len(gg)}")
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
            self.get_logger().info(f"碰撞检测后抓取姿态数量: {len(gg)}")
        if len(gg) == 0:
            response.grasp_poses = []
            response.success = False
            response.message = "未检测到有效抓取姿态，请调整输入点云或参数后重试."
            return response
        
        gg.nms()
        gg.sort_by_score()
        best_grasp = gg[0]
        print(best_grasp)

        R_align = np.array([
        [0, 0, 1],  # newX = oldZ
        [0, 1, 0],  # newY = oldY
        [-1, 0, 0],  # newZ = -oldX  (或者相当于 oldX = -newZ)
        ], dtype=float)

        best_grasp.rotation_matrix = best_grasp.rotation_matrix @ R_align
        #best_grasp.rotation_matrix = best_grasp.rotation_matrix

        pose_msg = Pose()
         
        #设置位置
        pose_msg.position = Point(
            x=float(best_grasp.translation[0]),
            y=float(best_grasp.translation[1]),
            z=float(best_grasp.translation[2])
        )
           
        #设置方向
        quat = R.from_matrix(best_grasp.rotation_matrix).as_quat()  # xyzw
        pose_msg.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3])  
        )
        
        self.vis_grasps(gg[:1], cloud)

        response.grasp_poses = [pose_msg]

        response.success = True
        response.message = f"成功检测到1个最佳抓取姿态."
        #准备响应数据       
        return response
    

def main(args=None):
    #初始化ros2客户端库
    rclpy.init(args=args)

    #创建服务器节点
    graspnet_server = GraspnetServer()

    try:
        rclpy.spin(graspnet_server)
    except KeyboardInterrupt:
        graspnet_server.get_logger().info('服务器节点被用户手动关闭...')
    finally:
        graspnet_server.get_logger().info('服务器节点已关闭.') 
        graspnet_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()