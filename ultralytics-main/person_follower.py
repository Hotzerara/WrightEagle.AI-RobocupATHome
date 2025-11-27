# python
import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import math

class PersonFollower: 
    def __init__(self): 
        rospy.init_node('person_follower_node') 

        # === 参数 ===
        self.goal_update_distance = rospy.get_param('~goal_update_distance', 0.6) 
        self.follow_distance = rospy.get_param('~follow_distance', 1.5) 
        self.global_frame = rospy.get_param('~global_frame', 'map') 

        # 当机器人距离桌子小于该半径时，停止跟随人，进入“去桌子模式” 
        self.table_stop_distance = rospy.get_param('~table_stop_distance', 1.5) 

        # 在“去桌子模式”下重复发送桌子目标的时间间隔（秒） 
        self.table_goal_interval = rospy.get_param('~table_goal_interval', 1.0) 

        # === 新增：桌子模式下静止检测参数 ===
        # 机器人在桌子模式下，如果连续 idle_timeout 秒内位移都小于 idle_move_threshold，则退出程序
        self.idle_timeout = rospy.get_param('~idle_timeout', 15)          # 静止时间阈值（秒）
        self.idle_move_threshold = rospy.get_param('~idle_move_threshold', 0.02)  # 允许的位移（米）

        # 记录进入桌子模式时的时间和位置
        self.idle_start_time = None           # rospy.Time
        self.idle_start_position = None       # (x, y)

        # === TF2 监听器 ===
        self.tf_buffer = tf2_ros.Buffer() 
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 

        # === 发布与订阅 ===
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1) 
        self.marker_pub = rospy.Publisher('/person_marker', Marker, queue_size=1) 

        # 关键：等待 /move_base_simple/goal 的订阅者准备好
        rospy.loginfo("等待 /move_base_simple/goal 的订阅者...") 
        while self.goal_pub.get_num_connections() == 0 and not rospy.is_shutdown(): 
            rospy.sleep(0.1) 
        rospy.loginfo("move_base 已连接。") 

        rospy.Subscriber('/person/base_link_3d_position', PointStamped, self.person_callback) 
        self.odom_sub = rospy.Subscriber('/local_odom', Odometry, self.odom_callback, queue_size=1) 

        # === 变量 ===
        # 上一次人物跟随目标位置
        self.last_goal_position = None

        # 当前机器人位姿（来自 /local_odom） 
        self.robot_pose = None

        # 是否已经进入“前往桌子”的模式（True 后不再跟随人） 
        self.final_goal_sent = False

        # 上一次发送桌子目标的时间，用于控制发送频率
        self.last_table_goal_time = rospy.Time(0) 
        from tf.transformations import euler_from_quaternion, quaternion_from_euler
        # === 预设桌子最终目标点 ===
        self.table_goal = PoseStamped() 
        _, _, yaw = euler_from_quaternion([0.1556566, 0.1422990, -0.7060529, 0.676026104]) 
        q = quaternion_from_euler(0.0, 0.0, yaw) 
        self.table_goal.header.frame_id = self.global_frame
        self.table_goal.pose.position.x = -5.0037099
        self.table_goal.pose.position.y = 0.868781
        self.table_goal.pose.position.z = 0.0
        self.table_goal.pose.orientation.x = q[0] 
        self.table_goal.pose.orientation.y = q[1] 
        self.table_goal.pose.orientation.z = q[2] 
        self.table_goal.pose.orientation.w = q[3] 

        rospy.loginfo("人物跟随节点已启动，带重复桌子终点逻辑与静止退出逻辑。") 

    # ========== 里程计回调，更新机器人当前位姿 ==========
    def odom_callback(self, odom_msg): 
        # 只保存平面位置与朝向, 这里直接存整个 Pose
        self.robot_pose = odom_msg.pose.pose

        # 每次收到里程计时检查是否接近桌子
        self.check_and_send_table_goal() 

        # 新增：在桌子模式下检测是否长时间静止
        self.check_idle_and_shutdown()

    # ========== 检查是否需要切换到桌子终点，并重复发送目标 ==========
    def check_and_send_table_goal(self): 
        # 需要有当前机器人位姿
        if self.robot_pose is None: 
            return

        # 计算机器人和桌子在平面上的距离
        dx = self.table_goal.pose.position.x - self.robot_pose.position.x
        dy = self.table_goal.pose.position.y - self.robot_pose.position.y
        dist = math.sqrt(dx * dx + dy * dy) 

        # 当距离小于等于阈值时，进入或保持“前往桌子模式” 
        if dist <= self.table_stop_distance: 
            # 第一次进入桌子范围：切换模式，停止跟随人
            if not self.final_goal_sent: 
                rospy.loginfo("接近桌子 %.2f m，切换到前往桌子的模式，停止跟随人。", dist) 
                self.final_goal_sent = True

                # 新增：刚进入桌子模式时，初始化静止检测
                if self.robot_pose is not None:
                    self.idle_start_time = rospy.Time.now()
                    self.idle_start_position = (
                        self.robot_pose.position.x,
                        self.robot_pose.position.y
                    )
                    rospy.loginfo("开始桌子模式静止监测。")

            # 控制桌子目标重复发送的频率
            now = rospy.Time.now() 
            if (now - self.last_table_goal_time).to_sec() >= self.table_goal_interval: 
                self.table_goal.header.stamp = now
                self.goal_pub.publish(self.table_goal) 
                self.last_table_goal_time = now
                rospy.loginfo("在桌子范围内，重复发送桌子目标。距离：%.2f m", dist) 

        # 如果希望离开桌子范围后恢复人物跟随，可以打开下面逻辑
        # else: 
        #     if self.final_goal_sent: 
        #         rospy.loginfo("已离开桌子范围，恢复人物跟随模式。") 
        #     self.final_goal_sent = False
        #     # 离开桌子模式，重置静止检测
        #     self.idle_start_time = None
        #     self.idle_start_position = None

    # ========== 新增：检测机器人在桌子模式下是否长时间静止 ==========
    def check_idle_and_shutdown(self):
        # 只有在桌子模式下才检查
        if not self.final_goal_sent:
            return

        if self.robot_pose is None:
            return

        # 若还没初始化静止检测，先初始化（防止极端情况漏掉）
        if self.idle_start_time is None or self.idle_start_position is None:
            self.idle_start_time = rospy.Time.now()
            self.idle_start_position = (
                self.robot_pose.position.x,
                self.robot_pose.position.y
            )
            return

        # 当前位移
        cur_x = self.robot_pose.position.x
        cur_y = self.robot_pose.position.y
        dx = cur_x - self.idle_start_position[0]
        dy = cur_y - self.idle_start_position[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # 若移动距离大于阈值，说明又动了，重新计时
        if dist > self.idle_move_threshold:
            self.idle_start_time = rospy.Time.now()
            self.idle_start_position = (cur_x, cur_y)
            return

        # 否则检查时间是否超过静止阈值
        elapsed = (rospy.Time.now() - self.idle_start_time).to_sec()
        if elapsed >= self.idle_timeout:
            rospy.loginfo("机器人在桌子模式下已静止 %.1f 秒(位移 %.3f m)，准备关闭节点。", elapsed, dist)
            rospy.signal_shutdown("Robot idle at table for too long")

    # ========== 人物位置回调 ==========
    def person_callback(self, point_stamped_msg): 
        # 若已经进入“前往桌子”的模式，则直接忽略人物位置，不再跟随
        if self.final_goal_sent: 
            return

        # 1. 坐标变换：将 base_link 下的点转换到 map 坐标系
        try: 
            transform = self.tf_buffer.lookup_transform( 
                self.global_frame, 
                point_stamped_msg.header.frame_id, 
                rospy.Time(0), 
                rospy.Duration(1.0) 
            ) 
            person_point_map = tf2_geometry_msgs.do_transform_point(point_stamped_msg, transform) 
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e: 
            rospy.logwarn("坐标变换失败: %s", e) 
            return

        # 2. 计算与上一个目标的距离，决定是否更新目标
        if self.last_goal_position: 
            dist = math.sqrt( 
                (person_point_map.point.x - self.last_goal_position.x) ** 2 +
                (person_point_map.point.y - self.last_goal_position.y) ** 2
            ) 
            if dist < self.goal_update_distance: 
                # 移动距离太小，不更新目标，但更新 Rviz 显示
                self.publish_person_marker(person_point_map.point) 
                return

        # 3. 生成新的导航目标 PoseStamped
        goal_msg = PoseStamped() 
        goal_msg.header.frame_id = self.global_frame
        goal_msg.header.stamp = rospy.Time.now() 

        try: 
            # 获取机器人在全局坐标系中的位置
            robot_transform = self.tf_buffer.lookup_transform( 
                self.global_frame, 'base_link', rospy.Time(0) 
            ) 
            robot_pos = robot_transform.transform.translation

            # 计算人到机器人的方向向量
            dx = person_point_map.point.x - robot_pos.x
            dy = person_point_map.point.y - robot_pos.y
            angle_to_person = math.atan2(dy, dx) 

            # 在人的位置基础上，沿着“人-机器人”反方向后退 follow_distance 的距离
            goal_x = person_point_map.point.x - self.follow_distance * math.cos(angle_to_person) 
            goal_y = person_point_map.point.y - self.follow_distance * math.sin(angle_to_person) 

            goal_msg.pose.position.x = goal_x
            goal_msg.pose.position.y = goal_y
            goal_msg.pose.position.z = 0.0

            # 让机器人朝向人的方向
            q = tf.transformations.quaternion_from_euler(0, 0, angle_to_person) 
            goal_msg.pose.orientation.x = q[0] 
            goal_msg.pose.orientation.y = q[1] 
            goal_msg.pose.orientation.z = q[2] 
            goal_msg.pose.orientation.w = q[3] 

        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e: 
            rospy.logwarn("获取机器人位置失败: %s, 将直接使用人的位置作为目标", e) 
            goal_msg.pose.position = person_point_map.point

        # 4. 发布目标并更新状态
        self.goal_pub.publish(goal_msg) 
        self.last_goal_position = goal_msg.pose.position
        rospy.loginfo("已更新跟随导航目标至: (%.2f, %.2f)", 
                      goal_msg.pose.position.x, goal_msg.pose.position.y) 

        # 5. 发布可视化标记
        self.publish_person_marker(person_point_map.point) 

    # ========== RViz 可视化人物标记 ==========
    def publish_person_marker(self, position): 
        marker = Marker() 
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now() 
        marker.ns = "person" 
        marker.id = 0
        marker.type = Marker.CYLINDER  # 使用圆柱体模拟人
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4  # 直径
        marker.scale.y = 0.4
        marker.scale.z = 1.5  # 高度
        marker.color.a = 1.0  # 不透明
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # 绿色
        marker.lifetime = rospy.Duration(1.0)  # 标记持续 1 秒
        self.marker_pub.publish(marker) 

    def run(self): 
        rospy.spin() 

if __name__ == '__main__': 
    try: 
        follower = PersonFollower() 
        follower.run() 
    except rospy.ROSInterruptException: 
        pass