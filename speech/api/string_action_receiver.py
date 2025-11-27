#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import rospy
import rospkg
from std_msgs.msg import String


def run_sh(script_name, args=None, timeout=None):
    """调用 sh 脚本；优先使用 ~scripts_dir，其次 ~script_pkg；都没有时回退到当前文件同级目录。"""
    scripts_dir = rospy.get_param("~scripts_dir", "").strip()
    script_pkg = rospy.get_param("~script_pkg", "").strip()

    script_path = ""
    if scripts_dir:
        script_path = os.path.join(scripts_dir, script_name)
    elif script_pkg:
        rp = rospkg.RosPack()
        pkg_path = rp.get_path(script_pkg)
        script_path = os.path.join(pkg_path, "scripts", script_name)
    else:
        # 回退到本文件所在目录（通常也是 scripts 目录）
        here = os.path.dirname(os.path.realpath(__file__))
        script_path = os.path.join(here, script_name)

    if not os.path.exists(script_path):
        rospy.logerr("脚本不存在: %s", script_path)
        return False

    cmd = ["bash", script_path]
    if args:
        cmd.extend(args)

    rospy.loginfo("执行命令: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        rospy.loginfo("脚本执行成功: %s", script_name)
        return True
    except subprocess.CalledProcessError as e:
        rospy.logerr("脚本返回非零状态码: %s, code=%s", script_name, e.returncode)
    except subprocess.TimeoutExpired:
        rospy.logerr("脚本执行超时: %s", script_name)
    except Exception as e:
        rospy.logerr("脚本执行异常: %s, err=%s", script_name, str(e))
    return False

def handle_pick(param, timeout):
    if not param:
        rospy.logerr("pick 缺少 parameter 参数")
        return False
    return run_sh("pick.sh", [param, "pick-"+param], timeout=None)

def handle_place(param, timeout):
    # args = [param] if param else []
    return run_sh("place.sh", ["place-"+param], timeout=None)

def parse_json_command(text):
    """
    期望 JSON 形如:
      {"robot": "a", "action": "pick", "parameter": "red_bottle"}
    返回解析后的字典；若不是合法 JSON 或不是对象，返回 None 让上层回退到 KV 解析。
    """
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    key_map = {
        "param": "parameter",
        "params": "parameter",
        "target": "parameter",
        "object": "parameter",
        "obj": "parameter",
        "location": "parameter",
        "place_location": "parameter",
    }

    out = {"_raw": text}
    for k, v in data.items():
        key = str(k).strip().lower()
        key = key_map.get(key, key)
        # 仅对字符串做 trim/小写归一；其他类型（数字、布尔、对象）原样保留
        if isinstance(v, str):
            val = v.strip()
            if key in ("action", "robot"):
                val = val.lower()
        else:
            val = v
        out[key] = val
    return out

def callback(msg):
    raw = msg.data
    rospy.loginfo("收到原始指令: %s", raw)
    data = parse_json_command(raw) #使用json格式，直接读取成对应的str，str->json
    
    rospy.loginfo("解析后: %s", {k: v for k, v in data.items() if k != "_raw"})

    robot_name = data.get("robot", "") #后续添加robot_name 区分不同机器人
    action = data.get("action", "")
    param = data.get("parameter", "")

    timeout = float(360)

    if action == "pick":
        ok = handle_pick(param, timeout)
    elif action == "place":
        ok = handle_place(param, timeout)
    else:
        rospy.logerr("未知或缺失的 action: %s", action)
        ok = False

    if ok:
        rospy.loginfo("动作执行完成")
    else:
        rospy.logerr("动作执行失败")

def main():
    rospy.init_node("string_action_receiver", anonymous=False)
    # topic = rospy.get_param("~topic", "/llm_command")
    topic = "/llm_command"
    rospy.Subscriber(topic, String, callback, queue_size=10)
    rospy.loginfo("string_action_receiver 已启动，订阅主题: %s", topic)
    rospy.spin()

if __name__ == "__main__":
    main()