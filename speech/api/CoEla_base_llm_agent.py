'''
Author: wenhao wenhaoyu@mail.ustc.edu.cn
Date: 2024-07-26 14:01:15
LastEditors: wenhao wenhaoyu@mail.ustc.edu.cn
LastEditTime: 2025-07-22 13:08:49
FilePath: /BestMan_Pybullet/Examples/tro_llm_agent.py
'''

import os
import re
import math
import json
import argparse
import threading
import pybullet as p
import numpy as np
import copy
import yaml
from openai import OpenAI
from RoboticsToolBox import Bestman_sim_ur5e_vacuum_long, Pose, Bestman_sim_panda, ur5e, LLM_agent
from Env import Client
from Visualization import Visualizer
from Motion_Planning.Manipulation import OMPL_Planner
from Motion_Planning.Navigation import *
from SLAM import simple_slam
from Utils import load_config
from string import Template
from datetime import datetime
from collections import deque, defaultdict

stop_event = threading.Event()
member_sumplan = defaultdict(dict)
leader_sumplan = defaultdict(dict)
part_success = [0.0]
to_save_task_state = [""]
select_leader_failed = [False]
RT_task_state = ""
condition = threading.Condition()
threads_in_step = 0       # 当前 step 的线程总数
threads_arrived = 0       # 已到达同步点的线程数
barrier_broken = False    # 障碍物破坏
openai_client = OpenAI(
    api_key="sk-39w3O4ogdA9dGpJWTFfDQC97w1k3sPkVJuV9VYxXIPU14idf",
    base_url="https://api.claude-plus.top/v1",
    )

name_mapping = {
    "mobile_manipulation": "Alice",
    "manipulation": "Bob",
    "mobile": "David",
    "drone": "Lucy",
}

def prompt_once_wo_func(system_prompt, user_prompt, LLM_type):
    completion = openai_client.chat.completions.create(
    model = LLM_type,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature = 0.5, # 0.5
    )
    return completion.choices[0].message.content

def prompt_once(system_prompt, user_prompt, LLM_type, func_call_format=None):
    completion = openai_client.chat.completions.create(
    model = LLM_type,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    functions=func_call_format,
    function_call="auto",
    temperature = 0.5, # 0.5
    )

    msg = completion.choices[0].message

    if getattr(msg, "function_call", None):
        return msg.function_call.arguments
    else:
        return msg.content

def extract_instruct(response):
    action, object_name, stand_pose, place_where, delta_x, delta_y, content, role = None, None, None, None, None, None, None, None
    # pattern = r'\s*(\w+)\((")?(\w+)(")?\)'
    pattern = r'\s*(\w+)\(\s*([\'"])?(\w+)\2?\s*\)'
    command = re.search(pattern, response.strip())
    if command == None:
        # pattern = r'\s*(\w+)\((")?(\w+)(")?,\s*(")?(\w+)(")?\)'
        pattern = r'\s*(\w+)\(\s*([\'"])?([^\'",]+)\2?\s*,\s*([\'"])?([^\'")]+)\4?\s*\)'
        command = re.search(pattern, response.strip())
    if command == None:
        pattern = r"\s*(\w+)\(((-)?\d+(\.\d+)?),\s*((-)?\d+(\.\d+)?)\)"
        command = re.search(pattern, response.strip())
    if command == None: # wait()
        pattern = r'\s*(\w+)\(\)'
        command = re.search(pattern, response.strip())
    if command == None:
        # pattern = r'\s*(\w+)\((")?([^"]+)(")?,\s*(")?(\w+)(")?\)'
        pattern = r"\s*(\w+)\(\s*(?:(['\"])(.*?)\2|(\w+))\s*,\s*(?:(['\"])(.*?)\5|(\w+))\s*\)"
        command = re.search(pattern, response.strip())
    # if command == None:
    #     pattern = r"\s*(\w+)\(\s*(?:(['\"])(.*?)\2|(\w+))\s*,\s*(?:(['\"])(.*?)\5|(\w+))\s*\)"
    #     true_response = ast.literal_eval(response.strip())
    #     command = re.search(pattern, true_response)
    
    if command == None:
        action = "wait"
        print("Debug: ###########command is None###########")
    else:
        action = command.group(1)
    if action == "move":
        delta_x = command.group(3)
        delta_y = command.group(5)
    elif action == "wait":
        print("wait")
    elif action == "place":
        object_name = command.group(3)
        place_where = command.group(5)
    elif action == "navigate":
        object_name = command.group(3)
        stand_pose = command.group(5)
    elif action == "communicate":
        response_json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if not response_json_match:
            raise ValueError("Failed to find legitimate JSON content")
        response_json_str = response_json_match.group(0)
        response_data = json.loads(response_json_str)
        response_contents = response_data.get("contents")
        match = re.search(r"communicate\((.*)\)", response_contents, re.DOTALL)
        if match:
            args_string = match.group(1)
            content = args_string
            role = []
            for robo_role in ["Alice", "Bob", "David", "Lucy"]:
                if robo_role in args_string:
                    role.append(robo_role)
        # content = command.group(3)
        # if len(command.groups()) == 5:
        #     role = command.group(5)
        # else:
        #     role =command.group(6)
    else:
        object_name = command.group(3)
    return [action, object_name, stand_pose, place_where, delta_x, delta_y, content, role]
# Alice 0
def mobile_manipulation_action(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, moma_agent, scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas):
    print(f"##################mobile_manipulation_{step}##################")

    # add task objective change to the user prompt
    if is_change == "CTO" and step >= 5:
        prefix = "Please note that the overall task objective has changed! "
        collaboration_user_prompt["mobile_manipulation"] = prefix + collaboration_user_prompt["mobile_manipulation"]

    if is_change == "ANC" and step >= 5:
        prefix = "Please note that new collaborators have joined the team! "
        collaboration_user_prompt["mobile_manipulation"] = prefix + collaboration_user_prompt["mobile_manipulation"]
    
    response = prompt_once_wo_func(collaboration_system_prompt["mobile_manipulation"], collaboration_user_prompt["mobile_manipulation"], LLM_type)
    
    tosave = {
        "moma_system_prompt": collaboration_system_prompt["mobile_manipulation"],
        "moma_user_prompt": collaboration_user_prompt["mobile_manipulation"],
        "moma_response": response.strip()
    }

    response_filepath = save_filepath + '/heterogeneous_collaboration/mobile_manipulation'
    os.makedirs(response_filepath, exist_ok=True)
    response_filename = os.path.join(response_filepath, f'log_step_{step}.json')
    with open(response_filename, 'w', encoding='utf-8') as json_file:
        json.dump(tosave, json_file, ensure_ascii=False, indent=4)

    

    instruct = extract_instruct(response)
    action, object_name, stand_pose, place_where, delta_x, delta_y, content, role = instruct[0], instruct[1], instruct[2], instruct[3], instruct[4], instruct[5], instruct[6], instruct[7]
    if action != "wait":
        collaboration_action_steps["mobile_manipulation"][0] += 1

    if action == "communicate":
        collaboration_communication_steps["mobile_manipulation"][0] += 1
    
    if action == "navigate":
        irz_flag = False
        if is_change == "IRZ" and step >= 5:
            if "moma" in restricted_areas.keys():
                if object_name in layout[restricted_areas["moma"]]:
                    is_success = False
                    irz_flag = True
                    action_feedback = f"The object {object_name} is in the restricted area {restricted_areas['moma']}, which is not accessible for you. Please entrust other collaborators to conduct exploration."
        if not irz_flag:
            is_success, path_len, action_feedback = moma_agent.navigate(object_name, stand_pose, scene_graph, object_graph)
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + object_name + " , " + stand_pose + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "open":
        is_success, action_feedback = moma_agent.open(object_name, scene_graph, object_graph)
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + object_name + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "pick":
        is_success, action_feedback = moma_agent.pick(object_name, scene_graph, object_graph)
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + object_name + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "move":
        move_status = True
        if moma_agent.to_pick_obj != None:
            submap, min_x, min_y = moma_agent.get_local_costmap(moma_agent.to_pick_obj)
            start_position = next((key for key, value in submap.items() if value == 2), None)
            target_position = next((key for key, value in submap.items() if value == 3), None)
            with open(f'./prompts/{moma_agent.task_type}/collaboration/move_system_prompt.json', 'r') as file:
                move_system_prompt = json.load(file)
            move_sys_info = {
                'to_pick_obj': moma_agent.to_pick_obj,
                'start_position': start_position,
                'target_position': target_position
            }
            move_system_prompt = Template(move_system_prompt).substitute(move_sys_info)
            move_user_prompt = f"The local costmap is as follows: {submap}"
            response = prompt_once_wo_func(move_system_prompt, move_user_prompt, LLM_type)

            tosave = {
                "move_system_prompt": move_system_prompt,
                "move_user_prompt": move_user_prompt,
                "move_response": response.strip()
            }

            response_filepath = save_filepath + '/heterogeneous_collaboration/mobile_manipulation/move'
            os.makedirs(response_filepath, exist_ok=True)
            response_filename = os.path.join(response_filepath, f'log_step_{step}.json')
            with open(response_filename, 'w', encoding='utf-8') as json_file:
                json.dump(tosave, json_file, ensure_ascii=False, indent=4)

            # target_tuple = tuple(map(int, re.findall(r'\d+', json.loads(response)["contents"])))
            # target_tuple = list(target_tuple)
            # target_tuple[0] += min_x
            # target_tuple[1] += min_y
            # delta_x, delta_y = moma_agent.get_delta_move(target_tuple)

            try:
                response_data = json.loads(response)
                contents = response_data.get("contents")

                if contents is None:
                    raise ValueError("contents is None")

                numbers = re.findall(r'\d+', contents)
                if not numbers:
                    raise ValueError("No numeric values found in contents")

                target_tuple = list(map(int, numbers))
                
                if len(target_tuple) < 2:
                    raise ValueError("Expected at least two coordinates in contents")

                target_tuple[0] += min_x
                target_tuple[1] += min_y

                delta_x, delta_y = moma_agent.get_delta_move(tuple(target_tuple))

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                action_feedback = "The move failed. A suitable target point for moving could not be found."
                move_status = False

        if move_status:
            is_success, path_len, action_feedback = moma_agent.move(float(delta_x), float(delta_y))
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + str(delta_x) + " , " + str(delta_y) + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "wait":
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # action_feedback = ""
        # # mobile manipulation history feedback
        # collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        # collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # # mobile manipulation history action
        # collaboration_history_action["mobile_manipulation"].append(action + "()")
        # collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "place":
        is_success, action_feedback = moma_agent.place(object_name, place_where, scene_graph, object_graph)
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + object_name + " , " + place_where + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
    elif action == "communicate":
        if name_mapping["mobile_manipulation"] in role:
            role.remove(name_mapping["mobile_manipulation"])
        robot_info = moma_agent.get_robot_info()
        collaboration_user_prompt["mobile_manipulation"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        action_feedback = f"The message has been successfully sent to {role}"
        # mobile manipulation history feedback
        collaboration_history_feedback["mobile_manipulation"].append(action_feedback)
        collaboration_user_prompt["mobile_manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile_manipulation"]))
        # mobile manipulation history action
        collaboration_history_action["mobile_manipulation"].append(action + "(" + content + " , " + str(role) + ")")
        collaboration_user_prompt["mobile_manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile_manipulation"]))
        # mobile manipulation receive message
        collaboration_history_receive_message["mobile_manipulation"].append(f"[I said to {role}: {content}]")
        for robo_role in role:
            role_key = next((key for key, val in name_mapping.items() if val == robo_role), None)
            if role_key is not None:
                collaboration_history_receive_message[role_key].append("[" + content + "]" + " from Alice. ")
            else:
                print("##### Invalid Role! #####")
    else:
        print(f"ToDo!!!-{action}")

    print("The mobile manipulation is waiting...")
    sync_point()
    
    collaboration_user_prompt["mobile_manipulation"] += "The historical receive messages, from oldest to newest, are as follows: " + str(list(collaboration_history_receive_message["mobile_manipulation"]))

    # RT_task_state = monitor_task_completion(objects_list, moma_agent, object_graph)
    collaboration_user_prompt["mobile_manipulation"] += RT_task_state

    # to_save_task_state[0] = RT_task_state

    if stop_event.is_set():
        return
        

# David 2
def mobile_action(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, mo_agent, scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas):
    print(f"##################mobile_{step}##################")

    # add task objective change to the user prompt
    if is_change == "CTO" and step >= 5:
        prefix = "Please note that the overall task objective has changed! "
        collaboration_user_prompt["mobile"] = prefix + collaboration_user_prompt["mobile"]

    if is_change == "ANC" and step >= 5:
        prefix = "Please note that new collaborators have joined the team! "
        collaboration_user_prompt["mobile"] = prefix + collaboration_user_prompt["mobile"]
    
    response = prompt_once_wo_func(collaboration_system_prompt["mobile"], collaboration_user_prompt["mobile"], LLM_type)
    
    tosave = {
        "mo_system_prompt": collaboration_system_prompt["mobile"],
        "mo_user_prompt": collaboration_user_prompt["mobile"],
        "mo_response": response.strip()
    }

    response_filepath = save_filepath + '/heterogeneous_collaboration/mobile'
    os.makedirs(response_filepath, exist_ok=True)
    response_filename = os.path.join(response_filepath, f'log_step_{step}.json')
    with open(response_filename, 'w', encoding='utf-8') as json_file:
        json.dump(tosave, json_file, ensure_ascii=False, indent=4)

    instruct = extract_instruct(response)
    action, object_name, stand_pose, place_where, delta_x, delta_y, content, role = instruct[0], instruct[1], instruct[2], instruct[3], instruct[4], instruct[5], instruct[6], instruct[7]
    
    if action != "wait":
        collaboration_action_steps["mobile"][0] += 1

    if action == "communicate":
        collaboration_communication_steps["mobile"][0] += 1

    if action == "navigate":
        irz_flag = False
        if is_change == "IRZ" and step >= 5:
            if "mo" in restricted_areas.keys():
                if object_name in layout[restricted_areas["mo"]]:
                    is_success = False
                    irz_flag = True
                    action_feedback = f"The object {object_name} is in the restricted area {restricted_areas['mo']}, which is not accessible for you. Please entrust other collaborators to conduct exploration."
        if not irz_flag:
            is_success, path_len, action_feedback = mo_agent.navigate(object_name, stand_pose, scene_graph, object_graph)
        robot_info = mo_agent.get_robot_info()
        collaboration_user_prompt["mobile"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # mobile history feedback
        collaboration_history_feedback["mobile"].append(action_feedback)
        collaboration_user_prompt["mobile"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile"]))
        # mobile history action
        collaboration_history_action["mobile"].append(action + "(" + object_name + " , " + stand_pose + ")")
        collaboration_user_prompt["mobile"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile"]))
    elif action == "communicate":
        if name_mapping["mobile"] in role:
            role.remove(name_mapping["mobile"])
        robot_info = mo_agent.get_robot_info()
        collaboration_user_prompt["mobile"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        action_feedback = f"The message has been successfully sent to {role}"
        # mobile history feedback
        collaboration_history_feedback["mobile"].append(action_feedback)
        collaboration_user_prompt["mobile"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile"]))
        # mobile history action
        collaboration_history_action["mobile"].append(action + "(" + content + " , " + str(role) + ")")
        collaboration_user_prompt["mobile"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile"]))
        # mobile receive message
        collaboration_history_receive_message["mobile"].append(f"[I said to {role}: {content}]")
        for robo_role in role:
            role_key = next((key for key, val in name_mapping.items() if val == robo_role), None)
            if role_key is not None:
                collaboration_history_receive_message[role_key].append("[" + content + "]" + " from David. ")
            else:
                print("##### Invalid Role! #####")
    elif action == "wait":
        robot_info = mo_agent.get_robot_info()
        collaboration_user_prompt["mobile"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # action_feedback = ""
        # # mobile history feedback
        # collaboration_history_feedback["mobile"].append(action_feedback)
        # collaboration_user_prompt["mobile"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["mobile"]))
        # # mobile history action
        # collaboration_history_action["mobile"].append(action + "()")
        # collaboration_user_prompt["mobile"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["mobile"]))
    else:
        print(f"ToDo!!!-{action}")
    
    print("The mobile is waiting...")
    sync_point()

    collaboration_user_prompt["mobile"] += "The historical receive messages, from oldest to newest, are as follows: " + str(list(collaboration_history_receive_message["mobile"]))

    # RT_task_state = monitor_task_completion(objects_list, mo_agent, object_graph)
    collaboration_user_prompt["mobile"] += RT_task_state

    # to_save_task_state[0] = RT_task_state

    if stop_event.is_set():
        return
# Bob 1
def manipulation_action(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, ma_agent, ma_scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, scenario_type, is_change):
    print(f"##################manipulation_{step}##################")

    # add task objective change to the user prompt
    if is_change == "CTO" and step >= 5:
        prefix = "Please note that the overall task objective has changed! "
        collaboration_user_prompt["manipulation"] = prefix + collaboration_user_prompt["manipulation"]

    if is_change == "ANC" and step >= 5:
        prefix = "Please note that new collaborators have joined the team! "
        collaboration_user_prompt["manipulation"] = prefix + collaboration_user_prompt["manipulation"]

    response = prompt_once_wo_func(collaboration_system_prompt["manipulation"], collaboration_user_prompt["manipulation"], LLM_type)

    tosave = {
        "ma_system_prompt": collaboration_system_prompt["manipulation"],
        "ma_user_prompt": collaboration_user_prompt["manipulation"],
        "ma_response": response.strip()
    }
    
    # update pose info about objects on table_0 to simulate perception
    ma_scene_graph = {}
    if scenario_type == "scenario_1":
        arm_reachable_range_x = (0, 1.3)
        arm_reachable_range_y = (0.6, 1.6)
    elif scenario_type == "scenario_2":
        arm_reachable_range_x = (2.0, 3.3)
        arm_reachable_range_y = (0.6, 1.6)
    elif scenario_type == "scenario_3":
        arm_reachable_range_x = (4.0, 5.3)
        arm_reachable_range_y = (-3.4, -2.4)
    
    for name in object_graph.keys():
        if "table_0" in name:
            if not (name == "tray" or name == "cutting_board" or "panel" in name):
                ma_scene_graph[name.split('-', 1)[1]] = object_graph[name]
                x = object_graph[name]["position"][0]
                y = object_graph[name]["position"][1]
                if (arm_reachable_range_x[0] <= x <= arm_reachable_range_x[1] and
                    arm_reachable_range_y[0] <= y <= arm_reachable_range_y[1]):
                    ma_scene_graph[name.split('-', 1)[1]]["whether_to_pick"] = "Yes"
                else:
                    ma_scene_graph[name.split('-', 1)[1]]["whether_to_pick"] = "No"
            else:
                ma_scene_graph[name.split('-', 1)[1]] = object_graph[name]
                # ma_scene_graph[name.split('-', 1)[1]]["whether_to_pick"] = "No"

    if ma_agent.task_type == "task_1" or ma_agent.task_type == "task_2": # pack objects and make sandwich
        if scenario_type == "scenario_1":
            Area = {}
            Area["place_where"] = [[0.8, 0.8, 1.15],[0.0, 1.7507963, 0.0]]
            ma_scene_graph["exchange_area"] = Area
        elif scenario_type == "scenario_2":
            Area = {}
            Area["place_where"] = [[2.8, 0.8, 1.15],[0.0, 1.7507963, 0.0]]
            ma_scene_graph["exchange_area"] = Area
        elif scenario_type == "scenario_3":
            Area = {}
            Area["place_where"] = [[4.8, -3.2, 1.15],[0.0, 1.7507963, 0.0]]
            ma_scene_graph["exchange_area"] = Area

    response_filepath = save_filepath + '/heterogeneous_collaboration/manipulation'
    os.makedirs(response_filepath, exist_ok=True)
    response_filename = os.path.join(response_filepath, f'log_step_{step}.json')
    with open(response_filename, 'w', encoding='utf-8') as json_file:
        json.dump(tosave, json_file, ensure_ascii=False, indent=4)

    print("1----------------------------------------------------------")
    instruct = extract_instruct(response)
    print("2----------------------------------------------------------")
    action, object_name, stand_pose, place_where, delta_x, delta_y, content, role = instruct[0], instruct[1], instruct[2], instruct[3], instruct[4], instruct[5], instruct[6], instruct[7]
    
    if action != "wait":
        collaboration_action_steps["manipulation"][0] += 1

    if action == "communicate":
        collaboration_communication_steps["manipulation"][0] += 1

    if action == "pick":
        is_success, action_feedback = ma_agent.pick(object_name, ma_scene_graph, object_graph)
        robot_info = ma_agent.get_robot_info()
        collaboration_user_prompt["manipulation"] = "scene_graph (It involves all objects located on table_0): " + json.dumps(ma_scene_graph) + robot_info
        # manipulation history feedback
        collaboration_history_feedback["manipulation"].append(action_feedback)
        collaboration_user_prompt["manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["manipulation"]))
        # manipulation history action
        collaboration_history_action["manipulation"].append(action + "(" + object_name + ")")
        collaboration_user_prompt["manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["manipulation"]))
    elif action == "place":
        is_success, action_feedback = ma_agent.place(object_name, place_where, ma_scene_graph, object_graph)
        robot_info = ma_agent.get_robot_info()
        collaboration_user_prompt["manipulation"] = "scene_graph (It involves all objects located on table_0): " + json.dumps(ma_scene_graph) + robot_info
        # manipulation history feedback
        collaboration_history_feedback["manipulation"].append(action_feedback)
        collaboration_user_prompt["manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["manipulation"]))
        # manipulation history action
        collaboration_history_action["manipulation"].append(action + "(" + object_name + " , " + place_where + ")")
        collaboration_user_prompt["manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["manipulation"]))
    elif action == "communicate":
        print(f"#### The message is: {content} and The role is: {role}. ####")
        if name_mapping["manipulation"] in role:
            role.remove(name_mapping["manipulation"])
        robot_info = ma_agent.get_robot_info()
        collaboration_user_prompt["manipulation"] = "scene_graph (It involves all objects located on table_0): " + json.dumps(ma_scene_graph) + robot_info
        action_feedback = f"The message has been successfully sent to {role}"
        # manipulation history feedback
        collaboration_history_feedback["manipulation"].append(action_feedback)
        collaboration_user_prompt["manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["manipulation"]))
        # manipulation history action
        collaboration_history_action["manipulation"].append(action + "(" + content + " , " + str(role) + ")")
        collaboration_user_prompt["manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["manipulation"]))
        # manipulation receive message
        collaboration_history_receive_message["manipulation"].append(f"[I said to {role}: {content}]")
        for robo_role in role:
            print(f"###robo_role:{robo_role}###")
            role_key = next((key for key, val in name_mapping.items() if val == robo_role), None)
            print(f"###role_key:{role_key}###")
            if role_key is not None:
                collaboration_history_receive_message[role_key].append("[" + content + "]" + " from Bob. ")
    elif action == "wait":
        robot_info = ma_agent.get_robot_info()
        collaboration_user_prompt["manipulation"] = "scene_graph (It involves all objects located on table_0): " + json.dumps(ma_scene_graph) + robot_info
        # action_feedback = ""
        # # manipulation history feedback
        # collaboration_history_feedback["manipulation"].append(action_feedback)
        # collaboration_user_prompt["manipulation"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["manipulation"]))
        # # manipulation history action
        # collaboration_history_action["manipulation"].append(action + "()")
        # collaboration_user_prompt["manipulation"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["manipulation"]))
    else:
        print(f"ToDo!!!-{action}")
    
    RT_task_state = monitor_task_completion(objects_list, ma_agent, object_graph)

    print("The manipulation is waiting...")
    sync_point()
    
    collaboration_user_prompt["manipulation"] += "The historical receive messages, from oldest to newest, are as follows: " + str(list(collaboration_history_receive_message["manipulation"]))

    collaboration_user_prompt["manipulation"] += RT_task_state

    to_save_task_state[0] = RT_task_state

    if stop_event.is_set():
        return

def drone_action(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, drone_agent, scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas):
    print(f"##################drone_{step}##################")

    # add task objective change to the user prompt
    if is_change == "CTO" and step >= 5:
        prefix = "Please note that the overall task objective has changed! "
        collaboration_user_prompt["drone"] = prefix + collaboration_user_prompt["drone"]

    if is_change == "ANC" and step >= 5:
        prefix = "Please note that new collaborators have joined the team! "
        collaboration_user_prompt["drone"] = prefix + collaboration_user_prompt["drone"]

    response = prompt_once_wo_func(collaboration_system_prompt["drone"], collaboration_user_prompt["drone"], LLM_type)
    
    tosave = {
        "drone_system_prompt": collaboration_system_prompt["drone"],
        "drone_user_prompt": collaboration_user_prompt["drone"],
        "drone_response": response.strip()
    }

    response_filepath = save_filepath + '/heterogeneous_collaboration/drone'
    os.makedirs(response_filepath, exist_ok=True)
    response_filename = os.path.join(response_filepath, f'log_step_{step}.json')
    with open(response_filename, 'w', encoding='utf-8') as json_file:
        json.dump(tosave, json_file, ensure_ascii=False, indent=4)
    print("1----------------------------------------------------------")
    instruct = extract_instruct(response)
    action, object_name, stand_pose, place_where, delta_x, delta_y, content, role = instruct[0], instruct[1], instruct[2], instruct[3], instruct[4], instruct[5], instruct[6], instruct[7]
    print("2----------------------------------------------------------")
    if action != "wait":
        collaboration_action_steps["drone"][0] += 1

    if action == "communicate":
        collaboration_communication_steps["drone"][0] += 1

    if action == "navigate":
        irz_flag = False
        if is_change == "IRZ" and step >= 5:
            if "drone" in restricted_areas.keys():
                if object_name in layout[restricted_areas["drone"]]:
                    is_success = False
                    irz_flag = True
                    action_feedback = f"The object {object_name} is in the restricted area {restricted_areas['drone']}, which is not accessible for you. Please entrust other collaborators to conduct exploration."
        if not irz_flag:
            is_success, path_len, action_feedback = drone_agent.fly(object_name, stand_pose, scene_graph, object_graph)
        robot_info = drone_agent.get_robot_info()
        collaboration_user_prompt["drone"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # drone history feedback
        collaboration_history_feedback["drone"].append(action_feedback)
        collaboration_user_prompt["drone"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["drone"]))
        # drone history action
        collaboration_history_action["drone"].append(action + "(" + object_name + " , " + stand_pose + ")")
        collaboration_user_prompt["drone"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["drone"]))
    elif action == "pick":
        is_success, action_feedback = drone_agent.fly_pick(object_name, scene_graph, object_graph)
        robot_info = drone_agent.get_robot_info()
        collaboration_user_prompt["drone"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # drone history feedback
        collaboration_history_feedback["drone"].append(action_feedback)
        collaboration_user_prompt["drone"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["drone"]))
        # drone history action
        collaboration_history_action["drone"].append(action + "(" + object_name + ")")
        collaboration_user_prompt["drone"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["drone"]))
    elif action == "place":
        is_success, action_feedback = drone_agent.fly_place(object_name, place_where, scene_graph, object_graph)
        robot_info = drone_agent.get_robot_info()
        collaboration_user_prompt["drone"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # drone history feedback
        collaboration_history_feedback["drone"].append(action_feedback)
        collaboration_user_prompt["drone"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["drone"]))
        # drone history action
        collaboration_history_action["drone"].append(action + "(" + object_name + " , " + place_where + ")")
        collaboration_user_prompt["drone"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["drone"]))
    elif action == "communicate":
        if name_mapping["drone"] in role:
            role.remove(name_mapping["drone"])
        robot_info = drone_agent.get_robot_info()
        collaboration_user_prompt["drone"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        action_feedback = f"The message has been successfully sent to {role}"
        # drone history feedback
        collaboration_history_feedback["drone"].append(action_feedback)
        collaboration_user_prompt["drone"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["drone"]))
        # drone history action
        collaboration_history_action["drone"].append(action + "(" + content + " , " + str(role) + ")")
        collaboration_user_prompt["drone"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["drone"]))
        # drone receive message
        collaboration_history_receive_message["drone"].append(f"[I said to {role}: {content}]")
        for robo_role in role:
            role_key = next((key for key, val in name_mapping.items() if val == robo_role), None)
            if role_key is not None:
                collaboration_history_receive_message[role_key].append("[" + content + "]" + " from Lucy. ")
    elif action == "wait":
        robot_info = drone_agent.get_robot_info()
        collaboration_user_prompt["drone"] = "scene_graph: " + json.dumps(scene_graph) + robot_info
        # action_feedback = ""
        # # drone history feedback
        # collaboration_history_feedback["drone"].append(action_feedback)
        # collaboration_user_prompt["drone"] += "The historical feedbacks, from oldest to newest, are as follows: " + str(list(collaboration_history_feedback["drone"]))
        # # drone history action
        # collaboration_history_action["drone"].append(action + "()")
        # collaboration_user_prompt["drone"] += "The historical actions, from oldest to newest, are as follows: " + str(list(collaboration_history_action["drone"]))
    else:
        print(f"ToDo!!!-{action}")

    print("The drone is waiting...")
    sync_point()

    collaboration_user_prompt["drone"] += "The historical receive messages, from oldest to newest, are as follows: " + str(list(collaboration_history_receive_message["drone"]))

    # RT_task_state = monitor_task_completion(objects_list, drone_agent, object_graph)
    collaboration_user_prompt["drone"] += RT_task_state

    # to_save_task_state[0] = RT_task_state

    if stop_event.is_set():
        return

def monitor_task_completion(objects_list, agent, object_graph):
    table_scene_graph = {}
    for name in object_graph.keys():
        if "table_0" in name:
            table_scene_graph[name.split('-', 1)[1]] = object_graph[name]

    RT_task_state = "Latest Task Progress Status: "
    object_finish, object_order = [], {}
    if agent.task_type == "task_1":  # pack objects
        # 步骤 1: 找出所有在托盘中的物体
        [min_x, min_y, min_z, max_x, max_y, max_z] = agent.client.get_bounding_box("tray")
        
        objects_in_tray = []
        for name in table_scene_graph.keys():
            # 跳过托盘本身
            if name == "tray":
                continue
            
            # 判断物体是否在托盘的 x, y 范围内
            x = table_scene_graph[name]["position"][0]
            y = table_scene_graph[name]["position"][1]
            if min_x < x < max_x and min_y < y < max_y:
                objects_in_tray.append(name)

        # 步骤 2: 根据新逻辑计算成功率
        # 将列表转换为集合，方便进行交集和差集运算
        # task_objects_set: 任务要求放入托盘的物体集合
        task_objects_set = set(objects_list)
        # objects_in_tray_set: 当前实际在托盘中的物体集合
        objects_in_tray_set = set(objects_in_tray)

        # 任务目标总数
        num_task_objects = len(task_objects_set)

        # 如果任务目标数量为 0，为防止除零错误，将成功率设为 1 (或根据业务逻辑设为 0)
        if num_task_objects == 0:
            success = 1.0
        else:
            # 计算交集: 完成的目标数量 (在托盘中且是任务目标)
            correctly_placed_objects = task_objects_set.intersection(objects_in_tray_set)
            num_correctly_placed = len(correctly_placed_objects)

            # 计算差集: 非目标但在托盘中的数量 (在托盘中但不是任务目标)
            incorrectly_placed_objects = objects_in_tray_set.difference(task_objects_set)
            num_incorrectly_placed = len(incorrectly_placed_objects)

            # 根据新公式计算 success
            numerator = num_correctly_placed - num_incorrectly_placed
            success = max(0, numerator / num_task_objects)
            # 稳健起见，确保 success 不会超过 1
            success = min(1.0, success)

        # 步骤 3: 更新状态和任务进度
        part_success[0] = success
        
        # 更新状态描述，现在使用 objects_in_tray 变量
        RT_task_state += f"{objects_in_tray} has been placed in the tray. "

        # 步骤 4: 判断任务是否最终完成
        # 任务完成的条件是：所有任务目标都在托盘里，且没有非任务目标在托盘里
        if task_objects_set == objects_in_tray_set:
            stop_event.set()
    elif agent.task_type == "task_2": # make sandwish
        [min_x, min_y, min_z, max_x, max_y, max_z] = agent.client.get_bounding_box("cutting_board")
        for name in table_scene_graph.keys():
            if name == "cutting_board" or name == "exchange_area":
                continue
            x = table_scene_graph[name]["position"][0]
            y = table_scene_graph[name]["position"][1]
            z = table_scene_graph[name]["position"][2]
            if x > min_x and x < max_x and y > min_y and y < max_y:
                # object_finish.append(name)
                object_order[name] = z
        object_order_bottom_up = sorted(object_order.items(), key=lambda x:x[1])
        object_order_final = dict(object_order_bottom_up)
        RT_task_state += f"{list(object_order_final.keys())} is placed on the cutting board one by one in stacking order. "
        part_success[0] = next((i for i, (a, b) in enumerate(zip(objects_list, list(object_order_final.keys()))) if a != b), len(list(object_order_final.keys()))) / len(objects_list)
        if objects_list == list(object_order_final.keys()):
            stop_event.set()
    elif agent.task_type == "task_3": # sort solid
        for name in table_scene_graph.keys():
            if "solid" in name:
                for color in ["yellow", "pink", "blue", "red", "green", "purple"]:
                    if color in name:
                        [min_x, min_y, min_z, max_x, max_y, max_z] = agent.client.get_bounding_box(color + "_panel")
                        x = table_scene_graph[name]["position"][0]
                        y = table_scene_graph[name]["position"][1]
                        if x > min_x and x < max_x and y > min_y and y < max_y:
                            object_finish.append(name)
        RT_task_state += f"The {object_finish} are correctly placed on the panels of the corresponding colors. "
        # part_success[0] = len(object_finish) / len(objects_list)
        object_finish_set = set(object_finish)
        objects_list_set = set(objects_list)
        completed_objects_in_list = object_finish_set.intersection(objects_list_set)
        part_success[0] = len(completed_objects_in_list) / len(objects_list_set)
        if set(objects_list).issubset(set(object_finish)):
            stop_event.set()

    return RT_task_state


def load_lists_from_json(filename): 
    with open(filename, 'r') as file:  
        return json.load(file)  

def sync_point(timeout=10):
    global threads_in_step, threads_arrived, barrier_broken
    with condition:
        if barrier_broken:
            return False
        threads_arrived += 1
        print(f"###threads_arrived: {threads_arrived}###")
        print(f"###threads_in_step: {threads_in_step}###")
        if threads_arrived == threads_in_step:
            threads_arrived = 0
            condition.notify_all()  # 唤醒所有线程继续
            print("###All threads have arrived###")
        else:
            success = condition.wait(timeout = timeout)
            if not success:
                barrier_broken = True
                condition.notify_all()  # 唤醒其他线程，不要卡死
                print("###Barrier broken due to timeout###")
                return False
        return True

def main(filename, agent_config):   
    # Load agent config
    scenario_type = agent_config['scenario_type']
    task_type = agent_config['task_type']
    objects_list = agent_config['objects_list']
    if agent_config['is_change'] == "CTO":
        change_objects_list = agent_config['change_objects_list']
    else:
        change_objects_list = []
    robot_type = agent_config['robot_type']
    LLM_type = agent_config['LLM_type']
    max_selection_num = agent_config['max_selection_num']
    max_history_length = agent_config['max_history_length']
    is_change = agent_config['is_change']
    layout = agent_config['layout']
    if agent_config['is_change'] == "IRZ":
        restricted_areas = agent_config['restricted_areas']
    else:
        restricted_areas = {}
    if agent_config['is_change'] == "ANC" or agent_config['is_change'] == "REC":
        extro_robot = agent_config['extro_robot']
    else:
        extro_robot = ["NO_EXTRO"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_abbr = "".join([name_mapping[t][0] for t in robot_type])
    if agent_config['is_change'] == "REC":
        save_filepath = f'./results/paper/{scenario_type}_{task_type}_{robot_abbr}_{agent_config["is_change"]}_{agent_config["extro_robot"]}_{agent_config["task_diff"]}_{timestamp}_{agent_config["exp_index"]}'
    else: 
        save_filepath = f'./results/paper/{scenario_type}_{task_type}_{robot_abbr}_{agent_config["is_change"]}_{agent_config["task_diff"]}_{timestamp}_{agent_config["exp_index"]}'
    os.makedirs(save_filepath, exist_ok=True)

    with open(f'./prompts/{task_type}/overall_task/overall_task.json', 'r') as file:
        overall_task = json.load(file)
    ot_info = {
        'overall_task': objects_list
    }
    overall_task = Template(overall_task).safe_substitute(ot_info)

    # overall task after change
    if is_change == "CTO":
        with open(f'./prompts/{task_type}/overall_task/change_overall_task.json', 'r') as file:
            change_overall_task = json.load(file)
        change_ot_info = {
            'overall_task': change_objects_list
        }
        change_overall_task = Template(change_overall_task).safe_substitute(change_ot_info)

    # ==========================Heterogeneous Collaboration ==========================
    print("###Heterogeneous Collaboration###")
    
    # generate collaboration system prompt
    collaboration_system_prompt = {}
    for rt in robot_type:
        with open(f'./prompts/{task_type}/collaboration/{rt}_system_prompt_no_discuss.json', 'r') as file:
            rt_system_prompt = json.load(file)
        with open(f'./prompts/{task_type}/collaboration/output_format.json', 'r') as file:
            output_format = json.load(file)
        rt_sys_info = {
            'name': name_mapping[rt],
            'target_task': overall_task,
            'teammates': json.dumps([name_mapping[item] for item in robot_type if item != rt]),
        }
        rt_system_prompt = Template(rt_system_prompt).substitute(rt_sys_info)
        with open(f'./prompts/{task_type}/collaboration/{rt}_example.json', 'r') as file:
            rt_examle = json.load(file)
        collaboration_system_prompt[rt] = rt_system_prompt + output_format + rt_examle
    
    if is_change == "CTO":
        change_collaboration_system_prompt = {}
        for rt in robot_type:
            with open(f'./prompts/{task_type}/collaboration/{rt}_system_prompt_no_discuss.json', 'r') as file:
                rt_system_prompt = json.load(file)
            with open(f'./prompts/{task_type}/collaboration/output_format.json', 'r') as file:
                output_format = json.load(file)
           
            rt_sys_info = {
                'name': name_mapping[rt],
                'target_task': change_overall_task,
                'teammates': json.dumps([name_mapping[item] for item in robot_type if item != rt]),

            }
            rt_system_prompt = Template(rt_system_prompt).substitute(rt_sys_info)
            with open(f'./prompts/{task_type}/collaboration/{rt}_example.json', 'r') as file:
                rt_examle = json.load(file)
            change_collaboration_system_prompt[rt] = rt_system_prompt + output_format + rt_examle

    if is_change == "ANC":
        extro_collaboration_system_prompt = {}
        for rt in robot_type + extro_robot:
            with open(f'./prompts/{task_type}/collaboration/{rt}_system_prompt_no_discuss.json', 'r') as file:
                rt_system_prompt = json.load(file)
            with open(f'./prompts/{task_type}/collaboration/output_format.json', 'r') as file:
                output_format = json.load(file)
          
            rt_sys_info = {
                'name': name_mapping[rt],
                'target_task': overall_task,
                'teammates': json.dumps([name_mapping[item] for item in (robot_type + extro_robot) if item != rt]),
            }
            rt_system_prompt = Template(rt_system_prompt).substitute(rt_sys_info)
            with open(f'./prompts/{task_type}/collaboration/{rt}_example.json', 'r') as file:
                rt_examle = json.load(file)
            extro_collaboration_system_prompt[rt] = rt_system_prompt + output_format + rt_examle
    # sellea_filepath = f'./results/test'
    # os.makedirs(sellea_filepath, exist_ok=True)
    # collsys_filename = os.path.join(sellea_filepath, f'collaboration_system_prompt.json')
    # with open(collsys_filename, 'w', encoding='utf-8') as json_file:
    #     json.dump(collaboration_system_prompt, json_file, ensure_ascii=False, indent=4)
    
    # with open(f'./results/test/collaboration_system_prompt.json', 'r') as file:
    #     collaboration_system_prompt = json.load(file)

    # Load environment config
    env_cfg = load_config(f'../Config/{scenario_type}_config.yaml')

    # Init client and visualizer
    client = Client(env_cfg.Client)
    visualizer = Visualizer(client, env_cfg.Visualizer)

    # Load scene
    scene_path = f'../Asset/Scene/Scene_for_exp/{scenario_type}_{task_type}_test.json'
    scene_graph, object_graph = client.create_scene(scene_path, visualizer, scenario_type)
    robot_scene_graph = {}
    for robot_name in ["mobile_manipulation", "manipulation", "mobile", "drone"]:
        robot_scene_graph[robot_name] = copy.deepcopy(scene_graph)
    

    # object graph on table_0 for manipulation
    # ma_scene_graph = {}
    # for name in object_graph.keys():
    #     if "table_0" in name:
    #         ma_scene_graph[name.split('-', 1)[1]] = object_graph[name]

    # if task_type == "task_2": # make sandwish
    #     Area = {}
    #     Area["position"] = [0.75, 1.0, 1.15]
    #     Area["orientation"] = [0.0, 1.7507963, 0.0]
    #     ma_scene_graph["exchange_area"] = Area

    # Start record
    # visualizer.start_record(f"{scenario_type}_{task_type}")

    # create agents and generate user prompt
    collaboration_user_prompt = {}
    collaboration_history_feedback = {}
    collaboration_history_receive_message = {}
    collaboration_history_action = {}
    collaboration_action_steps = {}
    collaboration_communication_steps = {}
    collaboration_func_call_format = {}
    for rt in robot_type + (extro_robot if is_change == "ANC" else []):
        if rt == "mobile_manipulation":
            moma_agent = LLM_agent(client, visualizer, env_cfg, rt, task_type)
            moma_robot_info = moma_agent.get_robot_info()
            collaboration_user_prompt[rt] = "scene_graph: " + json.dumps(scene_graph) + moma_robot_info
            collaboration_history_feedback[rt] = deque(maxlen=max_history_length)
            collaboration_history_receive_message[rt] = deque(maxlen=max_history_length)
            collaboration_history_action[rt] = deque(maxlen=max_history_length)
            collaboration_action_steps[rt] = [0]
            collaboration_communication_steps[rt] = [0]
            with open(f'./prompts/{task_type}/collaboration/{rt}_func_call_format.json', 'r') as file:
                collaboration_func_call_format[rt] = [json.load(file)]
        elif rt == "manipulation":
            ma_agent = LLM_agent(client, visualizer, env_cfg, rt, task_type)
            ma_robot_info = ma_agent.get_robot_info()
            ma_scene_graph = {}
            for name in object_graph.keys():
                if "table_0" in name:
                    ma_scene_graph[name.split('-', 1)[1]] = object_graph[name]
            collaboration_user_prompt[rt] = "scene_graph: " + json.dumps(ma_scene_graph) + ma_robot_info
            collaboration_history_feedback[rt] = deque(maxlen=max_history_length)
            collaboration_history_receive_message[rt] = deque(maxlen=max_history_length)
            collaboration_history_action[rt] = deque(maxlen=max_history_length)
            collaboration_action_steps[rt] = [0]
            collaboration_communication_steps[rt] = [0]
            with open(f'./prompts/{task_type}/collaboration/{rt}_func_call_format.json', 'r') as file:
                collaboration_func_call_format[rt] = [json.load(file)]
        elif rt == "mobile":
            mo_agent = LLM_agent(client, visualizer, env_cfg, rt, task_type)
            mo_robot_info = mo_agent.get_robot_info()
            collaboration_user_prompt[rt] = "scene_graph: " + json.dumps(scene_graph) + mo_robot_info
            collaboration_history_feedback[rt] = deque(maxlen=max_history_length)
            collaboration_history_receive_message[rt] = deque(maxlen=max_history_length)
            collaboration_history_action[rt] = deque(maxlen=max_history_length)
            collaboration_action_steps[rt] = [0]
            collaboration_communication_steps[rt] = [0]
            with open(f'./prompts/{task_type}/collaboration/{rt}_func_call_format.json', 'r') as file:
                collaboration_func_call_format[rt] = [json.load(file)]
        elif rt == "drone":
            drone_agent = LLM_agent(client, visualizer, env_cfg, rt, task_type)
            drone_robot_info = drone_agent.get_robot_info()
            collaboration_user_prompt[rt] = "scene_graph: " + json.dumps(scene_graph) + drone_robot_info
            collaboration_history_feedback[rt] = deque(maxlen=max_history_length)
            collaboration_history_receive_message[rt] = deque(maxlen=max_history_length)
            collaboration_history_action[rt] = deque(maxlen=max_history_length)
            collaboration_action_steps[rt] = [0]
            collaboration_communication_steps[rt] = [0]
            with open(f'./prompts/{task_type}/collaboration/{rt}_func_call_format.json', 'r') as file:
                collaboration_func_call_format[rt] = [json.load(file)]
        else:
            print(f"Error: Invalid robot type: {rt}")
            exit()

    finished_step = agent_config["max_run_steps"]
    max_run_steps = agent_config["max_run_steps"]
    extro_proplan_response = ""
    for step in range(max_run_steps):
        global threads_in_step, threads_arrived, barrier_broken
        threads_arrived = 0
        # change overall task goal when T = 5
        if is_change == "CTO" and step > 4:
            collaboration_system_prompt = change_collaboration_system_prompt
            objects_list = change_objects_list
        
        if is_change == "ANC" and step > 4:
            collaboration_system_prompt = extro_collaboration_system_prompt
            threads_in_step = len(robot_type + extro_robot)
        else:
            threads_in_step = len(robot_type)

        if is_change == "REC" and step > 4:
            threads_in_step = len(robot_type) - 1
        else:
            threads_in_step = len(robot_type)
        

        for rt in robot_type + extro_robot:
            if rt == "mobile_manipulation":
                if rt != extro_robot[0]: # The robot that do not belong to addition and deletion
                    moma_thread = threading.Thread(target=mobile_manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, moma_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                elif is_change == "ANC":
                    if step > 4:
                        moma_thread = threading.Thread(target=mobile_manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, moma_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                    else:
                        print(f"It is not yet the time for the {rt} robot to join the collaborative group.")
                elif is_change == "REC":
                    if step > 4:
                        print(f"{rt} robot has successfully disengaged from the cooperative group and will no longer execute actions.")
                    else:
                        moma_thread = threading.Thread(target=mobile_manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, moma_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
            elif rt == "manipulation":
                if rt != extro_robot[0]:
                    ma_thread = threading.Thread(target=manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, ma_agent, ma_scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, scenario_type, is_change))
                elif is_change == "ANC":
                    if step > 4:
                        ma_thread = threading.Thread(target=manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, ma_agent, ma_scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, scenario_type, is_change))
                    else:
                        print(f"It is not yet the time for the {rt} robot to join the collaborative group.")
                elif is_change == "REC":
                    if step > 4:
                        print(f"{rt} robot has successfully disengaged from the cooperative group and will no longer execute actions.")
                    else:
                        ma_thread = threading.Thread(target=manipulation_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, ma_agent, ma_scene_graph, object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, scenario_type, is_change))
            elif rt == "mobile":
                if rt != extro_robot[0]:
                    mo_thread = threading.Thread(target=mobile_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, mo_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                elif is_change == "ANC":
                    if step > 4:
                        mo_thread = threading.Thread(target=mobile_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, mo_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                    else:
                        print(f"It is not yet the time for the {rt} robot to join the collaborative group.")
                elif is_change == "REC":
                    if step > 4:
                        print(f"{rt} robot has successfully disengaged from the cooperative group and will no longer execute actions.")
                    else:
                        mo_thread = threading.Thread(target=mobile_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, mo_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
            elif rt == "drone":
                if rt != extro_robot[0]:
                    drone_thread = threading.Thread(target=drone_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, drone_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                elif is_change == "ANC":
                    if step > 4:
                        drone_thread = threading.Thread(target=drone_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, drone_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
                    else:
                        print(f"It is not yet the time for the {rt} robot to join the collaborative group.")
                elif is_change == "REC":
                    if step > 4:
                        print(f"{rt} robot has successfully disengaged from the cooperative group and will no longer execute actions.")
                    else:
                        drone_thread = threading.Thread(target=drone_action, args=(step, save_filepath, LLM_type, objects_list, collaboration_system_prompt, collaboration_user_prompt, collaboration_history_receive_message, collaboration_history_feedback, collaboration_history_action, drone_agent, robot_scene_graph[rt], object_graph, collaboration_action_steps, collaboration_communication_steps, collaboration_func_call_format, is_change, layout, restricted_areas))
        # print(f"extro_robot: {extro_robot}")
        for rt in robot_type + (extro_robot if is_change == "ANC" else []):
            if rt == "mobile_manipulation":
                if rt != extro_robot[0]:
                    moma_thread.start()
                elif is_change == "ANC":
                    if step > 4:
                        moma_thread.start()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        moma_thread.start()
            elif rt == "manipulation":
                if rt != extro_robot[0]:
                    ma_thread.start()
                elif is_change == "ANC":
                    if step > 4:
                        ma_thread.start()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        ma_thread.start()
            elif rt == "mobile":
                if rt != extro_robot[0]:
                    mo_thread.start()
                elif is_change == "ANC":
                    if step > 4:
                        mo_thread.start()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        mo_thread.start()
            elif rt == "drone":
                if rt != extro_robot[0]:
                    drone_thread.start()
                elif is_change == "ANC":
                    if step > 4:
                        drone_thread.start()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        drone_thread.start()

        for rt in robot_type + (extro_robot if is_change == "ANC" else []):
            if rt == "mobile_manipulation":
                if rt != extro_robot[0]:
                    moma_thread.join()
                elif is_change == "ANC":
                    if step > 4:
                        moma_thread.join()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        moma_thread.join()
            elif rt == "manipulation":
                if rt != extro_robot[0]:
                    ma_thread.join()
                elif is_change == "ANC":
                    if step > 4:
                        ma_thread.join()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        ma_thread.join()
            elif rt == "mobile":
                if rt != extro_robot[0]:
                    mo_thread.join()
                elif is_change == "ANC":
                    if step > 4:
                        mo_thread.join()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        mo_thread.join()
            elif rt == "drone":
                if rt != extro_robot[0]:
                    drone_thread.join()
                elif is_change == "ANC":
                    if step > 4:
                        drone_thread.join()
                    else:
                        continue
                elif is_change == "REC":
                    if step > 4:
                        continue
                    else:
                        drone_thread.join()

        if stop_event.is_set():
            finished_step = step
            break

    # End record
    # visualizer.end_record()

    result_tosave = {"finished_step": finished_step, "part_success": part_success, "task_diff": agent_config["task_diff"], "is_change":agent_config["is_change"], "agent_list":agent_config["robot_type"], "task_status": to_save_task_state, "task_objects": agent_config["objects_list"], "change_task_objects": change_objects_list, "select_leader_failed": select_leader_failed}
    for rt in robot_type:
        result_tosave[f"{rt}_action_steps"] = collaboration_action_steps[rt]
        result_tosave[f"{rt}_communication_steps"] = collaboration_communication_steps[rt]

    result_filepath = save_filepath + '/results'
    os.makedirs(result_filepath, exist_ok=True)
    result_filename = os.path.join(result_filepath, f'results.json')
    with open(result_filename, 'w', encoding='utf-8') as json_file:
        json.dump(result_tosave, json_file, ensure_ascii=False, indent=4)
    
    stop_event.clear()
    # disconnect
    client.wait(20)
    client.disconnect()

if __name__=='__main__':
    # set work dir to Examples
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # get current file name
    filename = os.path.splitext(os.path.basename(__file__))[0]

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, help="Path to config file")
    # args = parser.parse_args()

    # with open(args.config, 'r') as f:
    #     agent_config = json.load(f)

    # The room layout of the objects
    # Layout = {"scenario_1": {"task_1": {"kitchen": ["drawer", "cabinet_0", "cabinet_1", "dishwasher", "microwave", "fridge", "table_0", "table_1", "chair", "wall_top_0"], "living_room": ["sofa", "table_2", "table_3", "book_case"], "bathroom": ["tub", "washbasin", "faucet", "table_4"]}, "task_2": {}, "task_3": {}}, "scenario_2": {"task_1": {}, "task_2": {"kitchen": ["drawer", "cabinet_0", "cabinet_1", "dishwasher", "microwave", "fridge", "table_0", "table_1", "chair_1", "chair_2", "wall_top_0"], "living_room": ["book_case", "table_3", "chair_3", "chair_4"], "bathroom": ["washbasin", "faucet", "tub", "table_2"]}, "task_3": {}}, "scenario_3": {"task_1": {}, "task_2": {}, "task_3": {"kitchen":["fridge", "drawer", "cabinet_0", "cabinet_1", "table_0", "dishwasher", "microwave", "table_1"], "bedroom": ["wall_top_0", "sofa", "table_2", "bed", "tv"]}}}

    # test_difficulty_level = "hard"
    # test_task_type = "task_1"
    # difficulty_max_steps = {"easy": 20, "medium": 30, "hard": 50}
    # test_scenario_type = "scenario_1"
    # is_change = None # CTO, IRZ, ANC, REC
    # robot_type_mapping = {"ma+moma": ["mobile_manipulation", "manipulation"], "ma+drone": ["manipulation", "drone"], "ma+moma+mo": ["manipulation", "mobile_manipulation", "mobile"], "ma+mo+drone": ["manipulation", "mobile", "drone"], "ma+moma+drone": ["manipulation", "mobile_manipulation", "drone"], "ma+moma+mo+drone": ["manipulation", "mobile_manipulation", "mobile", "drone"]}
    # change_robo_comb = [("ma+moma", "ma+moma+drone"), ("ma+drone", "ma+moma+drone"), ("ma+moma+mo", "ma+moma+mo+drone"), ("ma+mo+drone", "ma+moma+mo+drone")]

    # exp_file = f"../Asset/Scene/Scene_for_exp/exp_list_{test_scenario_type}_{test_task_type}.json"
    # with open(exp_file, 'r') as f:
    #     exp_config = json.load(f)

    # robo_type = ["ma+mo+drone"] # "ma+moma", "ma+drone", "ma+moma+mo", "ma+moma+drone", "ma+mo+drone", "ma+moma+mo+drone" 
    # # if ANC select "ma+moma", "ma+drone", "ma+moma+mo", "ma+mo+drone";
    # # if REC select "ma+moma+drone", "ma+moma+mo+drone"
    # robo_rec = ["mobile_manipulation"] # mobile_manipulation, drone
    # sample_index = {"ma+moma": [0,1,2], "ma+drone": [0,1,2], "ma+moma+mo": [0,1,2], "ma+moma+drone": [2], "ma+mo+drone": [0,1,2], "ma+moma+mo+drone": [0,1,2]}

    # for rt in robo_type:
    #     filtered_data = [
    #     entry for entry in exp_config
    #     if entry.get("robot_category") == rt and entry.get("difficulty_level") == test_difficulty_level
    #     ]
        
    #     agent_config = {
    #     # "task_type" : "task_2",
    #     # "objects_list" : ["bread_slice_0", "cucumber", "ham", "bread_slice_1"],
    #     # "scenario_type" : "scenario_1",
    #     "LLM_type" : "gpt-4o",
    #     # "robot_type" : ["mobile_manipulation", "manipulation", "mobile"],
    #     "max_selection_num" : 3,
    #     "max_history_length" : 10,
    #     "max_run_steps": difficulty_max_steps[test_difficulty_level], # 20, 30, 50
    #     "is_change": is_change,
    #     "layout": Layout[test_scenario_type][test_task_type]
    #     }
    #     for index in sample_index[rt]:
    #         item = filtered_data[index]
    #         for num in range(3): # 每个任务跑3次
    #             agent_config["task_type"] = item["task"]
    #             agent_config["scenario_type"] = item["scenario"]
    #             agent_config["objects_list"] = item["objects"]
    #             if "change_objects" in item:
    #                 agent_config["change_objects_list"] = item["change_objects"]
    #             if "restricted_areas" in item:
    #                 agent_config["restricted_areas"] = item["restricted_areas"]
    #             agent_config["exp_index"] = num
    #             # if item["robot_category"] == "ma+moma":
    #             #     agent_config["robot_type"] = ["mobile_manipulation", "manipulation"]
    #             # elif item["robot_category"] == "ma+moma+mo":
    #             #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "mobile"]
    #             # elif item["robot_category"] == "ma+mo+drone":
    #             #     agent_config["robot_type"] = ["manipulation", "mobile", "drone"]
    #             # elif item["robot_category"] == "ma+drone":
    #             #     agent_config["robot_type"] = ["manipulation", "drone"]
    #             # elif item["robot_category"] == "ma+moma+drone":
    #             #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "drone"]
    #             # elif item["robot_category"] == "ma+moma+mo+drone":
    #             #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "mobile", "drone"]
    #             agent_config["robot_type"] = robot_type_mapping[item["robot_category"]]
    #             if is_change == "ANC":
    #                 for crc in change_robo_comb:
    #                     if item["robot_category"] == crc[0]:
    #                         agent_config["extro_robot"] = [x for x in robot_type_mapping[crc[1]] if x not in robot_type_mapping[item["robot_category"]]]
    #             elif is_change == "REC":
    #                 agent_config["extro_robot"] = robo_rec
    #             print("##############################################################")
    #             print(f"Running {agent_config.get('scenario_type', '')} {agent_config.get('task_type', '')} {agent_config.get('robot_type', '')} {item.get('difficulty_level', '')} {agent_config.get('objects_list', '')} {agent_config.get('change_objects_list', '')} {agent_config.get('restricted_areas', '')} {agent_config.get('extro_robot', '')}")
    #             print("##############################################################")

    #             main(filename, agent_config)

    Layout = {"scenario_1": {"task_1": {"kitchen": ["drawer", "cabinet_0", "cabinet_1", "dishwasher", "microwave", "fridge", "table_0", "table_1", "chair", "wall_top_0"], "living_room": ["sofa", "table_2", "table_3", "book_case"], "bathroom": ["tub", "washbasin", "faucet", "table_4"]}, "task_2": {}, "task_3": {}}, "scenario_2": {"task_1": {}, "task_2": {"kitchen": ["drawer", "cabinet_0", "cabinet_1", "dishwasher", "microwave", "fridge", "table_0", "table_1", "chair_1", "chair_2", "wall_top_0"], "living_room": ["book_case", "table_3", "chair_3", "chair_4"], "bathroom": ["washbasin", "faucet", "tub", "table_2"]}, "task_3": {}}, "scenario_3": {"task_1": {}, "task_2": {}, "task_3": {"kitchen":["fridge", "drawer", "cabinet_0", "cabinet_1", "table_0", "dishwasher", "microwave", "table_1"], "bedroom": ["wall_top_0", "sofa", "table_2", "bed", "tv"]}}}

    difficulty_max_steps = {"easy": 20, "medium": 30, "hard": 50}
    is_change = "CTO" # CTO, IRZ, ANC, REC
    change_robo_comb = [("ma+moma", "ma+moma+drone"), ("ma+drone", "ma+moma+drone"), ("ma+moma+mo", "ma+moma+mo+drone"), ("ma+mo+drone", "ma+moma+mo+drone")]

    
   

    test_difficulty_levels = ["medium"]  # 所有难度"easy", "medium", "hard"
    test_task_types = ["task_1", "task_2", "task_3"]  # 所有任务类型
    test_scenario_types = ["scenario_1"]  # 所有场景类型"scenario_1", "scenario_2", "scenario_3"
    
    difficulty_max_steps = {"easy": 20, "medium": 30, "hard": 50}
    # is_change = "ANC" # 可选: CTO, IRZ, ANC, REC
    robot_type_mapping = {"ma+moma": ["mobile_manipulation", "manipulation"], "ma+drone": ["manipulation", "drone"], "ma+moma+mo": ["manipulation", "mobile_manipulation", "mobile"], "ma+mo+drone": ["manipulation", "mobile", "drone"], "ma+moma+drone": ["manipulation", "mobile_manipulation", "drone"], "ma+moma+mo+drone": ["manipulation", "mobile_manipulation", "mobile", "drone"]}
    # change_robo_comb = [("ma+moma", "ma+moma+drone"), ("ma+drone", "ma+moma+drone"), ("ma+moma+mo", "ma+moma+mo+drone"), ("ma+mo+drone", "ma+moma+mo+drone")]

    robo_type = ["ma+moma", "ma+drone", "ma+moma+mo", "ma+moma+drone", "ma+mo+drone", "ma+moma+mo+drone"] # "ma+moma", "ma+drone", "ma+moma+mo", "ma+moma+drone", "ma+mo+drone", "ma+moma+mo+drone"
    # if ANC select "ma+moma", "ma+drone", "ma+moma+mo", "ma+mo+drone";
    # if REC select "ma+moma+drone", "ma+moma+mo+drone"
    robo_rec_list = [["drone"],["mobile_manipulation"]] # mobile_manipulation, drone
    robo_rec = ["mobile_manipulation"] # mobile_manipulation, drone
    sample_index = {"ma+moma": [0,1,2], "ma+drone": [0,1,2], "ma+moma+mo": [0,1,2], "ma+moma+drone": [0,1,2], "ma+mo+drone": [0,1,2], "ma+moma+mo+drone": [0,1,2]}

    # 定义场景和任务的对应关系
    scenario_task_mapping = {
        "scenario_1": ["task_1"],
        "scenario_2": ["task_2"], 
        "scenario_3": ["task_3"]
    }
    # for robo_rec in robo_rec_list:
    for test_scenario_type in test_scenario_types:
            for test_task_type in scenario_task_mapping[test_scenario_type]:
                for test_difficulty_level in test_difficulty_levels:

                    exp_file = f"../Asset/Scene/Scene_for_exp/exp_list_{test_scenario_type}_{test_task_type}.json"

                    with open(exp_file, 'r') as f:
                        exp_config = json.load(f)
                    for rt in robo_type:
                        filtered_data = [
                        entry for entry in exp_config
                        if entry.get("robot_category") == rt and entry.get("difficulty_level") == test_difficulty_level
                        ]
                        
                        agent_config = {
                        # "task_type" : "task_2",
                        # "objects_list" : ["bread_slice_0", "cucumber", "ham", "bread_slice_1"],
                        # "scenario_type" : "scenario_1",
                        "LLM_type" : "gpt-4o",
                        # "robot_type" : ["mobile_manipulation", "manipulation", "mobile"],
                        "max_selection_num" : 3,
                        "max_history_length" : 10,
                        "max_run_steps": difficulty_max_steps[test_difficulty_level], # 20, 30, 50
                        "is_change": is_change,
                        "layout": Layout[test_scenario_type][test_task_type],
                        "task_diff": test_difficulty_level
                        }
                        for index in sample_index[rt]:
                            item = filtered_data[index]
                            for num in range(3): # 每个任务跑3次
                                agent_config["task_type"] = item["task"]
                                agent_config["scenario_type"] = item["scenario"]
                                agent_config["objects_list"] = item["objects"]
                                if "change_objects" in item:
                                    agent_config["change_objects_list"] = item["change_objects"]
                                if "restricted_areas" in item:
                                    agent_config["restricted_areas"] = item["restricted_areas"]
                                agent_config["exp_index"] = num
                                # if item["robot_category"] == "ma+moma":
                                #     agent_config["robot_type"] = ["mobile_manipulation", "manipulation"]
                                # elif item["robot_category"] == "ma+moma+mo":
                                #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "mobile"]
                                # elif item["robot_category"] == "ma+mo+drone":
                                #     agent_config["robot_type"] = ["manipulation", "mobile", "drone"]
                                # elif item["robot_category"] == "ma+drone":
                                #     agent_config["robot_type"] = ["manipulation", "drone"]
                                # elif item["robot_category"] == "ma+moma+drone":
                                #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "drone"]
                                # elif item["robot_category"] == "ma+moma+mo+drone":
                                #     agent_config["robot_type"] = ["manipulation", "mobile_manipulation", "mobile", "drone"]
                                agent_config["robot_type"] = robot_type_mapping[item["robot_category"]]
                                if is_change == "ANC":
                                    for crc in change_robo_comb:
                                        if item["robot_category"] == crc[0]:
                                            agent_config["extro_robot"] = [x for x in robot_type_mapping[crc[1]] if x not in robot_type_mapping[item["robot_category"]]]
                                elif is_change == "REC":
                                    agent_config["extro_robot"] = robo_rec
                                print("##############################################################")
                                print(f"Running {agent_config.get('scenario_type', '')} {agent_config.get('task_type', '')} {agent_config.get('robot_type', '')} {item.get('difficulty_level', '')} {agent_config.get('objects_list', '')} {agent_config.get('change_objects_list', '')} {agent_config.get('restricted_areas', '')} {agent_config.get('extro_robot', '')}")
                                print("##############################################################")

                                main(filename, agent_config)