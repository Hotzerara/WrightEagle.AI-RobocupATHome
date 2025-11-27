import json
import re
from typing import Dict, List, Any

class InstructionConverter:
    """
    将语音识别的文本转换为机器人指令的类
    """

    def __init__(self):
        # 定义动作关键词映射
        self.action_keywords = {
            'move': ['前进', '后退', '左转', '右转', '向上', '向下', '移动'],
            'stop': ['停止', '停'],
            'grab': ['抓取', '拿起', '抓'],
            'release': ['放下', '释放'],
            'dance': ['跳舞', '舞蹈'],
            'sing': ['唱歌', '唱'],
            'light': ['灯', '灯光', '照明']
        }

        # 方向关键词映射
        self.direction_mapping = {
            '前': 'forward',
            '后': 'backward',
            '左': 'left',
            '右': 'right',
            '上': 'up',
            '下': 'down'
        }

        # 数字提取正则表达式
        self.number_pattern = re.compile(r'(\d+)')

    def extract_numbers(self, text: str) -> List[int]:
        """
        从文本中提取数字
        """
        numbers = self.number_pattern.findall(text)
        return [int(num) for num in numbers]

    def detect_action(self, text: str) -> str:
        """
        检测文本中的动作类型
        """
        for action, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return action
        return 'unknown'

    def detect_direction(self, text: str) -> str:
        """
        检测文本中的方向
        """
        for chinese_dir, english_dir in self.direction_mapping.items():
            if chinese_dir in text:
                return english_dir
        return 'unknown'

    def convert_to_instruction(self, recognized_text: str) -> Dict[str, Any]:
        """
        将语音识别的文本转换为机器人指令

        Args:
            recognized_text: 语音识别出的文本

        Returns:
            包含指令信息的字典
        """
        # 清理文本，移除多余空格
        text = recognized_text.strip()

        # 检测动作类型
        action = self.detect_action(text)

        # 检测方向
        direction = self.detect_direction(text)

        # 提取数字(如距离、时间等)
        numbers = self.extract_numbers(text)

        # 构建指令
        instruction = {
            'original_text': text,
            'action': action,
            'direction': direction,
            'numbers': numbers,
            'parameters': {}
        }

        # 根据不同动作设置参数
        if action == 'move':
            instruction['command'] = 'MOVE'
            if direction != 'unknown':
                instruction['parameters']['direction'] = direction.upper()
            if numbers:
                instruction['parameters']['distance'] = numbers[0]  # 假设第一个数字是距离

        elif action == 'stop':
            instruction['command'] = 'STOP'

        elif action == 'grab':
            instruction['command'] = 'GRAB'
            if numbers:
                instruction['parameters']['object_id'] = numbers[0]

        elif action == 'release':
            instruction['command'] = 'RELEASE'

        elif action == 'dance':
            instruction['command'] = 'DANCE'

        elif action == 'sing':
            instruction['command'] = 'SING'

        elif action == 'light':
            instruction['command'] = 'LIGHT'
            if numbers:
                # 假设数字表示亮度等级(0-100)
                brightness = min(max(numbers[0], 0), 100)
                instruction['parameters']['brightness'] = brightness

        else:
            instruction['command'] = 'UNKNOWN'
            instruction['error'] = f'无法识别指令: {text}'

        return instruction

    def execute_instruction(self, instruction: Dict[str, Any]) -> str:
        """
        执行指令(模拟)

        Args:
            instruction: 指令字典

        Returns:
            执行结果描述
        """
        if instruction['command'] == 'UNKNOWN':
            return f"未识别的指令: {instruction['original_text']}"

        # 模拟执行指令
        result = f"执行指令: {instruction['command']}"
        if instruction['parameters']:
            result += f", 参数: {instruction['parameters']}"

        return result

def main():
    """
    主函数 - 示例用法
    """
    converter = InstructionConverter()

    # 示例语音识别文本
    examples = [
        "向前移动10厘米",
        "请抓取物体",
        "向左转",
        "打开灯光到50亮度",
        "停止所有动作",
        "跳一支舞",
        "唱一首歌"
    ]

    print("语音指令转换器示例:")
    print("=" * 40)

    for text in examples:
        print(f"\n语音识别结果: {text}")
        instruction = converter.convert_to_instruction(text)
        print(f"转换后的指令: {json.dumps(instruction, ensure_ascii=False, indent=2)}")
        result = converter.execute_instruction(instruction)
        print(f"执行结果: {result}")

if __name__ == "__main__":
    main()
