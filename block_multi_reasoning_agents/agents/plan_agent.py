import json
import random
from typing import Dict, List, Tuple
import streamlit as st

from openai import OpenAI

from config import settings


class PlanningAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL
        )
        self.indent_client = OpenAI(
            api_key=settings.INDENT_API_KEY, base_url=settings.INDENT_BASE_URL
        )

        # 预定义不同积木数量能拼成的正方形数量的分配方案
        # 键为(总积木数, 目标正方形数), 值为可能的分配方案列表
        # 每个分配方案中的字典表示每个正方形使用的积木数量和使用不同长度积木的数量
        # {使用积木数量: (使用长度为3的积木数量, 使用长度为2的积木数量, 使用长度为1的积木数量)}
        self.distribution_patterns = {
            (19, 3): [  # 19个积木拼3个正方形的可能分配
                [{5: (0, 4, 1)}, {6: (1, 1, 4)}, {8: (0, 1, 7)}],
                [{5: (1, 2, 2)}, {5: (0, 4, 1)}, {9: {0, 0, 9}}],
                [{5: (1, 2, 2)}, {7: (0, 2, 5)}, {7: (0, 2, 5)}],
                [{6: (1, 1, 4)}, {6: (0, 3, 3)}, {7: (0, 2, 5)}],
            ],
            (5, 1): [
                [{5: (1, 2, 2)}],
                [{5: (0, 4, 1)}],
            ],  # 5个积木拼1个正方形
            (6, 1): [
                [{6: (1, 1, 4)}],
                [{6: (0, 3, 3)}],
            ],  # 6个积木拼1个正方形
            (7, 1): [
                [{7: (0, 2, 5)}],
            ],  # 7个积木拼1个正方形
            (8, 1): [
                [{8: (0, 1, 7)}],
            ],  # 8个积木拼1个正方形
            (9, 1): [
                [{9: (0, 0, 9)}],
            ],  # 9个积木拼1个正方形
        }

        print("PuzzleSolver初始化完成")

    def analyze_user_intent(self, task_description: str) -> Dict:
        """分析用户意图"""
        prompt = f"""分析以下拼图任务描述，提取关键信息：
{task_description}

请提取以下信息并以JSON格式返回：
1. 需要拼成几个正方形(num_squares)
2. 是否需要列出所有可能的解(need_all_solutions)
3. 其他特殊要求(special_requirements)

示例输出：
{{
    "num_squares": 3,
    "need_all_solutions": false,
}}"""
        print(f"意图识别使用模型：{settings.INDENT_MODEL_NAME}")
        response = self.indent_client.chat.completions.create(
            model=settings.INDENT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )

        json_str = response.choices[0].message.content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]

        return json.loads(json_str)

    def plan(self, container, blocks_data: Dict, task_description: str) -> Dict:
        """
        基于大模型生成积木拼图方案
        Args:
            blocks_data: 积木和框架数据
            task_description: 任务描述
        Returns:
            解决方案或错误信息
        """
        try:
            # Step 1: 分析任务描述
            analysis = self.analyze_user_intent(task_description)
            if "num_squares" not in analysis:
                raise ValueError("未能从任务描述中提取有效信息")

            num_squares = analysis["num_squares"]
            total_blocks = blocks_data["total_blocks"]

            print(f"方案生成使用模型：{settings.LLM_MODEL_NAME}")
            # Step 2: 查找预定义分配方案
            distributions = self.distribution_patterns.get(
                (total_blocks, num_squares), []
            )
            if not distributions:
                raise ValueError(
                    f"没有找到{total_blocks}个积木拼{num_squares}个正方形的分配方案"
                )

            # 打印预定义分配方案
            print(f"预定义的分配方案: {distributions}")

            # Step 3: 随机选择一种方案
            distribution = random.choice(distributions)
            print(f"使用的分配方案: {distribution}")

            # Step 4: 初始化生成状态
            origin_blocks = blocks_data["blocks"]

            remaining_blocks = blocks_data["blocks"]
            remaining_frame = blocks_data["frame"]
            solutions = []

            for square_idx in range(1, num_squares + 1):
                success = False
                original_prompt = None  # 记录初始 prompt
                updated_prompt = None  # 记录更新后的 prompt
                while not success:
                    # Step 4: 生成单个正方形
                    current_distribution = distribution[square_idx - 1]
                    if not original_prompt:
                        # 如果没有初始 prompt，则创建一个
                        original_prompt = self.create_prompt(
                            remaining_blocks,
                            remaining_frame,
                            current_distribution,
                            square_idx,
                        )
                    # 使用更新后的 prompt（如果有）
                    active_prompt = updated_prompt or original_prompt

                    response = self.call_model(active_prompt)

                    solution_str = ""
                    if "is_expanded" not in st.session_state:
                        st.session_state["is_expanded"] = True
                    with container:
                        expander = st.expander(
                            "生成方案中...", expanded=st.session_state["is_expanded"]
                        )
                        with expander:
                            # 创建一个空的占位符
                            text_area = st.empty()

                            for resp in response:
                                if resp.choices[0].delta.content:
                                    solution_str += resp.choices[0].delta.content
                                    # 使用write更新整个内容
                                    text_area.markdown(solution_str)
                            st.session_state["is_expanded"] = False

                    # Step 5: 解析生成方案
                    solution = self.parse_solution(solution_str)

                    # Step 6: 验证方案
                    sucess, reason = self.verify_single_square(
                        solution,
                        current_distribution,
                        remaining_frame,
                        origin_blocks,
                    )
                    if sucess:
                        # 验证成功，更新剩余积木和框架
                        remaining_blocks = self.update_remaining_data(
                            solution, remaining_blocks
                        )
                        solutions.append(solution)
                        success = True
                    else:
                        # 更新 prompt，将错误原因和无效方案附加到提示中
                        updated_prompt = (
                            f"## 注意：上一个生成方案无效，原因如下：\n"
                            f"{reason}\n\n"
                            f"\n\n{original_prompt}"
                        )
                        print(f"错误反馈给模型：{reason}")

            # Step 7: 返回最终解决方案
            return {"status": "success", "solutions": solutions}
        except Exception as e:
            print(f"规划失败: {e}")
            return {"status": "error", "message": str(e)}

    def create_prompt(
        self, remaining_blocks, remaining_frame, current_distribution, square_idx
    ):
        """
        构建提示词以生成单个正方形
        Args:
            remaining_blocks: 当前剩余积木
            remaining_frame: 当前剩余框架
            current_distribution: 当前正方形所需积木数量
            square_idx: 当前正方形索引
        Returns:
            prompt: 完整提示词
        """
        blocks_info = self.format_blocks(remaining_blocks)
        frame_info = self.format_frame(remaining_frame)
        # 动态生成分配方案的描述
        distribution_description = self.format_distribution(current_distribution)

        prompt = f"""以下是当前积木和框架的描述：
## 积木信息：
{blocks_info}

## 目标图形信息：
{frame_info}

## 当前任务要求：
- **必须使用 {current_distribution} 个积木填满目标图形（正方形{square_idx}），确保无重叠。**
- **每个积木只能用一次**
- 不需要考虑朝向，只需要考虑位置和形状匹配，可以旋转

- 必须按照以下分配规则使用积木：
{distribution_description}

## **积木放置规则**：
1. 长度为3的积木所有可能的摆放位置如下：
- (circle, square, triangle) -> [(b, e, h)]
2. 长度为2的积木所有可能的摆放位置如下：
- (square, square) -> [(c, f), (e, f)]
- (square, triangle) -> [(e, h), (f, i)]
- (square, circle) -> [(b, c), (d, e), (b, e)]
- (triangle, triangle) -> [(g, h), (h ,i)]
- (triangle, circle) -> [(d, g)]
- (circle, circle) -> [(a, b), (a, d)]
3. 长度为1的积木所有可能的摆放位置如下：
- (square) -> [(c), (e), (f)]
- (triangle) -> [(g), (h), (i)]
- (circle) -> [(a), (b), (d)]
4. 只能从以上可能的摆放位置中选择，不能有其他选择

## **注意**： 
1. 必须填满整个目标区域，无重叠且无空隙
2. 必须按照要求的数量放置积木
3. 禁止重复使用积木
4. 禁止出现同一位置被多个积木占用
5. 积木摆放位置的形状数量必须相同
6. **积木需要连续放置，并且在同一行或者同一列**

## 示例输出格式：
```json
{{
    "status": "success",
    "blocks": [
        {{
            "block_id": 1,
            "positions": ["a", "b", "c"]
        }},
        {{
            "block_id": 2,
            "positions": ["d", "e", "f"]
        }},
        {{
            "block_id": 3,
            "positions": ["g", "h", "i"]
        }}
    ]
}}"""
        return prompt

    def verify_single_square(
        self,
        solution: Dict,
        current_distribution: int,
        remaining_frame: dict,
        origin_blocks: dict,
    ) -> Tuple[bool, str]:
        """
        直接使用代码逻辑验证单个正方形方案的有效性
        Args:
            solution: 当前生成的正方形方案
            current_distribution: 期望的积木数量
            remaining_frame: 当前框架信息
            origin_blocks: 原始积木信息
        Returns:
            (bool, str): 是否验证通过，失败原因（如果有）
        """
        try:
            # 提取方案中的积木数量
            used_blocks = solution.get("blocks", [])
            used_block_count = len(used_blocks)

            # **验证1：积木数量是否符合要求**
            if used_block_count != list(current_distribution.keys())[0]:
                print(f"错误: 方案使用了 {used_block_count} 个积木，但要求使用 {list(current_distribution.keys())[0]} 个。")

                return (
                    False,
                    f"错误: 方案使用了 {used_block_count} 个积木，但要求使用 {list(current_distribution.keys())[0]} 个。",
                )

            # **验证2：积木是否重复使用**
            used_block_ids = set()
            for block in used_blocks:
                block_id = block["block_id"]
                if block_id in used_block_ids:
                    print(f"错误: 积木 {block_id} 被重复使用。")
                    return False, f"错误: 积木 {block_id} 被重复使用。"
                used_block_ids.add(block_id)

            # **验证3：积木是否覆盖所有框架位置**
            used_positions = set()
            for block in used_blocks:
                for pos in block["positions"]:
                    if pos in used_positions:
                        print(f"错误: 位置 {pos} 被多个积木重复使用。")
                        return False, f"错误: 位置 {pos} 被多个积木重复使用。"
                    used_positions.add(pos)

            # **验证4：所有框架位置是否被完全覆盖**
            frame_positions = {shape["position"] for shape in remaining_frame["shapes"]}
            if used_positions != frame_positions:
                missing_positions = frame_positions - used_positions
                extra_positions = used_positions - frame_positions
                if missing_positions:
                    print(f"错误: 框架位置 {missing_positions} 没有被覆盖。")
                    return False, f"错误: 框架位置 {missing_positions} 没有被覆盖。"
                if extra_positions:
                    print(f"错误: 积木覆盖了不属于框架的额外位置 {extra_positions}。")
                    return (
                        False,
                        f"错误: 积木覆盖了不属于框架的额外位置 {extra_positions}。",
                    )

            # **验证5：积木是否符合摆放规则**
            valid_placements = {
                3: [("b", "e", "h")],
                2: [
                    ("c", "f"),
                    ("c", "f"),
                    ("e", "f"),
                    ("f", "e"),
                    ("e", "h"),
                    ("f", "i"),
                    ("b", "c"),
                    ("c", "b"),
                    ("d", "e"),
                    ("e", "d"),
                    ("b", "e"),
                    ("e", "b"),
                    ("g", "h"),
                    ("h", "g"),
                    ("h", "i"),
                    ("i", "h"),
                    ("d", "g"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "d"),
                    ("d", "a")
                ],
                1: [
                    ("c",),
                    ("e",),
                    ("f",),
                    ("g",),
                    ("h",),
                    ("i",),
                    ("a",),
                    ("b",),
                    ("d",),
                ],
            }

            for block in used_blocks:
                positions = tuple(block["positions"])
                block_length = len(positions)

                # 检查是否在有效摆放规则内
                if positions not in valid_placements.get(block_length, []):
                    print(f"错误：积木 {block['block_id']} 在位置 {positions} 处的摆放不符合规则。")
                    return (
                        False,
                        f"错误: 积木 {block['block_id']} 在位置 {positions} 处的摆放不符合规则。",
                    )

            # **验证6：积木形状与其位置上的框架形状数量是否匹配**
            # 1. 统计框架位置上的形状数量
            frame_shapes_map = {
                shape["position"]: shape["type"] for shape in remaining_frame["shapes"]
            }

            for block in used_blocks:
                block_id = block["block_id"] - 1
                block_shapes = origin_blocks[block_id]["shapes"]
                block_positions = block.get("positions", [])

                # 统计积木本身的形状数量
                block_shape_counts = {}
                for shape in block_shapes:
                    block_shape_counts[shape["type"]] = (
                        block_shape_counts.get(shape["type"], 0) + 1
                    )

                # 统计摆放位置上的框架形状数量
                frame_shape_counts = {}
                for pos in block_positions:
                    frame_shape = frame_shapes_map[pos]
                    frame_shape_counts[frame_shape] = (
                        frame_shape_counts.get(frame_shape, 0) + 1
                    )

                # 比较两者是否一致
                if block_shape_counts != frame_shape_counts:
                    print(f"错误：积木 {block_id} 的形状数量 {block_shape_counts} 与位置 {block_positions} 上的框架形状数量 {frame_shape_counts} 不匹配。")
                    return (
                        False,
                        f"错误: 积木 {block_id} 的形状数量 {block_shape_counts} 与位置 {block_positions} 上的框架形状数量 {frame_shape_counts} 不匹配。",
                    )

            return True, None  # 所有验证通过

        except Exception as e:
            return False, f"方案验证过程中出错: {e}"

    def update_remaining_data(self, solution: Dict, remaining_blocks: list):
        """更新剩余的积木数据"""
        used_blocks = {block["block_id"] for block in solution["blocks"]}
        remaining_blocks = [b for b in remaining_blocks if b["id"] not in used_blocks]

        return remaining_blocks

    def format_blocks(self, blocks):
        formatted = []
        for block in blocks:
            shape_details = ", ".join(
                f"<{shape['type']}({shape['orientation']})>"
                for shape in block["shapes"]
            )
            formatted.append(
                f"(ID {block['id']} | Layout: {block['layout']} | Length: {block['length']} | Shapes: {shape_details})"
            )
        return "\n".join(formatted)

    def format_frame(self, frame):
        formatted = []
        for shape in frame["shapes"]:
            formatted.append(
                f"Position: {shape['position']} | Shape: {shape['type']} | Orientation: {shape['orientation']}"
            )
        return "\n".join(formatted)

    def format_distribution(self, distribution):
        """格式化分配规则，生成提示描述文本

        Args:
            distribution (_type_): 当前正方形的积木分配方案
        Returns:
            str: 分配规则描述字符串
        """
        description = []
        for num_blocks, (len3, len2, len1) in distribution.items():
            description.append(
                f"- **正方形需要 {num_blocks} 个积木**：长度为3的积木 {len3} 个，长度为2的积木 {len2} 个，长度为1的积木 {len1} 个。"
            )
        return "\n".join(description)

    def call_model(self, prompt):
        """调用大模型生成解决方案"""
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            stream=True,
        )
        return response

    def parse_solution(self, response):
        """解析模型返回的最后一个 JSON 格式解决方案"""
        potential_json = []
        stack = []
        start = None

        # 遍历字符，找到完整的 JSON 块
        for i, char in enumerate(response):
            if char == "{":
                stack.append("{")
                if start is None:
                    start = i
            elif char == "}":
                stack.pop()
                if not stack and start is not None:
                    # 找到完整的 JSON
                    potential_json.append(response[start : i + 1])
                    start = None

        if not potential_json:
            raise ValueError("无法找到 JSON 格式数据")

        # 尝试解析最后一个 JSON 块
        last_json = potential_json[-1]
        try:
            return json.loads(last_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解码失败: {e}")
