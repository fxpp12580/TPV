import json
from typing import Any, Dict, Optional, Sequence
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from pydantic import BaseModel, Field
import streamlit as st

from agents.plan_agent import PlanningAgent
from agents.vision_agent import VisionAgent
from core.display import display_state_details
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 初始化agents
vision_agent = VisionAgent()
planning_agent = PlanningAgent()


class AgentState(BaseModel):
    """系统状态定义"""

    block_img_path: str
    block_frame_img_path: str
    task_description: str
    messages: Sequence[BaseMessage] = Field(default_factory=list)
    blocks_data: Optional[Dict] = Field(default_factory=dict)
    planned_solution: Optional[Dict] = Field(default_factory=dict)
    state_containers: Optional[Dict] = None  # 添加状态容器字段
    final_solutions: Optional[str] = None  # 添加最终解决方案字段


def vision_node(state: AgentState) -> AgentState:
    """Vision Agent 节点处理函数"""
    try:
        vision_container = state.state_containers.get("vision")
        if vision_container:
            with vision_container:
                st.write("🔍 Performing Image Analysis... ")

                # state.blocks_data = {
                #     "total_blocks": 19,
                #     "blocks": [
                #         {
                #             "id": 1,
                #             "layout": "horizontal",
                #             "length": 3,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"},
                #                 {"type": "square", "position": 2, "orientation": "up"},
                #                 {
                #                     "type": "triangle",
                #                     "position": 3,
                #                     "orientation": "left",
                #                 },
                #             ],
                #         },
                #         {
                #             "id": 2,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"},
                #                 {"type": "square", "position": 2, "orientation": "up"},
                #             ],
                #         },
                #         {
                #             "id": 3,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"},
                #                 {
                #                     "type": "triangle",
                #                     "position": 2,
                #                     "orientation": "left",
                #                 },
                #             ],
                #         },
                #         {
                #             "id": 4,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"},
                #                 {"type": "circle", "position": 2, "orientation": "up"},
                #             ],
                #         },
                #         {
                #             "id": 5,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {
                #                     "type": "triangle",
                #                     "position": 1,
                #                     "orientation": "up",
                #                 },
                #                 {
                #                     "type": "triangle",
                #                     "position": 2,
                #                     "orientation": "up",
                #                 },
                #             ],
                #         },
                #         {
                #             "id": 6,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {
                #                     "type": "triangle",
                #                     "position": 1,
                #                     "orientation": "left",
                #                 },
                #                 {"type": "circle", "position": 2, "orientation": "up"},
                #             ],
                #         },
                #         {
                #             "id": 7,
                #             "layout": "horizontal",
                #             "length": 2,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"},
                #                 {"type": "circle", "position": 2, "orientation": "up"},
                #             ],
                #         },
                #         {
                #             "id": 8,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 9,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 10,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 11,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "square", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 12,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "triangle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 13,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "triangle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 14,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "triangle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 15,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "triangle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 16,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 17,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 18,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #         {
                #             "id": 19,
                #             "layout": "single",
                #             "length": 1,
                #             "shapes": [
                #                 {"type": "circle", "position": 1, "orientation": "up"}
                #             ],
                #         },
                #     ],
                #     "frame": {
                #         "total_shapes": 9,
                #         "shapes": [
                #             {"type": "circle", "position": "a", "orientation": "up"},
                #             {"type": "circle", "position": "b", "orientation": "up"},
                #             {"type": "square", "position": "c", "orientation": "up"},
                #             {"type": "circle", "position": "d", "orientation": "up"},
                #             {"type": "square", "position": "e", "orientation": "up"},
                #             {"type": "square", "position": "f", "orientation": "up"},
                #             {"type": "triangle", "position": "g", "orientation": "up"},
                #             {"type": "triangle", "position": "h", "orientation": "up"},
                #             {"type": "triangle", "position": "i", "orientation": "up"},
                #         ],
                #     },
                # }
                # 调用Vision Agent处理图片
                result = vision_agent.invoke(
                    {
                        "name": "vision_agent",
                        "args": {
                            "block_img_path": state.block_img_path,
                            "block_frame_img_path": state.block_frame_img_path,
                        },
                        "id": "vision_analysis",
                        "type": "tool_call",
                    }
                )
                data = json.loads(result.content)
                # 直接使用返回的字典结果
                if isinstance(data, dict) and data.get("status") == "success":
                    analysis = data.get("analysis", {})

                    # 获取积木和框架的分析结果
                    blocks_analysis = analysis.get("blocks", {})
                    frame_analysis = analysis.get("frame", {})

                    # 合并数据
                    if blocks_analysis.get("success") and frame_analysis.get("success"):
                        blocks_data = blocks_analysis.get("data", {})
                        frame_data = frame_analysis.get("data", {})

                        state.blocks_data = {
                            "total_blocks": blocks_data.get("total_blocks", 0),
                            "blocks": blocks_data.get("blocks", []),
                            "frame": frame_data
                        }

                        # 添加分析结果描述
                        state.messages = list(state.messages) + [
                            AIMessage(content=blocks_analysis.get("content", "")),
                            AIMessage(content=frame_analysis.get("content", "")),
                        ]
                    else:
                        error_msg = "图片分析部分失败"
                        if not blocks_analysis.get("success"):
                            error_msg += " - 积木分析失败"
                        if not frame_analysis.get("success"):
                            error_msg += " - 框架分析失败"
                        state.blocks_data = {"error": error_msg}
                        state.messages = list(state.messages) + [AIMessage(content=error_msg)]
                else:
                    error_msg = "视觉分析失败或返回格式不正确"
                    state.blocks_data = {"error": error_msg}
                    state.messages = list(state.messages) + [AIMessage(content=error_msg)]

                logger.info("======== Vision Agent result =========")
                logger.info(f"{state.blocks_data}")

                display_state_details(state.blocks_data, "blocks_data")

                return state

    except Exception as e:
        logger.error(f"Error in vision node: {str(e)}")
        error_msg = f"Vision Agent执行出错: {str(e)}"
        state.blocks_data = {"error": error_msg}
        state.messages = list(state.messages) + [AIMessage(content=error_msg)]
        if vision_container:
            with vision_container:
                st.error(f"Vision分析失败: {str(e)}")
        return state


def planning_node(state: AgentState) -> AgentState:
    """Planning Agent 节点处理函数"""
    try:
        planning_container = state.state_containers.get("planning")
        if planning_container:
            with planning_container:
                st.write("🎯 Task Planning in Progress...")

                # 检查是否有积木数据
                if "error" in state.blocks_data:
                    state.planned_solution = {"error": state.blocks_data.get("error")}
                    display_state_details(state.planned_solution, "planned_solution")

                    return state

                result = planning_agent.plan(
                    planning_container, state.blocks_data, state.task_description
                )

                state.planned_solution = result

                # 格式化解决方案消息
                solutions = result.get("solutions", [])
                solution_text = "===== The task planning result is as follows: =====\n\n"

                for idx, solution in enumerate(solutions, 1):
                    solution_text += f"【Target peg number {idx}】：\n\n"
                    for j, block in enumerate(solution.get("blocks", []), 1):
                        # 为每个积木创建带位置信息的描述
                        block_id = str(block["block_id"])
                        positions = block["positions"]
                        # 添加到文本中
                        solution_text += f"({block_id}, <{' '.join(positions)}>)\n\n"

                    solution_text += "\n"
                    state.final_solutions = solution_text

                # 如果没有解决方案，添加提示信息
                if not solutions:
                    state.final_solutions = "未找到有效的解决方案"

                display_state_details(state.planned_solution, "planned_solution")

                return state

    except Exception as e:
        logger.error(f"Error in planning node: {str(e)}")
        error_msg = f"Planning Agent执行出错: {str(e)}"
        state.planned_solution = {"error": error_msg}
        state.messages = list(state.messages) + [AIMessage(content=error_msg)]
        if planning_container:
            with planning_container:
                st.error(f"任务规划失败: {str(e)}")
        return state


def create_agent_graph() -> Graph:
    """创建智能体工作流图"""
    # 创建工作流
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("vision", vision_node)
    workflow.add_node("planning", planning_node)

    # 添加边和条件
    workflow.set_entry_point("vision")
    workflow.add_edge("vision", "planning")
    workflow.add_edge("planning", END)

    # 编译工作流
    return workflow.compile()
