import json
from typing import Any, Dict, List

import jieba
import streamlit as st


def display_state_details(state_data: Dict, section_title: str):
    """显示状态详细信息"""
    if not state_data:
        return

    with st.expander(f"{section_title} Detailed Information", expanded=False):
        # 检查错误
        if "error" in state_data:
            st.error(f"❌ {state_data['error']}")
            return

        # 针对不同类型的状态数据进行专门处理
        if "blocks_data" in section_title.lower():
            st.subheader("识别到的积木信息")
            st.metric("积木总数", state_data.get("total_blocks", 0))
            st.json(state_data)

        elif "planned_solution" in section_title.lower():
            st.subheader("规划方案")
            solutions = state_data.get("solutions", [])
            if solutions:
                for idx, solution in enumerate(solutions):
                    st.write(f"方案 #{idx + 1}")
                    st.json(solution)

        elif "execution_result" in section_title.lower():
            st.subheader("执行结果")
            st.metric("已放置积木数", len(state_data.get("block_placements", {})))
            st.write("积木放置位置:")
            st.json(state_data["block_placements"])

        elif "verification_result" in section_title.lower():
            st.subheader("验证结果")
            if state_data.get("status") == "verification_passed":
                st.success("✅ 验证通过")
                st.json(state_data["details"])
            else:
                st.error("❌ 验证失败")
                st.write(state_data.get("error", "未知错误"))

        # # 添加原始数据查看选项
        # with st.expander("查看原始数据"):
        #     st.json(state_data)


def display_workflow_progress(result: Dict[str, Any]):
    """显示工作流执行进度"""
    if not result:
        return

    # 检查是否有错误
    if "error" in result:
        st.error(result["error"])
        return

    # 显示各阶段结果
    stages = {
        "Vision Analysis": ("blocks_data", "🔍"),
        "Task Planning": ("planned_solution", "🎯"),
        "Plan Execution": ("execution_result", "⚙️"),
        "Result Verification": ("verification_result", "✅"),
    }

    # 创建进度追踪
    cols = st.columns(len(stages))
    for idx, (stage_name, (result_key, emoji)) in enumerate(stages.items()):
        with cols[idx]:
            stage_data = result.get(result_key, {})
            if "error" in stage_data:
                st.error(f"{emoji} {stage_name}")
            else:
                st.success(f"{emoji} {stage_name}")
                
    # 打印结果
    st.subheader("Full Results:：")
    st.write_stream(jieba.lcut(result.get('final_solutions', '')))