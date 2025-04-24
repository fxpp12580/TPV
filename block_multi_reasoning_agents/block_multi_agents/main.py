from pathlib import Path
import streamlit as st

from core.display import display_workflow_progress
from core.graph import AgentState, create_agent_graph
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_with_progress(
    block_img_path: str, block_frame_img_path: str, task_description: str
):
    """带进度显示的工作流运行函数"""
    try:
        # 创建占位显示区域
        vision_container = st.empty()
        planning_container = st.empty()
        execution_container = st.empty()
        verification_container = st.empty()

        # 初始化状态容器
        state_containers = {
            "vision": vision_container,
            "planning": planning_container,
            "execution": execution_container,
            "verification": verification_container,
        }

        # 初始化状态
        initial_state = AgentState(
            block_img_path=block_img_path,
            block_frame_img_path=block_frame_img_path,
            task_description=task_description,
            state_containers=state_containers,  # 传入状态容器
        )

        # 创建并运行工作流
        graph = create_agent_graph()
        final_state = graph.invoke(initial_state)

        return dict(final_state)
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        return {
            "error": f"工作流执行失败: {str(e)}",
            "block_img_path": block_img_path,
            "block_frame_img_path": block_frame_img_path,
            "task_description": task_description,
        }


def main():
    st.set_page_config(
        page_title="TPV System",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("TPV Task Reasoning System")
    st.markdown("---")

    # 主界面配置
    col1, col2 = st.columns(2)

    with col1:
        # 文件上传
        block_img = st.file_uploader(
            "Upload Image", type=["png", "jpg", "jpeg"], help="支持PNG、JPG格式的图片"
        )

    with col2:
        # 文件上传
        block_frame_img = st.file_uploader(
            "Upload the Target Pegs Image",
            type=["png", "jpg", "jpeg"],
            help="支持PNG、JPG格式的图片",
        )
    # 任务描述
    task_description = st.text_area(
        "Task Description",
        placeholder="例如：使用5-9块积木拼成一个正方形",
        help="请详细描述您想要完成的积木拼图任务",
    )

    if block_img and block_frame_img and task_description:
        # 保存上传的图片
        save_block_path = Path("data/uploads") / block_img.name
        save_block_frame_path = Path("data/uploads") / block_frame_img.name
        save_block_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_block_path, "wb") as f:
            f.write(block_img.getbuffer())
        with open(save_block_frame_path, "wb") as f:
            f.write(block_frame_img.getbuffer())

        # 显示图片
        col3, col4 = st.columns(2)
        with col3:
            st.image(
                save_block_path, caption="Uploaded Image", use_container_width=True
            )
        with col4:
            st.image(
                save_block_frame_path,
                caption="Uploaded the Target Pegs Image",
                use_container_width=True,
            )

        # 运行按钮
        if st.button("Begin Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing in Progress..."):
                try:
                    # 创建进度条
                    progress_bar = st.progress(0)
                    st.write("🔄 Commencing Multi-Agent Workflow Execution...")

                    # 运行工作流
                    result = run_with_progress(
                        str(save_block_path),
                        str(save_block_frame_path),
                        task_description,
                    )

                    progress_bar.progress(100)

                    # 显示结果
                    if result:
                        if "error" not in result:
                            st.success("✅ Workflow Execution Completed!")
                        display_workflow_progress(result)
                    else:
                        st.error("❌ Processing Failed, Please Retry")

                except Exception as e:
                    logger.error(f"Error in workflow: {str(e)}")
                    st.error(f"❌ An Error Has Occurred: {str(e)}")


if __name__ == "__main__":
    main()
