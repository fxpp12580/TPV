# 环境配置

创建conda环境

```
conda create -n block_agent python=3.10 -y
```

安装依赖包

```
conda activate block_agent
pip install -r requirements.txt
```

# 运行系统

```
streamlit run main.py
```

# 项目架构

```
block_multi_agents
  ├─config.py
  ├─main.py
  ├─README.md
  ├─requirements.txt
  ├─agents
  │   ├─plan_agent.py
  │   ├─vision_agent.py
  │   ├─__init__.py
  ├─core
  │   ├─graph.py
  │   ├─__init__.py
  ├─data
  │   ├─input19.png
  │   ├─input8.png
  │   └─uploads
  │       ├─input19.png
  │       └─input8.png
  ├─logs
  │   ├─agents.vision_agent_2025-01-14.log
  │   ├─agents.vision_agent_2025-01-15.log
  │   ├─core.graph_2025-01-14.log
  │   ├─core.graph_2025-01-15.log
  │   ├─utils.image_processing_2025-01-14.log
  │   ├─utils.image_processing_2025-01-15.log
  │   ├─__main___2025-01-14.log
  │   └─__main___2025-01-15.log
  ├─utils
  │   ├─image_processing.py
  │   ├─logger.py
  │   ├─__init__.py
  │   └─__pycache__
  │       ├─image_processing.cpython-310.pyc
  │       ├─logger.cpython-310.pyc
  │       └─__init__.cpython-310.pyc
  └─__pycache__
      └─config.cpython-310.pyc
```
