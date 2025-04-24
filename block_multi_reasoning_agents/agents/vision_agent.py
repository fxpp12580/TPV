import os
from typing import Dict, List, Optional, Any, Tuple, Union
from langchain_core.tools import BaseTool
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
import time
import json

from utils.image_processing import encode_image_to_base64, preprocess_image
from utils.logger import setup_logger
from config import settings

logger = setup_logger(__name__)


class Shape(BaseModel):
    """形状数据模型"""

    type: str
    position: Union[int, str]  # 允许整数或字符串类型的position
    orientation: str


class FrameShape(BaseModel):
    """积木框架中的形状数据模型"""

    type: str
    position: str  # 使用字母a-i标识位置
    orientation: str


class BlockFeature(BaseModel):
    """积木特征数据模型"""

    id: int
    layout: str
    length: int
    shapes: List[Shape]


class BlockFrameFeature(BaseModel):
    """积木框架特征数据模型"""

    total_shapes: int
    shapes: List[FrameShape]

    def validate_block_frame_attributes(self) -> bool:
        """验证积木框架属性的有效性"""
        shape_types = set(["circle", "square", "triangle"])
        orientation_types = set(["up", "left", "right"])
        valid_positions = set("abcdefghi")

        for shape in self.shapes:
            if shape.type not in shape_types:
                logger.error(f"Invalid shape type found in frame: {shape.type}")
                return False
            if shape.orientation not in orientation_types:
                logger.error(f"Invalid orientation found in frame: {shape.orientation}")
                return False
            if shape.position not in valid_positions:
                logger.error(f"Invalid position found in frame: {shape.position}")
                return False

        return True


class BlocksOutput(BaseModel):
    """积木输出数据模型"""

    total_blocks: int
    blocks: List[BlockFeature]

    def validate_block_attributes(self) -> bool:
        """验证积木属性的有效性"""
        shape_types = set(["circle", "square", "triangle"])
        layout_types = set(["horizontal", "single"])
        orientation_types = set(["left", "right", "up"])

        for block in self.blocks:
            if block.layout not in layout_types:
                logger.error(f"Invalid layout type found: {block.layout}")
                return False

            for shape in block.shapes:
                if shape.type not in shape_types:
                    logger.error(f"Invalid shape type found: {shape.type}")
                    return False
                if shape.orientation not in orientation_types:
                    logger.error(f"Invalid orientation type found: {shape.orientation}")
                    return False

            if block.length != len(block.shapes):
                logger.error(
                    f"Block length mismatch: expected {block.length}, got {len(block.shapes)}"
                )
                return False

        return True


class VisionAgent(BaseTool):
    """Vision Agent for analyzing building block images"""

    name: str = Field(default="vision_agent", description="Vision analysis tool")
    description: str = Field(
        default="Analyzes images of building blocks and extracts features"
    )
    response_format: str = Field(
        default="tool_message", description="Response format type"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )

    _client: Optional[OpenAI] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(
            api_key=settings.VL_API_KEY,
            base_url=settings.VL_BASE_URL,
        )

    def _run(
        self, block_img_path: str, block_frame_img_path: str, **kwargs: Any
    ) -> Dict:
        """运行Vision Agent分析图片"""
        try:
            # 加载并分析图片
            block_img = Image.open(block_img_path)
            block_frame_img = Image.open(block_frame_img_path)

            # 分析结果初始化
            result = {
                "status": "success",
                "analysis": {
                    "blocks": {"success": False, "content": None, "data": None},
                    "frame": {"success": False, "content": None, "data": None},
                },
            }

            # 处理积木分析结果
            block_img_result = self._analyze_block_img_with_retry(block_img)
            if block_img_result and block_img_result.validate_block_attributes():
                result["analysis"]["blocks"] = {
                    "success": True,
                    "content": self._generate_block_analysis_description(
                        block_img_result
                    ),
                    "data": block_img_result.model_dump(),
                }
            else:
                result["analysis"]["blocks"] = {
                    "success": False,
                    "content": "Failed to analyze block image",
                    "data": None,
                }

            # 处理框架分析结果
            block_frame_img_result = self._analyze_block_frame_img_with_retry(
                block_frame_img
            )
            if (
                block_frame_img_result
                and block_frame_img_result.validate_block_frame_attributes()
            ):
                result["analysis"]["frame"] = {
                    "success": True,
                    "content": self._generate_block_frame_analysis_description(
                        block_frame_img_result
                    ),
                    "data": block_frame_img_result.model_dump(),
                }
            else:
                result["analysis"]["frame"] = {
                    "success": False,
                    "content": "Failed to analyze frame image",
                    "data": None,
                }

            # 如果两种分析都失败，则返回整体失败状态
            if (
                not result["analysis"]["blocks"]["success"]
                and not result["analysis"]["frame"]["success"]
            ):
                return {
                    "status": "error",
                    "message": "Both block and frame analysis failed",
                    "analysis": result["analysis"],
                }
            logger.info("=========== vision agent run result ===========")
            logger.info(result)

            return result
        except Exception as e:
            error_message = f"Error in vision agent: {str(e)}"
            logger.error(error_message)
            return {
                "status": "error",
                "message": error_message,
                "analysis": {
                    "blocks": {"success": False, "content": None, "data": None},
                    "frame": {"success": False, "content": None, "data": None},
                },
            }

    def _generate_block_analysis_description(self, result: BlocksOutput) -> str:
        """生成分析结果描述"""
        total_blocks = result.total_blocks

        # 统计不同布局类型的积木数量
        layout_counts = {}
        for block in result.blocks:
            layout_counts[block.layout] = layout_counts.get(block.layout, 0) + 1

        # 生成描述
        description = [
            f"分析完成! 共检测到{total_blocks}个积木。",
            "积木布局分布:",
        ]

        for layout, count in layout_counts.items():
            description.append(f"- {layout}: {count}个")

        return "\n".join(description)

    def _generate_block_frame_analysis_description(
        self, result: BlockFrameFeature
    ) -> str:
        """生成框架分析结果描述"""
        total_shapes = len(result.shapes)

        # 统计不同形状类型的数量
        shape_counts = {}
        orientation_counts = {}
        for shape in result.shapes:
            shape_counts[shape.type] = shape_counts.get(shape.type, 0) + 1
            orientation_counts[shape.orientation] = (
                orientation_counts.get(shape.orientation, 0) + 1
            )

        # 生成描述
        description = [
            f"框架分析完成! 共检测到{total_shapes}个基础形状。",
            "形状分布:",
        ]

        for shape_type, count in shape_counts.items():
            description.append(f"- {shape_type}: {count}个")

        description.append("\n朝向分布:")
        for orientation, count in orientation_counts.items():
            description.append(f"- {orientation}: {count}个")

        return "\n".join(description)

    def _analyze_block_img_with_retry(
        self,
        image: Image.Image,
    ) -> Optional[BlocksOutput]:
        """带重试机制的图片分析"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._analyze_block_image(image)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                break

        logger.error(
            f"All {self.max_retries} attempts failed. Last error: {str(last_error)}"
        )
        return None

    def _analyze_block_frame_img_with_retry(
        self,
        image: Image.Image,
    ) -> Optional[BlockFrameFeature]:
        """带重试机制的框架图片分析"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._analyze_block_frame_image(image)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                break

        logger.error(
            f"All {self.max_retries} attempts failed. Last error: {str(last_error)}"
        )
        return None

    def _analyze_block_image(
        self,
        image: Image.Image,
    ) -> Optional[BlocksOutput]:
        """使用视觉模型分析图片"""
        try:
            # 预处理图片
            processed_image = preprocess_image(image)
            image_base64 = encode_image_to_base64(processed_image)

            # 构建提示词
            prompt = """分析这张积木图片，识别：
1. 积木总数
2. 每个积木的：
    - ID编号（从1开始）
    - 布局方式（横向/单个）
    - 包含的形状（圆形/方形/三角形）及其属性：
        * 位置（整数）
        * 朝向（上/左/右）

以JSON结构化的方式输出信息。
注意：
- 形状类型必须是: circle, square, triangle之一
- 布局类型必须是: horizontal, single之一
- 朝向类型必须是: up, left, right之一（circle和square都是up）
- 每个积木的length必须等于其包含的形状数量
- position必须为整数

示例输出：
{
  "total_blocks": 19,
  "blocks": [
    {
      "id": 1,
      "layout": "horizontal",
      "length": 3,
      "shapes": [
        {"type": "circle", "position": 1, "orientation": "up"},
        {"type": "square", "position": 2, "orientation": "right"},
        {"type": "triangle", "position": 3, "orientation": "left"}
      ]
    },
    {
      "id": 2,
      "layout": "horizontal",
      "length": 2,
      "shapes": [
        {"type": "square", "position": 1, "orientation": "up"},
        {"type": "square", "position": 2, "orientation": "up"}
      ]
    },
    {
      "id": 3,
      "layout": "single",
      "length": 1,
      "shapes": [
        {"type": "triangle", "position": 1, "orientation": "right"}
      ]
    },
    // ... 其他积木数据
    {
      "id": 19,
      "layout": "single",
      "length": 1,
      "shapes": [
        {"type": "circle", "position": 1, "orientation": "up"}
      ]
    }
  ]"""

            # 调用API
            completion = self._client.chat.completions.create(
                model=settings.VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # 解析响应
            response = completion.choices[0].message.content
            result = self._parse_block_response(response)
            # print(f"图片识别结果：{result.model_dump()}")

            return result

        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            raise

    def _analyze_block_frame_image(
        self,
        image: Image.Image,
    ) -> Optional[BlockFrameFeature]:
        """分析积木框架图片"""
        try:
            # 预处理图片
            processed_image = preprocess_image(image)
            image_base64 = encode_image_to_base64(processed_image)

            # 构建提示词
            prompt = """分析这张积木框架图片，识别：
1. 每个形状的：
    - 类型（圆形/方形/三角形）
    - 位置（使用字母a-i从左到右、从上到下编号）
    - 朝向（上/左/右）
2. 形状总数

请以JSON格式输出，格式如下：
{
    "total_shapes": 9,
    "shapes": [
        {"type": "circle", "position": "a", "orientation": "up"},
        {"type": "circle", "position": "b", "orientation": "up"},
        {"type": "square", "position": "c", "orientation": "up"},
        {"type": "circle", "position": "d", "orientation": "up"},
        {"type": "square", "position": "e", "orientation": "up"},
        {"type": "square", "position": "f", "orientation": "up"},
        {"type": "triangle", "position": "g", "orientation": "up"},
        {"type": "triangle", "position": "h", "orientation": "up"},
        {"type": "triangle", "position": "i", "orientation": "up"}
    ]
}

注意：
- 形状类型必须是: circle, square, triangle之一
- 位置必须是字母a-i
- 朝向必须是: up, left, right之一
- 圆形和方形的朝向都是up
- 三角形的朝向根据其尖端方向判断
- **total_shapes=len(shapes)**
"""

            # 调用API
            completion = self._client.chat.completions.create(
                model=settings.VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # 解析响应
            response = completion.choices[0].message.content
            result = self._parse_block_frame_response(response)
            return result

        except Exception as e:
            logger.error(f"Error in frame image analysis: {str(e)}")
            raise

    def _parse_block_response(self, response: str) -> BlocksOutput:
        """解析模型响应"""
        try:
            # 清理响应文本，提取JSON部分
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]

            data = json.loads(json_str)

            # 构建积木列表
            blocks = []
            for block in data["blocks"]:
                # 创建Shape对象列表
                shapes = [Shape(**shape) for shape in block["shapes"]]

                # 创建BlockFeature对象
                blocks.append(
                    BlockFeature(
                        id=block["id"],
                        layout=block["layout"],
                        length=block["length"],
                        shapes=shapes,
                    )
                )

            # 创建带有所有类型的输出对象
            output = BlocksOutput(total_blocks=data["total_blocks"], blocks=blocks)

            return output

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

    def _parse_block_frame_response(self, response: str) -> BlockFrameFeature:
        """解析框架图片的模型响应"""
        try:
            # 清理响应文本，提取JSON部分
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]

            data = json.loads(json_str)

            # 创建Shape对象列表
            shapes = [FrameShape(**shape) for shape in data["shapes"]]

            # 创建输出对象
            output = BlockFrameFeature(total_shapes=data["total_shapes"], shapes=shapes)

            return output

        except Exception as e:
            logger.error(f"Error parsing frame response: {str(e)}")
            raise

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """异步运行方法"""
        raise NotImplementedError("Async not implemented")

    def _format_tool_result(self, content: str, data: Optional[Dict]) -> Dict:
        """格式化工具返回结果"""
        return {"content": content, "additional_kwargs": {"data": data}}
