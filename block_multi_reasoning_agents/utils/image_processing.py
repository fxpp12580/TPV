import base64
from io import BytesIO

from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    预处理图片：调整大小，确保格式正确
    """
    try:
        # 确保图片不会太大
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 如果是RGBA格式，转换为RGB
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def encode_image_to_base64(image: Image.Image) -> str:
    """
    将PIL Image转换为base64字符串
    """
    try:
        # 创建一个字节缓冲区
        buffered = BytesIO()
        # 保存图片到缓冲区
        image.save(buffered, format="JPEG")
        # 获取base64编码
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise


def load_image(image_path: str) -> Image.Image:
    """
    加载图片文件
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {str(e)}")
        raise
