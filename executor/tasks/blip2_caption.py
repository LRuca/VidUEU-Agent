# executor/tasks/blip2_caption.py

import os
from typing import List, Union

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class Blip2Captioner:
    """
    基于 Salesforce/blip2-opt-2.7b 的图片描述模块。

    主要对外接口：
      - caption_image(image_path)    # 单张图片 -> caption（T3 用到）
      - caption_batch(image_paths)   # 多张图片 -> caption 列表
      - caption(...)                 # 向后兼容的别名
    """

    def __init__(self, config):
        self.model_path = config["models"]["blip2_model"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        print(f"🟢 [Blip2Captioner] configured model dir: {self.model_path}")

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[Blip2Captioner] model path not found: {self.model_path}")
        self.processor = Blip2Processor.from_pretrained(self.model_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    # ------------ 内部工具函数 ------------

    @staticmethod
    def _load_image(image: Union[str, Image.Image]) -> Image.Image:
        """把路径 / PIL.Image 统一成 RGB PIL.Image"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"[Blip2Captioner] 图像不存在: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"[Blip2Captioner] 不支持的图片类型: {type(image)}")

    # ------------ 对外主要接口 ------------

    @torch.inference_mode()
    def caption_image(self, image: Union[str, Image.Image], max_new_tokens: int = 64) -> str:
        """
        单张图片生成 caption。
        image 可以是路径或 PIL.Image。
        """
        self._ensure_model()
        img = self._load_image(image)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return caption

    @torch.inference_mode()
    def caption_batch(
        self, images: List[Union[str, Image.Image]], max_new_tokens: int = 64
    ) -> List[str]:
        """
        多张图片批量生成 caption。
        """
        self._ensure_model()
        if not images:
            return []

        pil_images = [self._load_image(im) for im in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        captions = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return [c.strip() for c in captions]

    # 向后兼容：如果之前有代码调用 caption(...)，让它等价于 caption_image
    def caption(self, image: Union[str, Image.Image], max_new_tokens: int = 64) -> str:
        return self.caption_image(image, max_new_tokens=max_new_tokens)
