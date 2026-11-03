import os
import json
from pathlib import Path

class DataAssembler:
    def __init__(self, config):
        self.config = config
        self.output_dir = config["paths"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.include_caption = config["tasks"]["T5_assemble"].get("include_caption", True)
        self.include_ocr = config["tasks"]["T5_assemble"].get("include_ocr", True)
        self.include_detection = config["tasks"]["T5_assemble"].get("include_detection", True)

    def run(self, frames, subtitles, detections, captions, ocr_results, video_path):
        print("🧩 T5: 整合多模态数据")
        video_id = Path(video_path).stem
        output_file = os.path.join(self.output_dir, f"{video_id}_dataset.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for frame_path in frames:
                frame_id = os.path.basename(frame_path)
                sample = {
                    "video_name": video_id,
                    "frame_id": frame_id,
                    "image_path": frame_path,
                    "sample_type": "image_text"
                }

                if self.include_caption and frame_id in captions:
                    sample["caption"] = captions[frame_id]

                if self.include_ocr and frame_id in ocr_results:
                    sample["ocr"] = ocr_results[frame_id]

                if self.include_detection and frame_id in detections:
                    sample["detections"] = detections[frame_id]

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"✅ 已生成结构化样本：{output_file}")
