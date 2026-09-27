import time
from typing import Dict, Any, List, Union, Tuple
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class PerceptionAgent:
    """
    Perception Agent utilizing Microsoft Florence-2 for real-time vision tasks,
    bounding box extraction, and spatial coordinate normalization for UAV control.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str = None,
        torch_dtype: torch.dtype = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        print(f"[PerceptionAgent] Initializing on {self.device} ({self.torch_dtype})...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        print("[PerceptionAgent] Model successfully loaded.")

    def process_frame(
        self,
        image: Union[Image.Image, str],
        task_prompt: str = "<OD>",
        text_input: str = None,
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Executes a vision task on an input frame and computes spatial offsets.

        Args:
            image: PIL Image object or path to image file.
            task_prompt: Task token (e.g., '<OD>', '<CAPTION_TO_PHRASE_GROUNDING>').
            text_input: Optional text input string required for grounding tasks.
            max_new_tokens: Maximum tokens for model text generation.

        Returns:
            Dictionary containing raw detections, normalized coordinates,
            pixel offsets from image center, frame dimensions, and inference latency.
        """
        start_time = time.time()

        # Handle file path or PIL image input
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError("Input image must be a PIL Image or file path string.")

        width, height = image.width, image.height
        center_x, center_y = width / 2.0, height / 2.0

        # Construct prompt
        prompt = task_prompt if text_input is None else task_prompt + text_input

        # Prepare inputs
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        # Generate vision tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
            )

        # Decode & post-process
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        raw_results = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(width, height)
        )

        parsed_data = raw_results.get(task_prompt, {})
        bboxes = parsed_data.get("bboxes", [])
        labels = parsed_data.get("labels", [])

        # Process spatial metrics for downstream agents
        detections: List[Dict[str, Any]] = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0

            # Pixel offset from frame center (-1.0 to 1.0 scale)
            offset_x = (bbox_center_x - center_x) / center_x
            offset_y = (bbox_center_y - center_y) / center_y

            detections.append({
                "label": label,
                "bbox_pixel": [round(c, 2) for c in [x1, y1, x2, y2]],
                "bbox_normalized": [
                    round(x1 / width, 4),
                    round(y1 / height, 4),
                    round(x2 / width, 4),
                    round(y2 / height, 4),
                ],
                "center_pixel": [round(bbox_center_x, 2), round(bbox_center_y, 2)],
                "center_offset_normalized": [round(offset_x, 4), round(offset_y, 4)],
            })

        latency = time.time() - start_time

        return {
            "frame_size": {"width": width, "height": height},
            "task": task_prompt,
            "num_detections": len(detections),
            "detections": detections,
            "latency_ms": round(latency * 1000, 2),
        }


# Quick test bench when executing this module directly
if __name__ == "__main__":
    import os

    test_image_path = "test_image.png"
    if os.path.exists(test_image_path):
        agent = PerceptionAgent()
        results = agent.process_frame(test_image_path, task_prompt="<OD>")
        
        print("\n--- Perception Agent Output ---")
        print(f"Frame Dimensions : {results['frame_size']}")
        print(f"Inference Latency: {results['latency_ms']} ms")
        print(f"Objects Found    : {results['num_detections']}")
        if results['detections']:
            print("First Detection Sample:", results['detections'][0])
    else:
        print(f"Test image '{test_image_path}' not found. Run from project root.")