import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# Set hardware device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 1. Specify your image path
IMAGE_PATH = "test_image.png"  # Update this if your file has a different name/extension
OUTPUT_PATH = "annotated_result.png"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Could not find '{IMAGE_PATH}' in {os.getcwd()}")

print(f"[INFO] Loading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")

# 2. Load Florence-2-base
model_id = "microsoft/Florence-2-base"
print(f"[INFO] Loading model {model_id} onto {device}...")

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    trust_remote_code=True,
    attn_implementation="eager"
).to(device)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. Run Object Detection (<OD>)
# Tip: Use '<DENSE_REGION_CAPTION>' instead if you want every small detail detected
task_prompt = "<OD>" 
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)

print("[INFO] Running vision detection...")
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=1,
    early_stopping=False
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
results = processor.post_process_generation(
    generated_text, 
    task=task_prompt, 
    image_size=(image.width, image.height)
)

parsed_data = results.get(task_prompt, {})
bboxes = parsed_data.get("bboxes", [])
labels = parsed_data.get("labels", [])

print(f"\n[RESULTS] Detected {len(bboxes)} objects:")
for label, box in zip(labels, bboxes):
    print(f" - {label}: {box}")

# 4. Draw bounding boxes onto the image
draw = ImageDraw.Draw(image)

for bbox, label in zip(bboxes, labels):
    x1, y1, x2, y2 = bbox
    
    # Draw red rectangle around detected object
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Draw text label above box
    draw.text((x1 + 4, max(0, y1 - 12)), label, fill="red")

# 5. Save the output
image.save(OUTPUT_PATH)
print(f"\n[SUCCESS] Annotated image saved as '{OUTPUT_PATH}'. Open it to view results!")