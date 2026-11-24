import h5py
import torch

from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

data_path = "/home/hatepp/SSD512/vla-learning/PickCube-v1/rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"

image_path = "/home/hatepp/SSD512/vla-learning/PickCube-v1/rgbd_单条示范/eb922225832cb874e2c7038c5075fdc2.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with h5py.File(data_path, "r") as data:
    image_array = data["traj_0/obs/sensor_data/base_camera/rgb"][0]

image = Image.fromarray(image_array)
image = Image.open(image_path).convert("RGB")

model_path = "/home/hatepp/SSD512/vla-learning/models/SmolVLM2-500M"
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": (
                " Find the transparent drink bottle with the blue and green label. "
                "The image is 1919 pixels wide and 1080 pixels high. "
                "Give the bounding box of the whole bottle as "
                "[x_min, y_min, x_max, y_max] in original image pixels. "
                "If you are unsure, say unknown."
            )},
        ],
    }
]
text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs = inputs.to(device)

for name, value in inputs.items():
    print(name, value.shape, value.dtype)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    local_files_only=True,
)
model = model.to(device)
model.eval()
print("模型类型:", type(model).__name__)
print("模型设备:", next(model.parameters()).device)
print("参数类型:", next(model.parameters()).dtype)
print("视觉编码器:", type(model.model.vision_model).__name__)
print("视觉连接器:", type(model.model.connector).__name__)
print("文本 embedding:", type(model.get_input_embeddings()).__name__)
with torch.inference_mode():
    image_output = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        pixel_attention_mask=inputs["pixel_attention_mask"],
    )
print(image_output.last_hidden_state.shape)
print(image_output.pooler_output.shape)

with torch.inference_mode():
    text_embeddings = model.get_input_embeddings()(inputs["input_ids"])
print(text_embeddings.shape)

# with torch.inference_mode():
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         do_sample=False,
#     )
# input_length = inputs["input_ids"].shape[1]
# new_ids = output_ids[:, input_length:]
# print("新生成的 token 数:", new_ids.shape[1])
# print("最后一个 token ID:", new_ids[0, -1].item())
# print("模型结束 token ID:", model.generation_config.eos_token_id)
# answers = processor.batch_decode(new_ids, skip_special_tokens=True)
# print("模型回答:", answers[0])
