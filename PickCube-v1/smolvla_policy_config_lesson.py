import json
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(
    repo_id="lerobot/smolvla_base",
    filename="config.json",
    revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
    local_files_only=True,
)

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

print(type(config))
for i, j in config.items():
    print(i, j)
