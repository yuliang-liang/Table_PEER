import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
# 指定模型名称
#model_name = "Qwen/Qwen2-VL-72B-Instruct"
model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"


# 下载模型（会自动缓存到 ~/.cache/huggingface/hub/ 下）

#cache_dir = "/Volumes/MBP2015/Project/"  # 这里修改为你想要的路径
#local_dir="/Volumes/MBP2015/Project/HF_Models/Qwen2-VL-72B-Instruct"
local_dir="/Volumes/MBP2015/Project/HF_Models/Llama-3.2-90B-Vision-Instruct"


model_path = snapshot_download(
    repo_id=model_name,
    #cache_dir=cache_dir,# 直接指定缓存目录
    local_dir=local_dir, 
    resume_download=True,  # 如果中断可续传
    local_dir_use_symlinks=False  # 可避免符号链接问题
)

print(f"✅ 模型已下载到: {model_path}")

