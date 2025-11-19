
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

#from PIL import Image
#from transformers import MllamaForConditionalGeneration, AutoProcessor


def qwen2vl_init(model_name):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="mps")
    processor = AutoProcessor.from_pretrained(model_name)
    
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name, 
                                            min_pixels=min_pixels, max_pixels=max_pixels
                                            )
    llm = {
        #"model_name": "Qwen2-VL-7B-Instruct",
        "model": model,
        "processor": processor,
        "generator": qwen2vl_generate
    }
    return llm

def qwen2vl_generate(llm, text, image=None, options=None):
    """
    调用 Qwen2-VL 模型生成结果，可兼容多种视觉问答任务。

    参数:
        llm: dict，包含模型和预处理器，如 {"model": model, "processor": processor}
        text: str，用户输入文本
        image: 可选，图像输入（路径、URL或tensor）
        options: 可选，生成参数，如 temperature、max_tokens 等
    """
    model = llm["model"]
    processor = llm["processor"]
    
    content = [{'type': 'text', 'text': text}]

    # ==== 构建 messages ====
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
        
    messages[0]["content"].append({"type": "text", "text": text})
    
    # ==== 应用 prompt 模板 ====

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024, 
                                    #temperature=0.1
                                    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def llama3_vision_init(model_name):
    model_id = model_name
    #model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]
    
    llm = {
        "model_name": "Llama-3.2-11B-Vision-Instruct",
        "model": model,
        "processor": processor,
        "messages": messages,
    }
    return llm
    

def llama3_vision_generate(llm, text, image=None, options=None):
    
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    model = llm["model"]
    processor = llm["processor"]
    #messages = llm["messages"] # need deep copy
    _messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]
    messages = _messages
    
    # Preparation for inference
    messages[0]["content"][0]["image"] = image 
    messages[0]["content"][1]["text"] = text
    
    if image is not None:
        image = Image.open(image)
    else:
        # delete  image token
        messages[0]["content"] = messages[0]["content"][1:]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=512,
                            temperature=0.1)# max_new_tokens=30
    #print(processor.decode(output[0]))
    
    output_text = processor.decode(output[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
    return output_text

def llm_init(model_name):
    if "Qwen2-VL-7B-Instruct" in model_name:
        llm = qwen2vl_init(model_name)
    elif "Llama-3.2-11B-Vision-Instruct" in model_name:
        llm = llama3_vision_init(model_name)
    elif "chatgpt" in model_name or "gpt-4o-mini" in model_name:
        llm = chatgpt_init(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return llm



def llm_generate(llm, text, image=None, options=None):
    generator = llm["generator"]
    output_text = generator(llm, text, image, options)
    return output_text



# ================== ChatGPT API ==================
def chatgpt_init(model_name):
    # model_name: "chatgpt" or "gpt-4"
    
    API_KEY = "sk-TxmUKMUzrkuLwuU1g6hChwqgkdSSBH95QD5wf9e8LdhgCkf1"

    # API 基础地址
    BASE_URL = "https://www.dmxapi.cn/"
    # 聊天补全接口端点
    API_ENDPOINT = BASE_URL + "v1/chat/completions"

    llm = {
        "model_name": model_name,
        "generator": chatgpt_generate,
        "api_key": API_KEY,
        "api_endpoint": API_ENDPOINT
    }
    return llm

def chatgpt_generate(llm, text, image=None, options=None):

    import base64
    import requests

    api_key = llm["api_key"]
    model_name = llm["model_name"]

    # 编码图片
    image_data = ""
    if image:
        with open(image, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    # 构造请求载荷
    content = [
        {"type": "text", "text": text}
    ]
    if image_data:
        ext = os.path.splitext(image)[1].lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_data}"}
        })

    payload = {
        "model": model_name,  # 可根据需要切换模型名
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.1,  # 默认更确定
        "max_tokens": 1024
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(
            llm["api_endpoint"],
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        print(f"ChatGPT API调用失败: {e}", flush=True)
        return ""




if __name__ == "__main__":
    model_name = "/Volumes/Lenovo/Project/HF_models/Qwen2-VL-7B-Instruct"
    llm = llm_init(model_name)
    #llm = llm_init("gpt-4o-mini")

    # text = "How much was the 2019 financing costs?"
    # image = "/Volumes/Lenovo/Project/datasets/MMTab/all_test_image/TAT-QA_02913daf-213d-46e7-bf29-a65a8e64550f.jpg"
    text = "northern saskatchewan and northern manitoba had consistently higher violent crime rates than the territories for males and females of all major age groups, what proportion did northern saskatchewan have the highest rate for canadians in the territories overall?"
    image = "/Volumes/Lenovo/Project/datasets/MMTab/all_test_image/HiTab_1355.jpg"
    output_text = llm_generate(llm, text, image)
    print(output_text)