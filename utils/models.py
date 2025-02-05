from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


def qwen2vl_init(model_name):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_name)
    
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name, 
                                            min_pixels=min_pixels, max_pixels=max_pixels
                                            )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    llm = {
        "model_name": "Qwen2-VL-7B-Instruct",
        "model": model,
        "processor": processor,
        "messages": messages,
    }
    return llm

def qwen2vl_generate(llm, text, image=None, options=None):

    model = llm["model"]
    processor = llm["processor"]
    messages = llm["messages"]
    
    _message = [{'role': "user", 'content': new_content}]
    # Preparation for inference
    # messages[0]["content"][0]["image"] = image 
    # messages[0]["content"][1]["text"] = text
    
    new_content = []
    if image is None:
        text_item = {'type': 'text', 'text': text}
        new_content.append(text_item)
    else :
        text_item = {'type': 'text', 'text': text}
        image_item = {'type': 'image', 'image': image}
        new_content.append(image_item)
        new_content.append(text_item)
    
    text = processor.apply_chat_template(
        _message, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(_message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024, 
                                    temperature=0.1
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
    if "Llama-3.2-11B-Vision-Instruct" in model_name:
        llm = llama3_vision_init(model_name)
    assert llm is not None
    return llm



def llm_generate(llm, text, image=None, options=None):
    if llm["model_name"] == "Qwen2-VL-7B-Instruct":
        output_text = qwen2vl_generate(llm, text, image, options)
    if llm["model_name"] == "Llama-3.2-11B-Vision-Instruct":
        output_text = llama3_vision_generate(llm, text, image, options)
    assert output_text is not None
    return output_text


if __name__ == "__main__":
    model_name = "/gly/guogb/lyl/HF_models/Llama-3.2-11B-Vision-Instruct"
    llm = llm_init(model_name)
    
    text = "How much was the 2019 financing costs ?"
    image = "/gly/guogb/lyl/Datasets/MMTab/all_test_image/TAT-QA_02913daf-213d-46e7-bf29-a65a8e64550f.jpg"
    output_text = llm_generate(llm, text,)
    print(output_text)