import json
import os
import time
import copy
import random
from tqdm import tqdm
from utils.models import llm_init, llm_generate
from utils.chain import generate_prompt_for_stage, dynamic_chain_exec_one_sample
from utils.evaluation import evaluate_tqa_tfv_one_item

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == "__main__":
        
    # loda dataset
    dataset_path = "/your_path/Datasets/MMTab/MMTab-eval_test_data_49K.json"
    image_folder = "/your_path/Datasets/MMTab/all_test_image/"
    
    model_name = "Qwen2-VL-7B-Instruct"
    model_path = "/your_path/HF_models/Qwen2-VL-7B-Instruct"
    # model_name = "Llama-3.2-11B-Vision-Instruct"
    # model_path = "/your_path/HF_models/Llama-3.2-11B-Vision-Instruct"
    
    method = "correct" 
    
    task_type = "TQA" 
    dataset_name = "TAT-QA" # TABMWP / WTQ  / HiTab / TAT-QA
    task_type = "TFV"
    dataset_name = "InfoTabs" # TabFact / InfoTabs 
    
    # log file
    answer_path = "./logs/" + time.strftime("%Y%m%d-%H%M%S") + "_" + method + "_" + model_name + "_" + \
        task_type + "_" + dataset_name + ".jsonl"
    answer_file = open(answer_path, "w")

    
    with open(dataset_path, "r") as f:
        samples = json.load(f)
    print("Total samples: ", len(samples))
    
    # filter samples by task_type and dataset_name
    samples = [s for s in samples if s["task_type"] == task_type and s["dataset_name"] == dataset_name]
    print("task_type: ", task_type, "dataset_name: ", dataset_name)
    print("Filtered samples: ", len(samples))
    
    # shuffle samples
    #random.shuffle(samples)
    demo = samples[0]
    
    # HF qwen2vl model
    llm = llm_init(model_path)
    
    # calculate the correct rate
    correct_count = 0
    total_count = 0
    
    # tqdm for progress bar
    for i, sample in enumerate(tqdm(samples)):
        feedback, answer, dynamic_chain_log = dynamic_chain_exec_one_sample(sample, image_folder, llm, llm_options=None,task_type=task_type, strategy="top", debug=True)

        answer_history = [log["response"] for log in dynamic_chain_log if log["stage"] in ["summary", "aggregation_reasoning"]]
        
        for i, ans in enumerate(answer_history):
            is_correct,_ = evaluate_tqa_tfv_one_item(answer, sample["answer_list"], task_type)
            print("Answer {}: {} is_correct: {}".format(i, ans, is_correct))
        
        # is_correct,_ = evaluate_tqa_tfv_one_item(answer, sample["answer_list"], task_type)
        # print("is_correct: ", is_correct, flush=True)
        
        # print 
        #print("Question: ",sample['original_query'], flush=True)
        #print("Output: ",answer, flush=True)
        print("Correct Answer: ", sample["answer_list"], flush=True)

        
        # save to file
        answer_file.write(json.dumps(
            {
                "item_id": sample["item_id"],
                "image": sample["image_id"],
                "original_query": sample["original_query"],
                "output": answer,
                "output_history": answer_history,
                "label": sample["answer_list"],
                "is_correct": is_correct,
                #"feedback": feedback,
                }) + "\n")

        answer_file.flush()
        if is_correct:
            correct_count += 1
        total_count += 1
        
        print("Correct count / Total count: {} / {}".format(correct_count, total_count))
        print("Correct rate: ", correct_count / total_count)

        pass

    #ans_file = open(answer_file, "w")