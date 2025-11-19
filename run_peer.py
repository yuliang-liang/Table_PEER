import json
import os
import time
from tqdm import tqdm
from utils.models import llm_init
from utils.chain import dynamic_chain_exec_one_sample
from utils.evaluation import evaluate_tqa_tfv_one_item
from config import DATASET_PATH, IMAGE_FOLDER, QWEN_MODEL_PATH
from utils.data_utils import filter_samples


# calculate the correct rate
results = {
    'TQA': {'correct': 0, 'total': 0},
    'TFV': {'correct': 0, 'total': 0},
    'datasets': {}  # will be filled dynamically
}

if __name__ == "__main__":

    # load dataset
    dataset_path = DATASET_PATH
    image_folder = IMAGE_FOLDER    
    
    model_path = QWEN_MODEL_PATH
    model_name = model_path.split("/")[-1]

    method = "peer"
    
    # task_type = "TQA" 
    # dataset_name = "TAT-QA" # TABMWP / WTQ  / HiTab / TAT-QA
    #task_type = "TFV"
    #dataset_name = "TabFact" # TabFact / InfoTabs 

    task_type = "all"
    dataset_name = "all"

    sample_ratio = 1.0  # 可以改为任意0到1之间的值，比如0.5表示使用50%的样本

    # log file
    log_path = "./logs/" + time.strftime("%Y%m%d-%H%M%S") + "_" + model_name + "_" + method + "_" + \
        task_type + "_" + dataset_name + ".jsonl"
    answer_file = open(log_path, "w")

    # load samples
    with open(dataset_path, "r") as f:
        samples = json.load(f)
    print("Total samples: ", len(samples))
    # only hold samples for TQA and TFV
    #samples = [s for s in samples if s["task_type"] in ["TQA", "TFV"]]
    #samples = [s for s in samples if s["dataset_name"] in ["TabFact", "TABMWP", "InfoTabs" "WTQ", "HiTab", "TAT-QA"]]
    print("Filtered TQA and TFV samples: ", len(samples))
    samples = filter_samples(samples, task_type, dataset_name, sample_ratio=sample_ratio)

    # optional shuffle
    # random.shuffle(samples)

    llm = llm_init(model_path)

    for i, sample in enumerate(tqdm(samples)):
        answer, dynamic_chain_log = dynamic_chain_exec_one_sample(
            sample, llm, task_type=sample["task_type"], debug=True
        )

        answer_history = [log["response"] for log in dynamic_chain_log if log.get("stage") in ["summary", "aggregation_reasoning"]]

        # evaluate final answer
        is_correct,_ = evaluate_tqa_tfv_one_item(answer, sample["answer_list"], sample["task_type"])

        # print results
        print("Question: ", sample.get('original_query'), flush=True)
        print("Output: ", answer, flush=True)
        print("Correct Answer: ", sample.get("answer_list"), flush=True)
        print("is_correct: ", is_correct, flush=True)

        # save to file
        answer_file.write(json.dumps(
            {
                "item_id": sample.get("item_id"),
                "image": sample.get("image_id"),
                "original_query": sample.get("original_query"),
                "output": answer,
                "output_history": answer_history,
                "label": sample.get("answer_list"),
                "is_correct": is_correct,
                "task_type": sample.get("task_type"),
                "dataset_name": sample.get("dataset_name"),
                #"feedback": feedback,
            }
        ) + "\n")
        answer_file.flush()

        # Update statistics
        task = sample.get("task_type")
        dataset = sample.get("dataset_name")
        
        results[task]['total'] += 1
        if is_correct:
            results[task]['correct'] += 1
            
        if dataset not in results['datasets']:
            results['datasets'][dataset] = {'correct': 0, 'total': 0}
        results['datasets'][dataset]['total'] += 1
        if is_correct:
            results['datasets'][dataset]['correct'] += 1
        
        # Print current results
        print("\n=== Current Results ===")
        print("\nBy Task Type:")
        for t in ['TQA', 'TFV']:
            if results[t]['total'] > 0:
                acc = results[t]['correct'] / results[t]['total']
                print(f"{t}: {results[t]['correct']}/{results[t]['total']} = {acc:.4f}")
        
        print("\nBy Dataset:")
        for d, stats in results['datasets'].items():
            acc = stats['correct'] / stats['total']
            print(f"{d}: {stats['correct']}/{stats['total']} = {acc:.4f}")
        print("\n")
            
    # Print final results
    print("\n=== Final Results ===")
    print("\nBy Task Type:")
    for task in ['TQA', 'TFV']:
        if results[task]['total'] > 0:
            acc = results[task]['correct'] / results[task]['total']
            print(f"{task}: {results[task]['correct']}/{results[task]['total']} = {acc:.4f}")
    
    print("\nBy Dataset:")
    for dataset, stats in results['datasets'].items():
        acc = stats['correct'] / stats['total']
        print(f"{dataset}: {stats['correct']}/{stats['total']} = {acc:.4f}")
        
    answer_file.close()