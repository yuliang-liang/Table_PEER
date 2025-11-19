import json
import os
import time
#from tqdm import tqdm
from utils.evaluation import evaluate_tqa_tfv_one_item
import re


# 由于llama3 生成答案不标准，需要对答案进行处理为标准格式
def transform_answer_to_standard_format(model_output, debug=False):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)

    # 未匹配
    if ret is None:
        ret = re.match('.*([\"\'\*][aA]nswer[\"\'\*]?\:)(.*).*',model_output)
        try:
            answer_str = ret.group(2).replace('*','').strip().replace("$","")
            answer_std = f'{{"answer":"{answer_str}"}}'
            # print("model_output: ", model_output)
            # print("answer_std: ", answer_std)
            return answer_std  
        except:
            if debug:
                print("match failed: ", model_output)
            return model_output
    else:
        answer_str = ret.group(1)#.replace("$","").replace(".00","")
        # '{"answer": [7]}' 这种情况无法处理
        match = re.search('"answer"\s*:\s*\[\s*(\d+)\s*\]', answer_str)
        if match:
            #print(match.group(1))  # 输出 '7'
            value = match.group(1)
            answer_std = f'{{"answer":"{value}"}}'
            return answer_std
        return answer_str
    #if ret is not None:
        
        # answer_str = ret.group(1)
        # try:
        #     answer_str = re.sub('[\"\']+',"\"",answer_str)
        #     answer_item = eval(answer_str)
        #     predicted_answer = answer_item['answer']
        #     if type(predicted_answer) != list and type(predicted_answer) == str:
        #         predicted_answer = [predicted_answer]
        #     elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
        #         predicted_answer = [str(predicted_answer)]
        #     else:
        #         pass
        # # The answer is considered to be wrong if we can not extract answer list from the json str
        # except:
        #     predicted_answer = []
        # return predicted_answer
    # else:
    #     return []


ans1_correct_count = 0
ans2_correct_count = 0
at_least_one_correct_count = 0
flip_correct_count = 0
flip_incorrect_count = 0
total_count = 0

if __name__ == '__main__':

    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241226-214705_correct_Qwen2-VL-7B-Instruct_TQA_TAT-QA.jsonl"
    #log_file = "./logs/20241225-183442_naive_Qwen2-VL-7B-Instruct_TQA_TAT-QA.jsonl"
    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241227-110650_correct_Qwen2-VL-7B-Instruct_TFV_TabFact.jsonl"
    
    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241228-155830_naive_Llama-3.2-11B-Vision-Instruct_TQA_TABMWP.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241228-155037_correct_Llama-3.2-11B-Vision-Instruct_TQA_TABMWP.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241229-133448_naive_Llama-3.2-11B-Vision-Instruct_TQA_WTQ.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241229-133506_naive_Llama-3.2-11B-Vision-Instruct_TQA_HiTab.jsonl"
    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241229-200350_naive_Llama-3.2-11B-Vision-Instruct_TQA_TAT-QA.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-003213_naive_Llama-3.2-11B-Vision-Instruct_TFV_InfoTabs.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-003158_naive_Llama-3.2-11B-Vision-Instruct_TFV_TabFact.jsonl"
    # log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-120725_correct_Llama-3.2-11B-Vision-Instruct_TQA_WTQ.jsonl"
    # log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-120804_correct_Llama-3.2-11B-Vision-Instruct_TQA_HiTab.jsonl"
    
    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-225213_correct_Llama-3.2-11B-Vision-Instruct_TFV_InfoTabs.jsonl"
    log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-164317_correct_Llama-3.2-11B-Vision-Instruct_TFV_TabFact.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-003158_naive_Llama-3.2-11B-Vision-Instruct_TFV_TabFact.jsonl"

    
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20250103-092052_cot_Llama-3.2-11B-Vision-Instruct_TQA_TABMWP.jsonl"
    # log_file = "/gly/guogb/lyl/Visual-Table/logs/20250103-103429_cot_Llama-3.2-11B-Vision-Instruct_TQA_WTQ.jsonl"
    # log_file = "/gly/guogb/lyl/Visual-Table/logs/20250103-111411_cot_Llama-3.2-11B-Vision-Instruct_TQA_HiTab.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241229-133506_naive_Llama-3.2-11B-Vision-Instruct_TQA_HiTab.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20250103-111424_cot_Llama-3.2-11B-Vision-Instruct_TQA_TAT-QA.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20241230-120804_correct_Llama-3.2-11B-Vision-Instruct_TQA_HiTab.jsonl"
    #log_file = "/gly/guogb/lyl/Visual-Table/logs/20250103-121758_cot_Llama-3.2-11B-Vision-Instruct_TFV_TabFact.jsonl"
    
    log_file = "/Users/doge/Desktop/Table_PEER/logs/20251117-190914_gpt-4o-mini_naive_all_all.jsonl"
    #log_file = "/Users/doge/Desktop/Table_PEER/logs/20251117-221402_gpt-4o-mini_peer_all_all.jsonl"

    # 读取日志文件到list
    samples = []
    with open(log_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    print(samples[0])

    samples = [s for s in samples if s["dataset_name"] in ["TabFact"]]#["TabFact", "TABMWP", "InfoTabs", "WTQ", "HiTab", "TAT-QA"]

    #samples = samples[0:6056]
    print("Total samples: ", len(samples))

    # if "TQA" in log_file:
    #     task_type = "TQA" 
    # if "TFV" in log_file:
    #     task_type = "TFV"
    task_type = "TFV" 
    dataset_name = "TabFact"
    
        
    if task_type == "TQA":
        for sample in samples:
            output = sample["output"]
            label = sample["label"]
        
            #output = "*Answer*: {\"answer\": [7]}"
            answer = transform_answer_to_standard_format(output)
            #answer = output
            
            # remove "%" and " " from the answer
            if "TQA_TAT" in dataset_name:
                answer = answer.lower()
                answer = answer.replace("%", "").replace(".0", "").replace(" million", "").replace(" billion", "").replace("$", "").replace(" rmb", "")
                if isinstance(label, list):
                    label = [l.lower().replace("%", "").replace("$", "")
                            for l in label
                            ]
            if "HiTab" in dataset_name:
                answer = answer.lower()
                answer = answer.replace("%", "").replace(".0", "").replace(" million", "")\
                            .replace(" billion", "").replace("$", "").replace(" rmb", "")\
                            .replace("$","").replace(".00","").replace(",","")
                if isinstance(label, list):
                    label = [l.lower().replace("%", "").replace("$", "").replace(".0", "")
                            for l in label
                            ]
            else:
                answer = answer.lower()
                if isinstance(label, list):
                    label = [l.lower() for l in label ]
                #answer = answer.replace("%", "")#.replace(" ", "")
                
            is_correct,_ = evaluate_tqa_tfv_one_item(answer, label, task_type)
            if is_correct:   
                ans1_correct_count += 1
            else:
                #print("output: ", output)
                print("transform_answer: ", answer)
                print("Label: ", label)
                pass

            print("is_correct: ", is_correct)

            # # at least one of the answers is correct
            # at_least_one_correct = is_correct or is_correct2
            # count
            #     flip_incorrect_count += 1
            total_count += 1
            print(f"cnt: {ans1_correct_count}/{total_count}")
            
    if task_type == "TFV":
        for sample in samples:
            output = sample["output"]
            label = sample["label"]
            
            answer = transform_answer_to_standard_format(output)   
            is_correct,_ = evaluate_tqa_tfv_one_item(answer, label, task_type)
            
            if is_correct:   
                ans1_correct_count += 1
                # if is_correct2:
                #     ans2_correct_count += 1
                # if at_least_one_correct:
                #     at_least_one_correct_count += 1
                # if is_correct and not is_correct2:
                #     flip_correct_count += 1
                # if not is_correct and is_correct2:
            else:
                print("Answer: ", answer)
                print("Label: ", label)

            # # at least one of the answers is correct
            # at_least_one_correct = is_correct or is_correct2
                    # count

            #     flip_incorrect_count += 1
            total_count += 1


        
    print("Answer correct count: ", ans1_correct_count)

    print("Total count: ", total_count)

    print("Answer 1 correct rate: ", ans1_correct_count / total_count)


    
 
    
