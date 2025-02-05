import json
import os
import time
import copy
from tqdm import tqdm
from utils.models import llm_init, llm_generate

plan_full_demo_simple = """

To answer a question related to a table, necessary information needs to be obtained from the table.
You need to give me the high-level planning of actions to acquire the critical information. 
Please provide a high-level plan of actions for identifying the critical information. Provide the chain of actions step-by-step with JSON format.
Below are examples of questions and corresponding chain of actions.

Question: Which country won the most bronze medals？
Chain of actions: 
{
    {
      "action": "identify_column",
      "description": "Locate the column labeled 'Bronze' in the table which contains the number of bronze medals won by each country."
    },
    {
      "action": "scan_values",
      "description": "Examine the values in the 'Bronze' column to find the highest number representing the most bronze medals won."
    },
    {
      "action": "locate_country",
      "description": "Identify the row that corresponds to the highest number of bronze medals and read the country name from the 'Nation' column of that row."
    }
    {
      "action": "<end>",
      "description": "Conclude the analysis and prepare to report which country has won the most bronze medals."
    }
}

Question: which nation earned the most gold medals in the 2004 summer paralympics?
Chain of actions: 
"""


plan_full_demo_simple = """
Question: name only the countries where all three medal counts are under 30.
Please tell me the information in the table so that I can answer the questions based on your information.
You don't need to answer the questions, just help me extract the information. 
The information you provide should be as detailed as possible, not just the cell value,. 
I need to answer the question based on your information because I cannot know its corresponding row and column.


Here are examples of extracted information to answer the question.

Image: 2004_Summer_Paralympics_medal_table.jpg
Question: name only the countries where all three medal counts are under 30.
Extracted information:
1. China won 63 gold medals, 46 silver medals, and 32 bronze medals, totaling 141 medals.
2. Great Britain secured 35 gold medals, 30 silver medals, and 29 bronze medals, totaling 94 medals.
3. Canada achieved 28 gold medals, 19 silver medals, and 25 bronze medals, totaling 72 medals.
4. The United States garnered 27 gold medals, 22 silver medals, and 39 bronze medals, totaling 88 medals.
5. Australia collected 26 gold medals, 38 silver medals, and 36 bronze medals, totaling 100 medals.
6. Ukraine won 24 gold medals, 12 silver medals, and 19 bronze medals, totaling 55 medals.
7. Spain secured 20 gold medals, 27 silver medals, and 24 bronze medals, totaling 71 medals.
8. Germany achieved 19 gold medals, 28 silver medals, and 31 bronze medals, totaling 78 medals.
9. France won 18 gold medals, 26 silver medals, and 30 bronze medals, totaling 74 medals.
10. Japan secured 17 gold medals, 16 silver medals, and 20 bronze medals, totaling 53 medals.
"""



naive_prompt = """
You are an expert at table understanding. This is a table QA task. The goal of this task is to answer the question given the table image.
Provide an answer in the JSON structure, using the format {\"answer\": [<a list of answer strings>]} such as {\"answer\": [\"1994\", \"25.34\"]}.
"""
# only for llama
# naive_prompt = """
# You are an expert at table understanding. This is a table QA task. The goal of this task is to answer the question given the table image. 
# Please directly provide the answer to the question without any other words., using the format {"answer": [<a list of answer strings>]} such as {"answer": ["1994"]} and {"answer": ["0.07"]}.
# """

cot_prompt = """
You are an expert at table understanding. This is a table QA task. The goal of this task is to answer the question given the table image.
Provide an answer in the JSON structure, using the format {\"answer\": [<a list of answer strings>]} such as {\"answer\": [\"1994\", \"25.34\"]}.
Let's think step by step.
"""
# The goal of this task is to extract relevant information from the table to answer the question accurately. Please analyze the table and provide the necessary details step-by-step.

"""
You are an expert at table understanding. This is a table question answering task.
Don't give answer, just extract the information related the question from the table image.
Please analyze the table and provide the related details step-by-step.
"""
perception_prompt = """
You are an expert at table understanding. This is a table question answering task. 
Your goal is to extract relevant information from the table to answer the question accurately. Please analyze the table and provide the related details step-by-step.
"""
# Your goal is to extract relevant information from the table to answer the question accurately. 

# You are an expert at table understanding.
# You need extract the detailed information from the table which is helpful for answer the question; 
# please convert the knowledge into sentences. No need to provide an answer to the question
# You need extract the detailed and useful information from the table related the question; 

summary_prompt = """
You are an expert at summarizing information.
You need to summary the information extracted from the table, and answer the question.
Provide an answer in the JSON structure, using the format {\"answer\": [<a list of answer strings>]} such as {\"answer\": [\"1994\", \"25.34\"]}.
"""

# feedback_prompt = """
# You are an Answer evaluator, You need to scan the whole table step-by-step and judge the correctness of the question-answer pairs based on the table in image.
# If the answer is correct, output True; otherwise, output False.
# Please only output the results without any other words in the JSON format of: {"feedback": True}. 
# The possible errors include:
# 1. The answer to the question does not match, even without looking at the table.
# 2. The question-answer pair is correct, but is not supported by the table.
# """ 

feedback_prompt = """
Based on the table image, the given question "{question}" have a preliminary answer "{answer}"
Please extract relevant information from the table to answer the question accurately. Please analyze the table and provide the related details step-by-step.
""" 
#Based on the information extracted from the table, provide feedback on the context of the question and answer.


tfv_naive_prompt = """
You are an expert at table understanding.
This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or
refuted by the given table. Provide an answer in the JSON structure, using the format {"answer": True} or {"answer": False}.
"""
tfv_cot_prompt = """
You are an expert at table understanding.
This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or
refuted by the given table. Provide an answer in the JSON structure, using the format {"answer": True} or {"answer": False}.
Let's think step by step.
"""
tfv_perception_prompt = """
You are an expert at table understanding. 
This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or
refuted by the given table. Don't give answer, just extract the information related the statement from the table image.
"""
# You are an expert at table understanding.
# You need extract the detailed information from the table which is helpful for answer the question; 
# please convert the knowledge into sentences. No need to provide an answer to the question
# You need extract the detailed and useful information from the table related the question; 

tfv_summary_prompt = """
You are an expert at summarizing information.
This is a fact verification task. Given extracted information from table, your goal is to distinguish whether the given statement is entailed or refuted. 
Provide an answer in the JSON structure, using the format {"answer": True} or {"answer": False}.
"""

_____tfv_feedback_prompt = """
You are a Visual-Language Answer Evaluator tasked with validating answers against visual evidence (e.g., a table image). 
Your task is as follows:

### Step 1: Re-evaluate the Given Answer  
- Carefully analyze the table in the provided image.  
- Verify if the given answer matches the information in the table.  

### Step 2: Explain Your Reasoning  
- If the answer is correct:  
   1. Confirm which specific part of the table supports the answer.  
   2. Highlight the exact data or evidence (e.g., row, column, value) from the table that supports your conclusion.  
   - Output: "correct. Evidence: [EXPLAIN DATA]."  

- If the answer is incorrect:  
   1. Identify the part of the table that disproves the answer.  
   2. Explain the discrepancy between the given answer and the actual evidence.  
   - Output: "incorrect. Reason: [EXPLAIN ERROR OR DISCREPANCY]."  

### Example:  
Image: A table showing "Year" and "Sales Data."  
Statement: "Sales in 2023 exceeded 1000 units."  
Answer: True

- If Table shows "Sales in 2023 = 980":  
   Output: "incorrect. Reason: Sales in 2023 are 980, which is less than 1000."  
- If Table shows "Sales in 2023 = 1200":  
   Output: "correct. Evidence: Sales in 2023 are 1200, which exceeds 1000."

Carefully re-check all relevant parts of the table before providing your response.
"""  

# answer = entailed / refuted 
tfv_feedback_prompt = """
You are an expert Visual-Language Answer Validator. 
The given statement "{statement}" is {answer} based on the information extracted from the table image.
Please provide the related context in the image of table, and don't repeat the statement.
"""



def get_table_info(sample, image_folder):
    table_info = sample
    table_info["image_path"] = os.path.join(image_folder, sample["image_id"]) + ".jpg"
    # table_info["original_query"] = sample['original_query']
    # table_info["input"] = sample["input"]
    return table_info



def generate_prompt_for_stage(
    sample,
    image_folder,
    llm,
    task_type,
    debug=False,
    llm_options=None,
    strategy="top",
    stage="perception",
    log = None
):
    table_info = get_table_info(sample, image_folder)
    #act_chain = table_info["act_chain"]

    # if debug:
    #     print("Table Info: ", table_info, flush=True)
        
    prompt = ""
    
    # ----------------- TQA -----------------
    if task_type == "TQA":
        if stage == "naive":
            prompt = naive_prompt
            prompt += "Question: " + table_info["original_query"] # MMTab query 
            response = llm_generate(llm, prompt, table_info["image_path"])
            if debug:
                print("Answer: ", response, flush=True)
        
        if stage == "cot":
            prompt = cot_prompt
            prompt += "Question: " + table_info["original_query"] # MMTab query 
            response = llm_generate(llm, prompt, table_info["image_path"])
            if debug:
                print("Answer: ", response, flush=True)
        
        if stage == "perception":
            prompt += perception_prompt + "\n\n"
            prompt += "Question: " + table_info["original_query"] + "\n"
            prompt += "Extracted information: "
            response = llm_generate(llm, prompt, table_info["image_path"])
            
            if debug:
                print("Perception: ", response, flush=True)
                
        if stage == "summary":
            #prompt += "Information extracted from table:" 
            prompt = summary_prompt 
            prompt += "Information extracted from table:" + log[-1]["response"] + "\n\n"
            #prompt += log[-1]["response"] + "\n\n"
            prompt += "Question: " + table_info["original_query"] + "\n"

            response = llm_generate(llm, prompt,)

            if debug:
                print("Reasoning: ", response, flush=True)
        
        if stage == "feedback":
            prompt += feedback_prompt.format(question=table_info["original_query"],answer=log[-1]["response"]) + "\n\n"
            # prompt += "Question: " + table_info["original_query"] + "\n"
            # prompt += "Answer: " + log[-1]["response"] + "\n"
        
            response = llm_generate(llm, prompt, table_info["image_path"])
            if debug:
                print("Feedback: ", response, flush=True)
                pass
            
        if stage == "aggregation_reasoning":
            prompt = "Information extracted from table:" 
            prompt += "### Evidence1: " + log[-3]["response"] + "\n"
            prompt += "### Evidence2: " + log[-1]["response"] + "\n\n"
            # prompt + log[-3]["response"] + "\n"
            # prompt += log[-1]["response"] + "\n\n"
            prompt += summary_prompt + "\n"
            prompt += "Question: " + table_info["original_query"] + "\n"

            response = llm_generate(llm, prompt,)

            if debug:
                print("Reasoning_agg: ", response, flush=True)
    
    
    # ----------------- TFV -----------------
    if task_type == "TFV":
        if stage == "naive":
            #prompt += table_info["input"] # MMTab query
            prompt = tfv_naive_prompt + "\n\n"
            prompt += "Statement: " + table_info["original_query"] + "\n"
            response = llm_generate(llm, prompt, table_info["image_path"])
            
        if stage == "cot":
            #prompt += table_info["input"] # MMTab query
            prompt = tfv_cot_prompt + "\n\n"
            prompt += "Statement: " + table_info["original_query"] + "\n"
            response = llm_generate(llm, prompt, table_info["image_path"])

        if stage == "perception":
            prompt += tfv_perception_prompt + "\n\n"
            prompt += "Statement: " + table_info["original_query"] + "\n"
    
            response = llm_generate(llm, prompt, table_info["image_path"])
            
            if debug:
                print("Statement: ", table_info["original_query"], flush=True)
                print("Perception: ", response, flush=True)
                
        if stage == "summary":
            prompt += tfv_summary_prompt + "\n"
            prompt += "Information extracted from table:" + log[-1]["response"] + "\n\n"
            prompt += "Statement: " + table_info["original_query"] + "\n"

            response = llm_generate(llm, prompt,)

            if debug:
                print("Summary: ", response, flush=True)
        
        if stage == "feedback":
            ans = "True" if "True" in log[-1]["response"] else "False"
            prompt += tfv_feedback_prompt.format(statement=table_info["original_query"], answer=ans) + "\n\n"
            #prompt += "Information extracted from table: " + log[-2]["response"] + "\n\n"
            # prompt += "Statement: " + table_info["original_query"] + "\n"
            # prompt += "Answer: " + log[-1]["response"] + "\n"

            response = llm_generate(llm, prompt, table_info["image_path"])
        
            if debug:
                    print("Feedback: ", response, flush=True)
                    pass
                    #print("Label:", sample['answer_list'], flush=True)
        
        if stage == "aggregation_reasoning":
            prompt = "Information extracted from table:"
            prompt += "### Evidence1: "
            prompt += log[-3]["response"] + "\n"
            prompt += "### Evidence2: "
            prompt += log[-1]["response"] + "\n\n"
            prompt += tfv_summary_prompt + "\n"
            prompt += "Statement: " + table_info["original_query"] + "\n"

            response = llm_generate(llm, prompt,)

            if debug:
                print("Reasoning_agg: ", response, flush=True)
    
    log =  {
        "stage": stage,
        "prompt": prompt,
        "response": response
    }
    return response,log



def dynamic_chain_exec_one_sample(
    sample,
    image_folder,
    llm,
    llm_options=None,
    task_type="TQA", # TQA, TFV
    strategy="top",
    debug=False,
    operation_parameter_dict=None,
    max_iteration=3
):

    # perception -> reasoning ->  feedback
    dynamic_chain_log = []
    last_answer = None

    current_sample = copy.deepcopy(sample)
    #current_sample = sample
    #if dynamic_chain_log == []:
    
    # perception
    response, log = generate_prompt_for_stage(
        current_sample,
        image_folder,
        llm=llm,
        llm_options=llm_options,
        strategy=strategy,
        debug=debug,
        task_type=task_type,
        stage="perception",
        log = dynamic_chain_log
        )
    dynamic_chain_log.append(log)
    
    # summary
    answer, log = generate_prompt_for_stage(
        current_sample,
        image_folder,
        llm=llm,
        llm_options=llm_options,
        task_type=task_type,
        strategy=strategy,
        debug=debug,
        stage="summary",
        log = dynamic_chain_log
        )
    dynamic_chain_log.append(log)
    last_answer = answer
    
    # Loop until the answer is stable
    for i in range(max_iteration):
        # feedback
        feedback, log = generate_prompt_for_stage(
            current_sample,
            image_folder,
            llm=llm,
            llm_options=llm_options,
            task_type=task_type,
            strategy=strategy,
            debug=debug,
            stage="feedback",
            log = dynamic_chain_log
            )
        dynamic_chain_log.append(log)
        
        # aggregation_reasoning
        answer, log = generate_prompt_for_stage(
            current_sample,
            image_folder,
            llm=llm,
            llm_options=llm_options,
            task_type=task_type,
            strategy=strategy,
            debug=debug,
            stage="aggregation_reasoning",
            log = dynamic_chain_log
            )
        dynamic_chain_log.append(log)
        
        if answer == last_answer:
            break
        
        last_answer = answer
        
    final_answer = answer

    return feedback, final_answer, dynamic_chain_log

