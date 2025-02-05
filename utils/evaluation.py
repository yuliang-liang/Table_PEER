
import json
import random
import re
import tqdm
from collections import defaultdict
#from sacrebleu.metrics import BLEU

# The codes is from MMTab: Multimodal Table Understanding 24'ACL paper.

def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub('[\"\']+',"\"",answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = []
        return predicted_answer
    else:
        return []



all = {'supported', 'entailed', 'opposes', 'contradicts', 'refutes', 'substantiates', 'entails', 'contradicted', 'refuted',
       'affirmed', 'affirms', 'challenged', 'substantiated', 'supports', 'upheld', 'disputes', 'opposed', 'challenges', 
       'conflicts with', 'disproved', 'confirmed', 'disproves', 'confirms', 'upholds', 'disputed'}

entailed_list = {
    'supported', 'entailed', 'substantiates', 'entails', 'affirmed', 'affirms', 
    'substantiated', 'supports', 'upheld', 'confirmed', 'confirms', 'upholds'
}
refuted_list = {
    'opposes', 'contradicts', 'refutes', 'contradicted', 'refuted', 'challenged', 
    'disputes', 'opposed', 'challenges', 'conflicts with', 'disproved', 'disproves', 'disputed'
}

def evaluate_tqa_tfv_one_item(pred, label, task_type, feedback=None):
    """
    Evaluation for table question answering (TQA) and table fact verification (TFV) benchmark.
    Metric: accuracy.
    Note that some baseline models can not strictly follow instructions to output the final answer in the required JSON format.
    For instance, Qwen-VL may only output a short answer due to the potential overfitting of training data.
    In such cases, the evaluation script needs to be changed according to the characteristic of certain model output.
    """
    
    assert task_type in ['TQA', 'TFV']
    
    is_correct = False
    is_failed = False
    if task_type == 'TQA':
        try:
            # parse the predicted answer list
            predicted_answer_list = extract_tqa_answer_list(pred)
            gold_answer_list = label
            # Sometimes the order of multiple answer text is not necessarily same as the gold answer,
            # so we convert the answer list to a set for comparison
            if set(gold_answer_list) == set(predicted_answer_list):
                is_correct = True
                
            # parse feedback
            if 'feedback' is not None:
                pass
            
        except Exception:
            is_failed = True
    if task_type == 'TFV':
        try:
            # parse the predicted answer list
            predicted_answer = extract_tqa_answer_list(pred)
            
            # parse the gold answer list
            gold_answer = True if label[0] in entailed_list else False
            if predicted_answer == gold_answer:
                is_correct = True
            
        except Exception:
            is_failed = True
        
    return is_correct, is_failed
