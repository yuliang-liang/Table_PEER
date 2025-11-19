import random
from typing import Union, Optional

# MMTab 数据集过滤
def filter_samples(
    samples: list,
    task_type: Union[str, None] = None,
    dataset_name: Union[str, None] = None,
    sample_ratio: float = 1.0,
    random_seed: int = 42
) -> list:
    """
    根据task_type和dataset_name筛选样本，并支持随机抽样
    
    Args:
        samples (list): 原始样本列表
        task_type (str or None): 任务类型 (如 "TQA", "TFV")，当为 None 或 "all" 时不进行该维度筛选
        dataset_name (str or None): 数据集名称 (如 "TabFact", "TAT-QA")，当为 None 或 "all" 时不进行该维度筛选
        sample_ratio (float): 采样比例，范围[0,1]，默认1.0表示使用全部样本
        random_seed (int): 随机种子，用于复现采样结果
        
    Returns:
        list: 筛选并采样后的样本列表
    """
    def should_filter(value: Optional[str]) -> bool:
        return value is not None and value != "all"
    
    # 根据task_type和dataset_name筛选
    filtered_samples = []
    for s in samples:
        match = True
        if should_filter(task_type):
            match = match and s["task_type"] == task_type
        if should_filter(dataset_name):
            match = match and s["dataset_name"] == dataset_name
        if match:
            filtered_samples.append(s)
            
    print(f"After filtering - task_type: {task_type}, dataset_name: {dataset_name}")
    print(f"Filtered samples: {len(filtered_samples)}")
    
    if sample_ratio < 1.0:
        # 设置随机种子以确保可复现性
        random.seed(random_seed)
        # 计算需要的样本数量
        sample_size = int(len(filtered_samples) * sample_ratio)
        if sample_size == 0 and sample_ratio > 0:
            # 确保至少有一个样本
            sample_size = 1
        # 随机采样
        filtered_samples = random.sample(filtered_samples, sample_size)
        print(f"After sampling {sample_ratio*100}% - Number of samples: {len(filtered_samples)}")
    
    return filtered_samples