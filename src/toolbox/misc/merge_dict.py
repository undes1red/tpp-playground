from functools import reduce
from typing import List, Dict, Any

def merge_list_of_dicts(input_dict: List[Dict[Any, Any]]):
    return reduce(dict.__or__, input_dict)