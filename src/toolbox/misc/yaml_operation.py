import yaml
import io
import os
from typing import Dict


# Read and convert a YAML file into a dict object.
def read_yaml(yaml_path: str) -> Dict:
    """Read a yaml file

    Args:
        yaml_path (str): The path to the yaml file.

    Returns:
        Dict: The data
    """
    a = {}
    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            try:
                a = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    
    if a is None:
        a = {}

    return a


def write_yaml(data: Dict, yaml_path: str, yaml_file: str) -> None:
    """Write a dict into a yaml file.

    Args:
        data (Dict): The data
        yaml_path (str): Folder where we place the yaml file.
        yaml_file (str): The name of this yaml file.
    """
    with io.open(os.path.join(yaml_path, yaml_file), "w", encoding="utf8") as outfile:
        yaml.safe_dump(data, outfile, default_flow_style=False, allow_unicode=True)
