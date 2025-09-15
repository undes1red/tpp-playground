import yaml
import io, os

# Read and convert a YAML file into a dict object.
def read_yaml(yaml_path):
    a = {}
    if yaml_path is not None:
        with open(yaml_path, 'r') as f:
            try:
                a = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    
    if a is None:
        a = {}

    return a


def write_yaml(data, yaml_path, yaml_file):
    with io.open(os.path.join(yaml_path, yaml_file), 'w', encoding = 'utf8') as outfile:
        yaml.safe_dump(data, outfile, default_flow_style = False, allow_unicode = True)