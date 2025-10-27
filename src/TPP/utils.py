import argparse

suffix_shortcut_dict = {
    'model_name': '',
    'lr': 'lr',
    'training_batch_size': 'bs',
    'used_batch_size': 'bs',
    'n_training_steps': 'nts',
    'dataloader_config': '',
    'used_dataloader_config': '',
    'model_config': '',
    'procedure_config': '',
    'used_procedure_config': '',
    'task_config': ''
}


def suffix(opt: argparse.Namespace, *args) -> str:
    """Help construct the output dir name using model hyperparameters.

    Args:
        opt (argparse.Namespace): the argument namespace

    Returns:
        str: the output dir name
    """
    output = []
    for item in args:
        hyperparameter = getattr(opt, item)
        translated_suffix = suffix_shortcut_dict[item] + str(hyperparameter)
        output.append(translated_suffix)

    return "_".join(output)


def easy_model_load(root_path: str, replace_id: str, dataset_name: str, dataset_name_for_model: str, device: str, evaluation: bool, **kwargs):
    kwargs_should_have = ['model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config']

    from types import SimpleNamespace

    from src.toolbox.evaluation import load_checkpoint
    from src.toolbox.misc import get_logger, read_yaml
    from src.TPP import get_model

    logger = get_logger(__name__)

    dataset_card = root_path / 'data' / 'TPP' / dataset_name / 'dataset_card.yml'
    info_dict = read_yaml(dataset_card)

    model_load_args = SimpleNamespace(procedure = 'TPP', model_name = kwargs['model_name'])
    model_class = get_model(model_load_args)

    abs_model_config = (root_path / 'config' / 'TPP' / kwargs['model_name'] / dataset_name_for_model / kwargs['model_config']) if kwargs['model_config'] else None
    model_param = read_yaml(abs_model_config)
    dataset_info_dict = SimpleNamespace(info_dict = info_dict, compile = False)
    model = model_class(device = device, opt = dataset_info_dict, **model_param)

    model_identifier = suffix(SimpleNamespace(**kwargs), *kwargs_should_have)
    checkpoint_folder_suffix = 'model_' + model_identifier
    checkpoint_folder = root_path / 'model' / 'TPP' / replace_id / dataset_name / checkpoint_folder_suffix

    return load_checkpoint(logger, checkpoint_folder / 'checkpoint.chkpt', model, device = device, evaluation = evaluation)

