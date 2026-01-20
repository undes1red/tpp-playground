import argparse

suffix_shortcut_dict = {
    "model_name": "",
    "lr": "lr",
    "training_batch_size": "bs",
    "used_batch_size": "bs",
    "n_training_steps": "nts",
    "dataloader_config": "",
    "used_dataloader_config": "",
    "model_config": "",
    "procedure_config": "",
    "used_procedure_config": "",
    "task_config": "",
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


def easy_model_load():
    pass
