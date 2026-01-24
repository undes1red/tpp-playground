import argparse

suffix_shortcut_dict = {
    "lr": "lr",
    "training_batch_size": "bs",
    "used_batch_size": "bs",
    "n_training_steps": "nts",
}


def suffix(opt: argparse.Namespace, *args, translate_dict = suffix_shortcut_dict) -> str:
    """Help construct the output dir name using model hyperparameters.

    Args:
        opt (argparse.Namespace): the argument namespace

    Returns:
        str: the output dir name
    """
    output = []
    for item in args:
        hyperparameter = getattr(opt, item)
        translated_suffix = translate_dict.get(item, "") + str(hyperparameter)
        output.append(translated_suffix)

    return "_".join(output)
