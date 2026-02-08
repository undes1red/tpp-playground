import argparse
from typing import Self

from src.toolbox.misc import convert_module_to_path, suffix


class BasicEvaluatorArguments:
    def __init__(self: Self, parser: argparse.ArgumentParser) -> Self:
        """Create a basic argument parser for the evaluator.

        Args:
            self (Self): The evaluator argument.
            parser (argparse.ArgumentParser): The parser.

        Returns:
            Self: The evaluator argument.
        """
        self.parser = parser

        # The Ultimate
        self.parser.add_argument(
            "--no_seed",
            action="store_true",
            help="This argument tells our code to randomly select a seed. You can use this option to explore your model's robustness.",
        )
        self.parser.add_argument("--seed", type=int, default=32, help="Set global random seed.")
        self.parser.add_argument(
            "--cuda",
            action="store_true",
            help="Set it to true if you want to use GPU to accelerate model training.",
        )
        self.parser.add_argument(
            "--cuda_device",
            type=int,
            default=0,
            help="Select which CUDA device you want to use. Default number is 0. This argument does nothing if --cuda is not set.",
        )
        self.parser.add_argument(
            "--replace",
            action="store_true",
            help="True: Replace existing everything, such as logs, model checkpoints, and results with the new one.\n False: Do not replace.",
        )
        self.parser.add_argument(
            "--model_index",
            nargs="+",
            default=None,
            help="Pick the model by its index.",
        )
        self.parser.add_argument(
            "--compile",
            action="store_true",
            help="True: Use torch.compile() to compile models or functions to speed up training and evaluation.\n False: disable torch.compile().",
        )
        self.parser.add_argument(
            "--dtype",
            type=str,
            default="float32",
            help="Train the MTPP model in different precision. Useful when training MTPP on devices fast on lower precision like bfloat16 or float16 but very slow on float32, for example A100.",
        )

        # The number of Dataloader worker
        self.parser.add_argument(
            "--n_worker",
            default=8,
            type=int,
            help="The number of dataloader workers. For most datasets, multithreading might speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet 'received 0 items of ancdata' exception.",
        )
        self.parser.add_argument(
            "--sleep",
            default=0,
            type=int,
            help="This task is delayed and will start in the amount of time you have set.",
        )

        # Input data
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="Name of the used dataset. All datasets should be placed in {root}/data/input.",
        )
        self.parser.add_argument(
            "--dataset_type",
            type=str,
            default="parquet",
            help="File type of the selected dataset.",
        )
        self.parser.add_argument(
            "--dataloader_name",
            default=None,
            help="Name of the used dataloader. All dataloaders are stored in {root}/src/TPP/dataloader.",
        )
        self.parser.add_argument(
            "--dataloader_config",
            type=str,
            default=None,
            help="Relative path to the custom dataloader config file. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{dataloader_config}.",
        )
        self.parser.add_argument(
            "--used_dataloader_config",
            type=str,
            default=None,
            help="The name of dataloader config file used during training. We only need the filename, not the relative path.",
        )
        self.parser.add_argument(
            "--combine_used_and_current_dataloader_config",
            action="store_true",
            help="Combine the settings defined in used_dataloader_config and dataloader_config when set. Settings in dataloader_config will be overwritten by used_dataloader_config if exists in used_dataloader_config.",
        )

        self.parser.add_argument(
            "--training_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--evaluate_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--test_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )

        # Evaluation related hyperparameters
        self.parser.add_argument(
            "--n_training_steps",
            type=int,
            default=10000,
            help="How many steps did we use to train this model?",
        )
        self.parser.add_argument(
            "--agg_update_step",
            type=int,
            default=1,
            help="The number of minibatches between two adjacent optimizer steps.\
                                                                                 The number of practical training steps is agg_update_step * n_training_steps.",
        )

        # Model save and log management
        self.parser.add_argument(
            "--save_mode",
            type=str,
            choices=["all", "best"],
            default="best",
            help="Store all model checkpoints or only store the best one.",
        )

        # Training procedure related hyperparameters
        self.parser.add_argument(
            "-ub",
            "--used_batch_size",
            type=int,
            default=2048,
            help="Batch size used for training the model.",
        )
        self.parser.add_argument(
            "-tb",
            "--training_batch_size",
            type=int,
            default=1,
            help="Batch size used for training set.",
        )
        self.parser.add_argument(
            "-eb",
            "--evaluation_batch_size",
            type=int,
            default=1,
            help="Batch size used for evaluation set.",
        )
        self.parser.add_argument(
            "--used_procedure_config",
            type=str,
            default=None,
            help="Relative path to the custom setting file, in which settings are applied to all tasks under the procedure. The absolute file path is {root}/config/${main_procedure_name}/${procedure_config}",
        )

        # Model-related hyperparameters
        self.parser.add_argument("--model_name", default=None, help="The model name.")
        self.parser.add_argument(
            "--model_config",
            type=str,
            default=None,
            help="Relative path to the custom model config file used for training. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{model_config}.",
        )

        # Optimizer-related hyperparameters
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.1,
            help="The learning rate used when training the model.",
        )

        # Which task you'd like to run and where is the task config file?
        self.parser.add_argument(
            "--procedure_config",
            type=str,
            default=None,
            help="Relative path to the custom setting file, in which settings are applied to all tasks under the procedure. The absolute file path is {root}/config/${main_procedure_name}/${procedure_config}",
        )
        self.parser.add_argument(
            "--task_name",
            type=str,
            help="Define which evaluation task you'd like to start.",
        )
        self.parser.add_argument(
            "--task_config",
            type=str,
            help="Relative path to the custom subtask config file used for training. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{task_config}.",
        )

        self.additional_parameters()
        self.parser.set_defaults(called_subparser=self)

    def additional_parameters(self) -> Self:
        """Please inherit this function for adding procedure-specific arguments.
           This function will be called after postprocess().

        Args:
            opt (argparse.Namespace): the original argument dict
            root_path (str): the root dir of this project

        Returns:
            argparse.Namespace: the processed argument dict
        """
        pass

    def additional_parameters_postprocess(self, opt: argparse.Namespace) -> argparse.Namespace:
        """Please inherit this function to process procedure-specific arguments.
           This function will be called after postprocess().

        Args:
            opt (argparse.Namespace): the original argument dict
            root_path (str): the root dir of this project

        Returns:
            argparse.Namespace: the processed argument dict
        """
        return opt

    def postprocess(self, opt: argparse.Namespace) -> argparse.Namespace:
        """postprocess the trainer arguments.

        Args:
            opt (argparse.Namespace): the original argument dict
            root_path (str): the root dir of this project

        Returns:
            argparse.Namespace: the processed argument dict
        """

        # Gradient aggergation check
        if opt.agg_update_step > 1:
            opt.n_training_steps *= opt.agg_update_step

        # Convert the name from "a.b" to "a/b" for path assembly.
        opt.procedure_path = convert_module_to_path(opt.procedure)
        opt.model_name_path = convert_module_to_path(opt.model_name)
        opt.dataset_name_path = convert_module_to_path(opt.dataset_name)
        opt.dataloader_name_path = convert_module_to_path(opt.dataloader_name)

        opt.data_path = opt.root_path / "data" / opt.procedure_path / opt.dataset_name_path

        opt.abs_dataloader_config = (
            opt.root_path / "config" / opt.procedure_path / "dataloader" / opt.dataloader_name_path / opt.dataloader_config
            if opt.dataloader_config
            else None
        )
        opt.abs_procedure_config = (
            opt.root_path / "config" / opt.procedure_path / opt.procedure_config if opt.procedure_config else None
        )

        opt.procedure_config = opt.abs_procedure_config.name if opt.procedure_config else None
        opt.dataloader_config = opt.abs_dataloader_config.name if opt.dataloader_config else None

        opt.abs_model_config = (
            opt.root_path / "config" / opt.procedure_path / "model" / opt.model_name_path / opt.model_config if opt.model_config else None
        )
        opt.model_config = opt.abs_model_config.name if opt.model_config else None

        opt.abs_task_config = (
            opt.root_path / "config" / opt.procedure_path / "model" / opt.model_name_path / opt.task_config if opt.task_config else None
        )
        opt.task_config = opt.abs_task_config.name if opt.abs_task_config else None

        opt.abs_used_dataloader_config = (
            opt.root_path / "config" / opt.procedure_path / "dataloader" / opt.dataloader_name_path / opt.used_dataloader_config
            if opt.used_dataloader_config
            else None
        )
        opt.used_dataloader_config = opt.abs_used_dataloader_config.name if opt.used_dataloader_config else None

        opt.model_dir = opt.root_path / "model" / opt.procedure_path
        opt.result_dir = opt.root_path / "results" / opt.procedure_path
        opt.model_identifier = suffix(
            opt,
            "model_name",
            "dataloader_name",
            "lr",
            "used_batch_size",
            "n_training_steps",
            "used_procedure_config",
            "used_dataloader_config",
            "model_config",
        )
        opt.task_identifier = suffix(opt, "procedure_config", "dataloader_config", "task_config")

        return self.additional_parameters_postprocess(opt)
