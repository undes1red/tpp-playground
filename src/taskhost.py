import argparse
import importlib
import os
import random
import secrets
import time
from typing import Self

import matplotlib as mpl
import numpy as np
import torch

from src.toolbox.misc import get_logger, version_check

# The TaskHost executes tasks using pytorch.multiprocessing. Credits to the neural_stpp created by RTQ Chen from Facebook.
logger = get_logger("TaskHost")


class TaskHost:
    def __init__(self: Self, parser: argparse.ArgumentParser, root_path: str) -> Self:
        """Spawn a TaskHost

        Args:
            self (Self): The TaskHost
            parser (argparse.ArgumentParser): The parser which stores all arguments
            root_path (str): The root of the codebase.

        Returns:
            Self: The TaskHost
        """
        self.opt = parser.parse_args()
        self.opt.root_path = root_path

        # Parsing and postprocessing the input arguments.
        self.procedure = importlib.import_module("src." + self.opt.procedure)
        self.opt = getattr(self.procedure, f"{self.opt.required_worker}_postprocess")(self.opt, root_path)

        if self.opt.sleep > 0:
            logger.info(f"Now, I will take a nap. See you {self.opt.sleep}s later.")
            time.sleep(self.opt.sleep)
        self.pytorch_warning_dict = getattr(self.procedure, "pytorch_version_warnings")

    def pytorch_warning(self: Self, version: str) -> None:
        """Some pytorch releases have some known bugs that could affect this codebase.
        We can use this function to send warnings to the user, informing them that they are using a buggy pytorch release.

        Args:
            self (Self): The TaskHost
            version (str): The version of the installed pytorch
        """
        for key, warning_message in self.pytorch_warning_dict.items():
            if version_check(version, key):
                warning, action = warning_message
                if action == "continue":
                    logger.warning(warning)
                    logger.warning("Continue training.")
                else:
                    logger.exception(warning)

    def global_settings(self: Self) -> None:
        """Configuring pytorch and other packages.

        Args:
            self (Self): The TaskHost
        """
        if self.opt.no_seed:
            import time

            logger.warning(
                "For reproducibility, you need to assign a value to the random seed. If you want reproducible results, please ABORT this run ASAP and manually provide a random seed using '--seed'"
            )
            logger.warning(
                "The user doesn't provide a random seed. We will randomly select a number as the random seed."
            )
            random.seed(int(time.time()) % int.from_bytes(os.urandom(3), byteorder="big"))
            self.opt.seed = secrets.randbelow(1000000)
            logger.info(f"The model prefers {self.opt.seed} this time.")
        else:
            logger.info(f"We will use number {self.opt.seed} as the random seed.")

        # set up random seed for various packages
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        np.random.default_rng(self.opt.seed)
        torch.backends.cudnn.benchmark = False

        # Please read documentations and check if you have used any operations which don't have a deterministic implementation before
        # set it to True.
        torch.use_deterministic_algorithms(mode=True)

        # For gradient debug usage.
        # torch.autograd.set_detect_anomaly(True)

        # Allow tf32 in matmul to improve speed on recent hardware.
        torch.backends.cuda.matmul.allow_tf32 = True

        # Might benefit the Dataloader.
        torch.multiprocessing.set_sharing_strategy("file_system")

        # Do not break the CUDA graph from "tensor.item()".
        torch._dynamo.config.capture_scalar_outputs = True

        # Model explicit casting.
        if self.opt.dtype != "float32":
            logger.warning(
                f"Explicit casting enabled! We will train our model in {self.dict_torch_dtype[self.opt.dtype]}."
            )
            logger.warning(
                "Training MTPP models using lower precision may cause suboptimal results or failed training. If your model is sensitive to the precision. We recommend to stay on float32."
            )
            torch.set_default_dtype(self.dict_torch_dtype[self.opt.dtype])

    dict_torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    def cuda(self: Self) -> None:
        """Check cuda availability. We force using CPU if cuda is unavailable even the user wants to use cuda.

        Args:
            self (Self): The TaskHost
        """
        if self.opt.cuda and not torch.cuda.is_available():
            logger.warning(
                "You expect cuda acceleration but cuda is unavailable in this machine. Please check your cuda configuration and make sure that you have installed pytorch with cuda support."
            )
            logger.warning("We use cpu now.")
            self.opt.cuda = False
        elif self.opt.cuda and torch.cuda.is_available():
            logger.warning("We use cuda to speed up model training!")
            logger.info(f"We use PyTorch compiled against CUDA {torch.version.cuda}.")
            logger.info(f"Found {torch.cuda.device_count()} CUDA device(s).")
            logger.info(f"We use the CUDA device with id {self.opt.cuda_device}.")
            props = torch.cuda.get_device_properties(self.opt.cuda_device)
            logger.info(f"CUDA Device name: {props.name} \t Memory: {props.total_memory / (1024**3):.2f}GiB.")
            if props.major > 6:
                logger.info(
                    f"Device supports CUDA {props.major}.{props.minor} higher than 6.0. torch.compile() is possible."
                )
            else:
                logger.info(
                    f"Device supports CUDA {props.major}.{props.minor} not higher than 6.0. torch.compile() is impossible."
                )
                self.opt.compile = False
        else:
            logger.warning("We use cpu now.")

        # Limit the number of executing thread when running code on CPU.
        if not self.opt.cuda:
            logger.info("Setting available CPU threads.")
            logger.info(f"Available CPU threads: {torch.get_num_threads()}.")

            torch.set_num_threads(torch.get_num_threads())

    def start(self: Self) -> None:
        """start.py calls this function to start the task.

        Args:
            self (Self): The TaskHost.
        """
        logger.debug(f"Root path: {self.opt.root_path}.")
        logger.warning(
            f"Main procedure name: {self.opt.displayed_procedure_name}. Sub-procedure name: {self.opt.displayed_task_category}."
        )

        # Show the config file of matplotlib.
        logger.info(f"The current active matplotlib config file is {mpl.matplotlib_fname()}.")

        # Show the version and other information of the installed PyTorch.
        # Then do some configurations.
        logger.info(f"PyTorch Version: {torch.__version__}.")
        self.pytorch_warning(torch.__version__)
        self.global_settings()

        # Report device properties.
        self.cuda()
        # For now, this codebase only supports single GPU training.
        self.opt.device = torch.device(f"cuda:{self.opt.cuda_device}" if self.opt.cuda else "cpu")

        # load the related packages and start the task.
        root_package = importlib.import_module("src")
        self.worker = getattr(root_package, self.opt.required_worker)(opt=self.opt, procedure=self.procedure)
        self.worker.work()
