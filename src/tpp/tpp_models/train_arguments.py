from typing import Self

from src.trainer_arguments import BasicTrainerArguments


class TrainerArguments(BasicTrainerArguments):
    def additional_parameters(self: Self) -> Self:
        """Inheriting the basic trainer argument to create the trainer argument class for the TPP procedure.

        Args:
            self (Self): the trainer argument
            parser (argparse.ArgumentParser): the item storing all arguments
            root_path (str): the root dir of this project

        Returns:
            Self: the trainer argument
        """

        # self identification.
        self.parser.set_defaults(
            procedure=__loader__.name.split('.', 1)[1].rsplit('.', 1)[0], \
            displayed_procedure_name="Temporal Point Process", \
            required_worker="Trainer", \
            displayed_task_category="Model Training"
        )
