from typing import Self

from src.evaluator_arguments import BasicEvaluatorArguments


class EvaluatorArguments(BasicEvaluatorArguments):
    def additional_parameters(self: Self) -> Self:
        """Inheriting the basic evaluator argument to create the evaluator argument class for the TPP procedure.

        Args:
            self (Self): the evaluator argument
            parser (argparse.ArgumentParser): the item storing all arguments
            root_path (str): the root dir of this project

        Returns:
            Self: the evaluator argument
        """
        self.parser.set_defaults(
            procedure=__loader__.name.split('.', 1)[1].rsplit('.', 1)[0], \
            displayed_procedure_name="Temporal Point Process", \
            required_worker="Evaluator", \
            displayed_task_category="Model Evaluation"
        )
