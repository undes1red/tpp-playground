import importlib
from src.taskhost import getLogger

# We use this parameter to control model's memory usage while running event-time prediction tasks.
# Used in FENN, FullyNN, SAHP, RHP, and THP
memory_ceiling = 3e7

# The lower and upper boundary of the inversed transform sampling.
# The final trick to make IFIB generate sane samples by avoiding the long tail.
its_lower_bound = 0.0
its_upper_bound = 0.9

logger = getLogger(__name__)

# One should register their models here.
# The key of each model is foremost.

def get_model(name, rank = 0):
    try:
        model = model_zoo(name)
        if rank == 0:
            logger.info(f"Model named {name} is retrieved.")
        return model
    except Exception as e:
        if rank == 0:
            logger.exception(f'{e}.')
            logger.exception(f"Model named {name} is not found! Please register your model in src/model/{name}/__init__.py and try again.")


def model_zoo(name):
    module = importlib.import_module('.' + name, package = 'src.TPP.model')
    return module.get_model()