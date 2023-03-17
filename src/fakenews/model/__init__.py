import importlib
from src.TPP.utils import getLogger

logger = getLogger(__name__)

# One should register their models here.
# The key of each model is foremost.

def get_model(name, rank = 0):
    try:
        model = model_zoo(name)
        if rank == 0:
            logger.info(f"Model named {name} is retrieved.")
        return model
    except:
        if rank == 0:
            logger.exception(f"Model named {name} is not found! Please register your model in src/model/{name}/__init__.py and try again.")


def model_zoo(name):
    module = importlib.import_module('.' + name, package = 'src.fakenews.model')
    return module.get_model()