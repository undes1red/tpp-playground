import importlib
from src.taskhost import get_logger

logger = get_logger(__name__)

# One should register their models here.
# The key of each model is foremost.

def get_model(opt):
    try:
        model = model_zoo(opt)
    except Exception as e:
        logger.exception(f"Model named {opt.model_name} is not found! Please register your model in src/model/{opt.model_name}/__init__.py and try again.")

    logger.info(f"Model named {opt.model_name} found and imported.")
    return model


def model_zoo(opt):
    module = importlib.import_module('.' + opt.model_name, package = f'src.{opt.procedure}.model')
    return module.get_model()