import importlib
from ..utils import getLogger

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
            logger.exception(f"Model named {name} is not found! Please register your model in src/model/__init__.py and try again.")


dataloader_modulepath = {
    'dwg': ['dwg.model', 'TemporalModel'],
    'fullynn': ['fullynn.model', 'FullyNNModel'],
    'ctlstm': ['ctlstm.model', 'CTLSTMwrapper'],
    'cnf': ['cnf.model', 'CNFWrapper'],
    'ifl': ['ifl.model', 'IFL'],
    'rmtpp': ['rmtpp.model', 'RMTPP']

    # 2021-10-14 update: all legacy models are deprecated and out of maintenance.
    # Please take your own risk when you readd and use them. 
}

def model_zoo(name):
    path, model_name = dataloader_modulepath[name]
    module = importlib.import_module('.' + path, package = 'src.TPP.model')
    return getattr(module, model_name)