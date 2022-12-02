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


modelpath = {
    # [file name, Model wrapper name]
    # Actively maintained models
    'dwg': ['dwg.model', 'TemporalModel'],
    'fullynn': ['fullynn.model', 'FullyNNModel'],
    'multi_fullynn': ['multi_fullynn.model', 'MultiFullyNNModel'],
    'ifl': ['ifl.model', 'IFL'],
    'rmtpp': ['rmtpp.model', 'RMTPP'],
    'thp': ['thp.model', 'THP'],
    'sahp': ['sahp.model', 'SAHP'],
    
    # Temporarily abandoned models
    'ctlstm': ['ctlstm.model', 'CTLSTMwrapper'],
    'cnf': ['cnf.model', 'CNFWrapper'],
    'fullynn_v2': ['fullynn_v2.model', 'FullyNN2Model'],
    'attn_cm': ['attn_cm.model', 'AttnCMWrapper'],
    'transnn': ['transnn.model', 'TransNNModel'],
    'multi_fullynn_arg': ['multi_fullynn_arg.model', 'MultiFullyNNModel'],
}

def model_zoo(name):
    path, model_name = modelpath[name]
    module = importlib.import_module('.' + path, package = 'src.TPP.model')
    return getattr(module, model_name)