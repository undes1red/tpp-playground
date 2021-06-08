from .dwg.model import TemporalModel
from .fullynn.model import FullyNNModel
from .ctlstm.model import CTLSTMwrapper
from .cnf.model import CNFWrapper

from ..utils import getLogger

logger = getLogger(__name__)

# One should register their model here. 
model_zoo = {
    'dwg': TemporalModel,
    'fullynn': FullyNNModel,
    'ctlstm': CTLSTMwrapper,
    'cnf': CNFWrapper
}

def get_model(name):
    try:
        model = model_zoo[name]
        logger.info(f"Model named {name} is retrieved.")
        return model
    except:
        logger.exception(f"Model named {name} is not found! Please register your model in src/model/__init__.py and try again.")