from .dwg.model import TemporalModel
from .dwg_new.model import TemporalModel as TemporalModel_new
from .fullynn.model import FullyNNModel
from .ctlstm.model import CTLSTMwrapper
from .cnf.model import CNFWrapper
from .ifl.model import ifl
from .rmtpp.model import RMTPP

from .utils import BasicModule

from ..utils import getLogger

logger = getLogger(__name__)

# One should register their models here.
# The key of each model is foremost.
model_zoo = {
    'dwg': TemporalModel,
    'dwg_new': TemporalModel_new,
    'fullynn': FullyNNModel,
    'ctlstm': CTLSTMwrapper,
    'cnf': CNFWrapper,
    'ifl': ifl,
    'rmtpp': RMTPP
}

def get_model(name, rank = 0):
    try:
        model = model_zoo[name]
        if rank == 0:
            logger.info(f"Model named {name} is retrieved.")
        return model
    except:
        if rank == 0:
            logger.exception(f"Model named {name} is not found! Please register your model in src/model/__init__.py and try again.")