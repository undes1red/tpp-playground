from ..utils import getLogger

logger = getLogger(__name__)

# One should register their models here.
# The key of each model is foremost.

def get_model(name, rank = 0):
    try:
        model = model_zoo[name]()
        if rank == 0:
            logger.info(f"Model named {name} is retrieved.")
        return model
    except:
        if rank == 0:
            logger.exception(f"Model named {name} is not found! Please register your model in src/model/__init__.py and try again.")

'''
Without this, we can not continue our work when we conduct model training procedures.
'''
def dwg():
    '''
    New dwg model loader.
    '''
    from .dwg.model import TemporalModel
    return TemporalModel

def fullynn():
    '''
    FullyNN model loader
    '''
    from .fullynn.model import FullyNNModel
    return FullyNNModel

def ctlstm():
    '''
    CTLSTM model loader
    '''
    from .ctlstm.model import CTLSTMwrapper
    return CTLSTMwrapper

def cnf():
    '''
    Continuous normalized flow based model loader.
    '''
    from .cnf.model import CNFWrapper
    return CNFWrapper

def ifl():
    '''
    Intensity-free learning model loader.
    '''
    from .ifl.model import IFL
    return IFL

def rmtpp():
    '''
    RMTPP loader.
    '''
    from .rmtpp.model import RMTPP
    return RMTPP


model_zoo = {
    'dwg': dwg,
    'fullynn': fullynn,
    'ctlstm': ctlstm,
    'cnf': cnf,
    'ifl': ifl,
    'rmtpp': rmtpp,

    # 2021-10-14 update: all legacy models are deprecated and out of maintenance.
    # Please take your own risk when you readd and use them. 
}