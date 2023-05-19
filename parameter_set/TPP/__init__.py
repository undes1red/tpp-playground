# FENN
from parameter_set.TPP.fenn_parameter_set import training_hyperparameter as feps_t
from parameter_set.TPP.fenn_parameter_set import plot_hyperparameter as feps_p

# TFENN
from parameter_set.TPP.tfenn_parameter_set import training_hyperparameter as tfeps_t
from parameter_set.TPP.tfenn_parameter_set import plot_hyperparameter as tfeps_p

# FullyNN 
from parameter_set.TPP.fullynn_parameter_set import training_hyperparameter as fps_t
from parameter_set.TPP.fullynn_parameter_set import plot_hyperparameter as fps_p

# TFullyNN
from parameter_set.TPP.tfullynn_parameter_set import training_hyperparameter as tfps_t
from parameter_set.TPP.tfullynn_parameter_set import plot_hyperparameter as tfps_p

# Transformer Hawkes Process(THP)
from parameter_set.TPP.thp_parameter_set import training_hyperparameter as thp_t
from parameter_set.TPP.thp_parameter_set import plot_hyperparameter as thp_p

# Recurrent Marked Hawkes Process(RMTPP)
from parameter_set.TPP.rmtpp_parameter_set import training_hyperparameter as rmtpp_t
from parameter_set.TPP.rmtpp_parameter_set import plot_hyperparameter as rmtpp_p

# LogNormMix
from parameter_set.TPP.lognormmix_parameter_set import training_hyperparameter as ifl_t
from parameter_set.TPP.lognormmix_parameter_set import plot_hyperparameter as ifl_p

# Self-attentive Hawkes Process(SAHP)
from parameter_set.TPP.sahp_parameter_set import training_hyperparameter as sahp_t
from parameter_set.TPP.sahp_parameter_set import plot_hyperparameter as sahp_p

# Recurrent Hawkes Process(RHP), a.k.a. CTLSTM
from parameter_set.TPP.rhp_parameter_set import training_hyperparameter as rhp_t
from parameter_set.TPP.rhp_parameter_set import plot_hyperparameter as rhp_p

# Intensity-free Integral-based model-numerical(IFIB-C)
from parameter_set.TPP.ifib_c_parameter_set import training_hyperparameter as ifcps_t
from parameter_set.TPP.ifib_c_parameter_set import plot_hyperparameter as ifcps_p

# Intensity-free Integral-based model-numerical(IFIB-N)
from parameter_set.TPP.ifib_n_parameter_set import training_hyperparameter as ifnps_t
from parameter_set.TPP.ifib_n_parameter_set import plot_hyperparameter as ifnps_p

# Transformer-powered Intensity-free Integral-based model-numerical(TIFIB-C)
from parameter_set.TPP.tifib_parameter_set import training_hyperparameter as tifps_t
from parameter_set.TPP.tifib_parameter_set import plot_hyperparameter as tifps_p


plot_parameter_set = {
    'fenn': {'train': feps_t, 'plot': feps_p},
    'tfenn': {'train': tfeps_t, 'plot': tfeps_p},
    'fullynn': {'train': fps_t, 'plot': fps_p},
    'tfullynn': {'train': tfps_t, 'plot': tfps_p},
    'thp': {'train': thp_t, 'plot': thp_p},
    'rmtpp': {'train': rmtpp_t, 'plot': rmtpp_p},
    'lognormmix': {'train': ifl_t, 'plot': ifl_p},
    'sahp': {'train': sahp_t, 'plot': sahp_p},
    'rhp': {'train': rhp_t, 'plot': rhp_p},
    'ifib_c': {'train': ifcps_t, 'plot': ifcps_p},
    'tifib': {'train': tifps_t, 'plot': tifps_p},
    'ifib_n': {'train': ifnps_t, 'plot': ifnps_p},
}

def parameter_retriver(opt):
    model_parameter_set = plot_parameter_set[opt.model]
    required_parameter_set = model_parameter_set[opt.script_type][opt.dataset]
    
    return required_parameter_set
    