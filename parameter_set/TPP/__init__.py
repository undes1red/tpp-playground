# fenn parameter set
from parameter_set.TPP.fenn_parameter_set import training_hyperparameter as feps_t
from parameter_set.TPP.fenn_parameter_set import plot_hyperparameter as feps_p

# tfenn parameter set
from parameter_set.TPP.tfenn_parameter_set import training_hyperparameter as tfeps_t
from parameter_set.TPP.tfenn_parameter_set import plot_hyperparameter as tfeps_p

# fullynn parameter set
from parameter_set.TPP.fullynn_parameter_set import training_hyperparameter as fps_t
from parameter_set.TPP.fullynn_parameter_set import plot_hyperparameter as fps_p

# tfullynn parameter set
from parameter_set.TPP.tfullynn_parameter_set import training_hyperparameter as tfps_t
from parameter_set.TPP.tfullynn_parameter_set import plot_hyperparameter as tfps_p

# Transformer Hawkes Process(THP) parameter set
from parameter_set.TPP.thp_parameter_set import training_hyperparameter as thp_t
from parameter_set.TPP.thp_parameter_set import plot_hyperparameter as thp_p

# Recurrent Marked Hawkes Process(RMTPP) parameter set
from parameter_set.TPP.rmtpp_parameter_set import training_hyperparameter as rmtpp_t
from parameter_set.TPP.rmtpp_parameter_set import plot_hyperparameter as rmtpp_p

# LogNormMix parameter set
from parameter_set.TPP.lognormmix_parameter_set import training_hyperparameter as ifl_t
from parameter_set.TPP.lognormmix_parameter_set import plot_hyperparameter as ifl_p

# Self-attentive Hawkes Process(SAHP) parameter set
from parameter_set.TPP.sahp_parameter_set import training_hyperparameter as sahp_t
from parameter_set.TPP.sahp_parameter_set import plot_hyperparameter as sahp_p

# Recurrent Hawkes Process(RHP) parameter set
from parameter_set.TPP.rhp_parameter_set import training_hyperparameter as rhp_t
from parameter_set.TPP.rhp_parameter_set import plot_hyperparameter as rhp_p

# fullynn_probability(IBIF) parameter set
from parameter_set.TPP.ifib_parameter_set import training_hyperparameter as ifps_t
from parameter_set.TPP.ifib_parameter_set import plot_hyperparameter as ifps_p

# continuous fullynn_probability(CIBIF) parameter set
from parameter_set.TPP.cifib_parameter_set import training_hyperparameter as cifps_t
from parameter_set.TPP.cifib_parameter_set import plot_hyperparameter as cifps_p

# transformer-based fullynn_probability(IBIF) parameter set
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
    'ifib': {'train': ifps_t, 'plot': ifps_p},
    'tifib': {'train': tifps_t, 'plot': tifps_p},
    'cifib': {'train': cifps_t, 'plot': cifps_p},
}

def parameter_retriver(opt):
    model_parameter_set = plot_parameter_set[opt.model]
    required_parameter_set = model_parameter_set[opt.script_type][opt.dataset]
    
    return required_parameter_set
    