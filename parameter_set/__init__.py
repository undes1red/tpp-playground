# multi_fullynn parameter set
from .multi_fullynn_parameter_set import training_hyperparameter as mfps_t
from .multi_fullynn_parameter_set import plot_hyperparameter as mfps_p

# fullynn parameter set
from .fullynn_parameter_set import training_hyperparameter as fps_t
from .fullynn_parameter_set import plot_hyperparameter as fps_p

# Transformer Hawkes Process(THP) parameter set
from .thp_parameter_set import training_hyperparameter as thp_t
from .thp_parameter_set import plot_hyperparameter as thp_p

plot_parameter_set = {
    'multi_fullynn': {'train': mfps_t, 'plot': mfps_p},
    'fullynn': {'train': fps_t, 'plot': fps_p},
    'thp': {'train': thp_t, 'plot': thp_p}
}

def parameter_retriver(opt):
    model_parameter_set = plot_parameter_set[opt.model]
    required_parameter_set = model_parameter_set[opt.script_type][opt.dataset]
    
    return required_parameter_set
    