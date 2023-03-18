# parameter sets of model thp

stackoverflow_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "stackoverflow/shift.json",
    "--dataset_name", ["stackoverflow"], \
    "--n_training_steps", "3000", \
    "--n_evaluation_steps", "50", \
    "--n_report_steps", "50", \
    "--b", "128", \
    "--n_warmup_steps", "600", \
    "--model_name", "rmtpp", \
    "--model_config", ["stackoverflow/rmtpp.json"],
    "--lr", "0.001", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

retweet_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "retweet/shift.json",
    "--dataset_name", ["retweet"], \
    "--n_training_steps", "6000", \
    "--n_evaluation_steps", "50", \
    "--n_report_steps", "50", \
    "--b", "128", \
    "--n_warmup_steps", "1200", \
    "--model_name", "rmtpp", \
    "--model_config", ["retweet/rmtpp.json"],
    "--lr", "0.001", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

mimic_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "mimic/shift.json",
    "--dataset_name", ["mimic"], \
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "100", \
    "--n_report_steps", "100", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "rmtpp", \
    "--model_config", ["mimic/rmtpp.json"],
    "--lr", "0.001", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

bookorder_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "bookorder/shift.json",
    "--dataset_name", ["bookorder"], \
    "--n_training_steps", "2000", \
    "--n_evaluation_steps", "25", \
    "--n_report_steps", "25", \
    "--b", "64", \
    "--n_warmup_steps", "500", \
    "--model_name", "rmtpp", \
    "--model_config", ["bookorder/rmtpp.json"],
    "--lr", "0.001", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

syn_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "100", \
    "--n_report_steps", "100", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "rmtpp", \
    "--model_config", ["syn/rmtpp.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

retweet_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "rmtpp", \
    "--model_config", ["retweet/rmtpp.json"], \
    "--lr", "0.001", \
    "--batch_size", "128", \
    "--n_training_steps", "6000", \
    "--dataset_name", "retweet", \
    "--dataloader_name", "syn", \
    "--figure_count", "1", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "shift.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", ["debug"], \
    "--dataloader_config", "retweet/plot.json", \
    "--resolution", "200"
]

stackoverflow_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "rmtpp", \
    "--model_config", ["stackoverflow/rmtpp.json"], \
    # "--model_config", ["bookorder/fullynn_zero_shift.json"], \
    "--lr", "0.001", \
    "--batch_size", "128", \
    "--n_training_steps", "3000", \
    "--dataset_name", "stackoverflow", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", ["shift.json"], \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", ["debug"], \
    "--dataloader_config", "stackoverflow/plot.json", \
    "--resolution", "200"
]

mimic_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "rmtpp", \
    "--model_config", ["mimic/rmtpp.json"], \
    # "--model_config", ["bookorder/fullynn_zero_shift.json"], \
    "--lr", "0.001", \
    "--batch_size", "128", \
    "--n_training_steps", "10000", \
    "--dataset_name", "mimic", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "shift.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", ["debug"], \
    # "--plot_type", ["intensity", "probability"], \
    "--dataloader_config", "mimic/plot.json", \
    "--resolution", "200"
]

bookorder_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "rmtpp", \
    "--model_config", ["bookorder/rmtpp.json"], \
    # "--model_config", ["bookorder/fullynn_zero_shift.json"], \
    "--lr", "0.001", \
    "--batch_size", "64", \
    "--n_training_steps", "2000", \
    "--dataset_name", "bookorder", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "shift.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", ["debug"], \
    "--dataloader_config", "bookorder/plot.json", \
    "--resolution", "200"
]

syn_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "rmtpp", \
    "--model_config", ["syn/rmtpp.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "10000", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    # "--plot_type", ["intensity", "probability"], \
    "--plot_type", ["debug"], \
    "--dataloader_config", "syn/plot.json", \
    "--resolution", "200", \
    "--synthetic_evaluation"
]

training_hyperparameter = {
    'stackoverflow': stackoverflow_training_hyperparameter_list,
    'retweet': retweet_training_hyperparameter_list,
    'bookorder': bookorder_training_hyperparameter_list,
    'mimic': mimic_training_hyperparameter_list,
    'syn': syn_training_hyperparameter_list
}

plot_hyperparameter = {
    'retweet': retweet_plot_hyperparameter_list,
    'stackoverflow': stackoverflow_plot_hyperparameter_list,
    'mimic': mimic_plot_hyperparameter_list,
    'bookorder': bookorder_plot_hyperparameter_list,
    'syn': syn_plot_hyperparameter_list
}