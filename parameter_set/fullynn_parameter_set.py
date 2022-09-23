# parameter sets of model multi_fullynn

stackoverflow_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "stackoverflow/no_norm.json",
    "--dataset_name", ["stackoverflow"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["stackoverflow/fullynn.json", "stackoverflow/fullynn_no_split.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

retweet_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "retweet/shift.json",
    "--dataset_name", ["retweet"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["retweet/fullynn.json", "retweet/fullynn_no_split.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

mimic_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "mimic/no_norm.json",
    "--dataset_name", ["mimic"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["mimic/fullynn.json", "mimic/fullynn_no_split.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

bookorder_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "bookorder/no_norm.json",
    "--dataset_name", ["bookorder"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["bookorder/fullynn.json", "bookorder/fullynn_no_split.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

syn_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["ablation/fullynn.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

retweet_plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["retweet/fullynn.json", "retweet/fullynn_no_split.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "50000", \
    "--dataset_name", "retweet", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "shift.json", \
    "--plot_type", ["intensity", "probability", "debug"], \
    "--dataloader_config", "retweet/plot.json", \
    "--resolution", "200"
]

stackoverflow_plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["stackoverflow/fullynn.json", "stackoverflow/fullynn_no_split.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "50000", \
    "--dataset_name", "stackoverflow", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "no_norm.json", \
    "--plot_type", ["intensity", "probability", "debug"], \
    "--dataloader_config", "stackoverflow/plot.json", \
    "--resolution", "200"
]

mimic_plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["mimic/fullynn.json", "mimic/fullynn_no_split.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "50000", \
    "--dataset_name", "mimic", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "no_norm.json", \
    "--plot_type", ["intensity", "probability", "debug"], \
    "--dataloader_config", "mimic/plot.json", \
    "--resolution", "200"
]

bookorder_plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["bookorder/fullynn.json", "bookorder/fullynn_no_split.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "50000", \
    "--dataset_name", "bookorder", \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--used_dataloader_config", "no_norm.json", \
    "--plot_type", ["intensity", "probability", "debug"], \
    "--dataloader_config", "bookorder/plot.json", \
    "--resolution", "200"
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
}