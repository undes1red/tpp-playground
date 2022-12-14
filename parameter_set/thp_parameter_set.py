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
    "--model_name", "fullynn", \
    "--model_config", ["syn/fullynn.json", "syn/fullynn_no_split.json", "syn/fullynn_zero_shift.json", "syn/fullynn_CL.json", "syn/fullynn_zero_shift_CL.json"],
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
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "100", \
    "--n_report_steps", "100", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "fullynn", \
    "--model_config", ["syn/fullynn.json", "syn/fullynn_no_split.json", "syn/fullynn_zero_shift.json", "syn/fullynn_CL.json", "syn/fullynn_zero_shift_CL.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",\
    "--wandb"
]

syn_plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "thp", \
    "--model_config", ["syn/thp.json"], \
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
    "--plot_type", ["intensity", "probability"], \
    # "--plot_type", ["debug"], \
    "--dataloader_config", "syn/plot.json", \
    "--resolution", "200"
]

training_hyperparameter = {
    'syn': syn_training_hyperparameter_list
}

plot_hyperparameter = {
    'syn': syn_plot_hyperparameter_list
}