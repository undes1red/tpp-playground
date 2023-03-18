# parameter sets of model tfullynn

stackoverflow_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "stackoverflow/tfullynn_dl.json",
    "--dataset_name", "stackoverflow", \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "10000", \
    "--model_name", "tfullynn", \
    "--model_config", "stackoverflow/tfullynn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

retweet_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "retweet/tfullynn_dl.json",
    "--dataset_name", "retweet", \
    "--n_training_steps", "100000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "20000", \
    "--model_name", "tfullynn", \
    "--model_config", "retweet/tfullynn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

mooc_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "mooc/tfullynn_dl.json",
    "--dataset_name", "mooc", \
    "--n_training_steps", "400000", \
    "--n_evaluation_steps", "5000", \
    "--n_report_steps", "5000", \
    "--b", "32", \
    "--n_warmup_steps", "80000", \
    "--model_name", "tfullynn", \
    "--model_config", "mooc/tfullynn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

bookorder_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataloader_config", "bookorder/tfullynn_dl.json",
    "--dataset_name", "bookorder", \
    "--n_training_steps", "20000", \
    "--n_evaluation_steps", "100", \
    "--n_report_steps", "100", \
    "--b", "8", \
    "--n_warmup_steps", "4000", \
    "--model_name", "tfullynn", \
    "--model_config", "bookorder/tfullynn.json",
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "--n_cycles", "0.5",
]

syn_training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "generic", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"], \
    "--n_training_steps", "10000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "128", \
    "--n_warmup_steps", "1000", \
    "--model_name", "tfullynn", \
    "--model_config", "syn/tfullynn.json",
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
    "--model_name", "fullynn", \
    "--model_config", ["retweet/fullynn.json", "retweet/fullynn_no_split.json"], \
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
    "--model_name", "fullynn", \
    "--model_config", ["stackoverflow/fullynn.json", "stackoverflow/fullynn_no_split.json"], \
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

mooc_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["mimic/fullynn.json", "mimic/fullynn_no_split.json"], \
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
    "--used_dataloader_config", "no_norm.json", \
    # "--plot_type", ["intensity", "probability", "debug"], \
    "--plot_type", ["debug"], \
    # "--plot_type", ["intensity", "probability"], \
    "--dataloader_config", "mimic/plot.json", \
    "--resolution", "200"
]

bookorder_plot_hyperparameter_list = [
    "train.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["bookorder/fullynn.json", "bookorder/fullynn_no_split.json"], \
    # "--model_config", ["bookorder/fullynn_zero_shift.json"], \
    "--lr", "0.001", \
    "--batch_size", "64", \
    "--n_training_steps", "1000", \
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
    "--model_name", "fullynn", \
    "--model_config", ["syn/fullynn.json", "syn/fullynn_no_split.json"], \
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
    'mooc': mooc_training_hyperparameter_list,
    'syn': syn_training_hyperparameter_list
}

plot_hyperparameter = {
    'retweet': retweet_plot_hyperparameter_list,
    'stackoverflow': stackoverflow_plot_hyperparameter_list,
    'bookorder': bookorder_plot_hyperparameter_list,
    'mooc': mooc_plot_hyperparameter_list,
    'syn': syn_plot_hyperparameter_list
}