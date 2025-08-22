1. Rewrite task deployer.
2. Merge argument "--task_name" and "--subtask_name".

uv run start.py NTPP_evaluate --no_seed --model_name ctlstm --model_config railway/ctlstm_note.yml --lr 0.002 --used_batch_size 32 --n_training_steps 50000 --dataset_name railway --dataloader_name generic --test_data_name test_emb --used_dataloader_config ctlstm_dl.yml --dataloader_config railway/plot.yml --task_name will_llm_assign_higher_probability_to_better_events --task_config railway/evaluate_llm_probability.yml --replace --procedure_config llm_offline.yml --used_procedure_config llm_offline.yml --cuda