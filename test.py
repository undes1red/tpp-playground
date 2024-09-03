import submitit

def add(a, b):
    return a + b

# set timeout in min, and partition for running the job
function = submitit.helpers.CommandFunction(["nvidia-smi", ])
# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder = "log_test")
slurm_kwargs = {
    'slurm_partition': 'SCT',
    'slurm_timeout_min': 4,
    'slurm_job_name': 'test',
    'slurm_cpus_per_task': 8,
    'slurm_mem': '32GB',
    'slurm_gres': 'gpu:1',
    'slurm_qos': 'normal'
}
executor.update_parameters(**slurm_kwargs)
job = executor.submit(function)
# print(job.result())  # ID of your job
submitit.helpers.monitor_jobs([job])