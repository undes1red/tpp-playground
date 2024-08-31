# A more neat way to print hyperparameters:
def print_args(opt, opt_name):
    output = f'\n{opt_name}:\n'
    for key, value in opt.__dict__.items():
        output += f'{str(key)}: {str(value)}\n'

    return output