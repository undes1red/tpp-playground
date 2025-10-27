def easy_model_load(procedure_name, *args, **kwargs):
    import importlib
    module = importlib.import_module(f'src.{procedure_name}')
    return module.easy_model_load(*args, **kwargs)


if __name__ == '__main__':
    root_path = '/home/undesired/coderepo/workflow'

    '''
    easy_model_load for TPP.
    root_path = root_path
    replace_idx = '1'
    dataset_name = 'retweet'
    dataset_name_for_model = 'retweet',
    device = 'cuda:0'
    evaluation = True,
    model_name = 'ifib_c'
    lr = '0.002'
    used_batch_size = 32,
    n_training_steps = 400000
    used_dataloader_config = 'ifib_c_dl.yml'
    model_config = 'ifib_c.yml'
    '''
    model = easy_model_load('TPP', root_path, '1', 'retweet', 'retweet', 'cuda:0', evaluation = True, \
                            model_name = 'ifib_c', lr = '0.002', used_batch_size = 32, \
                            n_training_steps = 400000, used_dataloader_config = 'ifib_c_dl.yml', model_config = 'ifib_c.yml')