import torch


def load_checkpoint(logger, checkpoint_dir, model, device, evaluation = True):
    '''
    Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DP if possible.
    '''
    model_raw = torch.load(checkpoint_dir, weights_only = False, map_location = device)
    model_state_dict = model_raw['model']
    model.load_state_dict(model_state_dict)
    if evaluation:
        model.requires_grad_(requires_grad = False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model restore completed. The number of trainable parameters in this model: {trainable_parameters} out of {total_params}.')

    return model