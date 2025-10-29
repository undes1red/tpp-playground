import torch


def load_checkpoint(logger, checkpoint_dir, model, device, evaluation=True, compile=False):
    '''
    Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DP if possible.
    '''
    model_state_dict = torch.load(checkpoint_dir, map_location=device)

    # remove _orig_mod from keys if any of them exist.
    # _orig_mod exists when the original model is replaced by a compiled version.
    new_dict = {}
    recorded_keys = []
    # Restore the key if torch.compile is used.

    # Remove _orig_mod from the key name.
    # Now we can load compiled checkpoints into uncompiled models.
    if not compile:
        for item in model_state_dict:
            if '_orig_mod' in item:
                new_dict[item.replace('._orig_mod', '')] = model_state_dict[item]
                recorded_keys.append(item)

    for recorded_key in recorded_keys:
        model_state_dict.pop(recorded_key, None)
    model_state_dict.update(new_dict)

    model.load_state_dict(model_state_dict)
    if evaluation:
        model.requires_grad_(requires_grad = False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model restore completed. The number of trainable parameters in this model: {trainable_parameters} out of {total_params}.')

    return model
