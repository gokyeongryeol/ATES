# custom wandb callbacks
def replace_with_custom_callbacks(instance):
    """
    Add integration callbacks from various sources to the instance's callbacks.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """
    # Load training callbacks
    from .wb import callbacks as wb_cb

    callbacks_list = [wb_cb]

    # Replace the recent callbacks with the custom callbacks
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if k == "on_pretrain_routine_end":
                instance.callbacks[k].append(v)
            else:
                instance.callbacks[k][-1] = v
