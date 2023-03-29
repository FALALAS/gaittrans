import torch


def parameter_split(model):
    conv_params = []
    trans_params = []
    for pname, p in model.named_parameters():
        if 'conv' in pname:
            conv_params += [p]
        else:
            trans_params += [p]
    return conv_params, trans_params
