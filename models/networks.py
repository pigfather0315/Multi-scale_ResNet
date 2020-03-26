import torch
import models.archs.Multi_scale_ResNet_arch as Multi_scale_ResNet_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'Multi_scale_ResNet':
        netG = Multi_scale_ResNet_arch.Multi_scale_ResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


