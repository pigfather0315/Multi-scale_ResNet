import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    elif model == 'multi_sr':  # GAN-based super resolution, SRGAN / ESRGAN
        from .Multi_SR_model import Multi_SRModel as M
    elif model == 'multi_sr_dloss':  # GAN-based super resolution, SRGAN / ESRGAN
        from .Multi_SR_model_dloss import Multi_SRModel_dloss as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
