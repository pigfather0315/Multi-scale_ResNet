import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import pdb
import torch


class Multi_scale_ResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(Multi_scale_ResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 64, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(16, 64, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
        self.upconv5 = nn.Conv2d(16, 72, 3, 1, 1, bias=True)
        #self.upconv6 = nn.Conv2d(16, 72, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_shuffle1 = nn.PixelShuffle(3)
        '''
            elif self.upscale == 16:
            self.upconv1 = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(16, 64, 3, 1, 1, bias=True)
            self.upconv3 = nn.Conv2d(16, 64, 3, 1, 1, bias=True)
            self.upconv4 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)'''
        self.postconv = nn.Conv2d(16, 8, 1, 1)
        self.HRconv = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(8, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        '''
        if self.upscale == 12:
            arch_util.initialize_weights(self.upconv2, 0.1)
            arch_util.initialize_weights(self.upconv5, 0.1)
        if self.upscale == 8:
            arch_util.initialize_weights(self.upconv2, 0.1)
            arch_util.initialize_weights(self.upconv3, 0.1)
        elif self.upscale == 24:
            arch_util.initialize_weights(self.upconv2, 0.1)
            arch_util.initialize_weights(self.upconv3, 0.1)
            arch_util.initialize_weights(self.upconv6, 0.1)
            '''

        if self.upscale == 16:
            arch_util.initialize_weights(self.upconv2, 0.1)
            arch_util.initialize_weights(self.upconv3, 0.1)
            arch_util.initialize_weights(self.upconv4, 0.1)
            arch_util.initialize_weights(self.upconv5, 0.1)



        self.Baseconv = nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=True)

    def forward(self, x16, x8 = torch.zeros(1,3,1,1).cuda(0), x12 = torch.zeros(1,3,1,1).cuda(0), x24 = torch.zeros(1,3,1,1).cuda(0) ):

        fea = self.lrelu(self.conv_first(x8))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out))) # upsample x2  B,nf,H,W -> B,64,H,W -> B,16,2*H,2*W
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out))) # upsample x2  B,16,2*H,2*W -> B,64,2*H,2*W -> B,16,4*H,4*W
        out = self.lrelu(self.pixel_shuffle(self.upconv3(out))) # upsample x2  B,16,4*H,4*W -> B,64,4*H,4*W -> B,16,8*H,8*W
        out_x8 = self.postconv(out)
        out_x8 = self.conv_last(self.lrelu(self.HRconv(out_x8)))
        base = F.interpolate(x8, scale_factor=8, mode='bilinear', align_corners=False)
        base = self.Baseconv(base)
        out_x8 += base

        #pdb.set_trace()
        fea = self.lrelu(self.conv_first(x12))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out))) # upsample x2  B,nf,H,W -> B,64,H,W -> B,16,2*H,2*W
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out))) # upsample x2  B,16,2*H,2*W -> B,64,2*H,2*W -> B,16,4*H,4*W
        out_x12 = self.lrelu(self.pixel_shuffle1(self.upconv5(out))) # upsample x3  B,16,4*H,4*W -> B,144,4*H,4*H, -> B,16,8*H,8*W
        out_x12 = self.conv_last(self.lrelu(self.HRconv(out_x12)))
        base = F.interpolate(x12, scale_factor=12, mode='bilinear', align_corners=False)
        base = self.Baseconv(base)
        out_x12 += base

        fea = self.lrelu(self.conv_first(x24))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))
        out_x24 = self.lrelu(self.pixel_shuffle1(self.upconv5(out)))
        out_x24 = self.conv_last(self.lrelu(self.HRconv(out_x24)))
        base = F.interpolate(x24, scale_factor=24, mode='bilinear', align_corners=False)
        base = self.Baseconv(base)
        out_x24 += base

        fea = self.lrelu(self.conv_first(x16))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))
        out_x16 = self.lrelu(self.pixel_shuffle(self.upconv4(out)))
        out_x16 = self.conv_last(self.lrelu(self.HRconv(out_x16)))
        base = F.interpolate(x16, scale_factor=16, mode='bilinear', align_corners=False)
        base = self.Baseconv(base)
        out_x16 += base

        return out_x8, out_x12, out_x24, out_x16
