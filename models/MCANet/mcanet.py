import torch
import torch.nn.functional as F
from torch import nn
from .aspp import _ASPP
# from models.utils import initialize_weights
from .mcam import MCAM


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if downsample:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=3),
        )

    def forward(self, x):
        return self.decode(x)


class MCANet(nn.Module):
    def __init__(self, num_classes, atrous_rates=[6,12,18]):
        super(MCANet, self).__init__()
        self.sar_en1 = _EncoderBlock(1, 64) # 256->128, 1->64
        self.sar_en2 = _EncoderBlock(64, 256)  # 128->64, 64->256
        self.sar_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.sar_en4 = _EncoderBlock(512, 1024, downsample=False)  # 32->32 *** , 512->1024
        self.sar_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048

        self.opt_en1 = _EncoderBlock(4, 64) # 256->128, 4->64
        self.opt_en2 = _EncoderBlock(64, 256)  # 128->64, 64->256
        self.opt_en3 = _EncoderBlock(256, 512)  # 64->32, 256->512
        self.opt_en4 = _EncoderBlock(512, 1024, downsample=False)  # 32->32 *** , 512->1024
        self.opt_en5 = _EncoderBlock(1024, 2048, downsample=False)  # 32->32 *** , 1024->2048

        self.aspp = nn.Sequential(
            _ASPP(256 * 2, 256, atrous_rates),
            nn.Conv2d(256 * 5, 256, kernel_size=1, stride=1, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1),
        )

        self.low_level_mcam = MCAM(in_channels=256)
        self.high_level_mcam = MCAM(in_channels=2048)

        self.low_level_down = nn.Conv2d(256 * 2, 48, kernel_size=1, stride=1, padding=0)

        self.sar_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.opt_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        self.final = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        initialize_weights(self)

    def forward(self, sar, opt):
        sar_en1 = self.sar_en1(sar)
        sar_en2 = self.sar_en2(sar_en1)
        sar_en3 = self.sar_en3(sar_en2)
        sar_en4 = self.sar_en4(sar_en3)
        sar_en5 = self.sar_en5(sar_en4)

        opt_en1 = self.opt_en1(opt)
        opt_en2 = self.opt_en2(opt_en1)
        opt_en3 = self.opt_en3(opt_en2)
        opt_en4 = self.opt_en4(opt_en3)
        opt_en5 = self.opt_en5(opt_en4)

        # low_level_mcam = self.low_level_mcam(sar_en2, opt_en2)
        # low_level_features = self.low_level_down(torch.cat([low_level_mcam, sar_en2, opt_en2], 1)) # 768->48

        low_level_features = self.low_level_down(torch.cat([sar_en2, opt_en2], 1))

        # high_level_mcam = self.high_level_mcam(sar_en5, opt_en5)
        # high_level_features = torch.cat([high_level_mcam, self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5)], 1) # 2048->256
        high_level_features = torch.cat([self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5)], 1)
        # print(high_level_features.shape)
        high_level_features = self.aspp(high_level_features)

        high_level_features = F.upsample(high_level_features, sar_en2.size()[2:], mode='bilinear')

        low_high = torch.cat([low_level_features, high_level_features], 1)
        sar_opt_decoder = self.decoder(low_high)
        final = sar_opt_decoder
        # final = self.final(sar_opt_decoder)
        return F.upsample(final, sar.size()[2:], mode='bilinear')


if __name__ == "__main__":
    model = MCANet(num_classes=8)
    model.train()
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 4, 256, 256)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)
