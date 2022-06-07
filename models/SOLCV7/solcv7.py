import torch
import torch.nn.functional as F
from torch import nn
from aspp import _ASPP
from models.utils import initialize_weights
from models.SOLCV7.aspp import BasicRFB
from functools import reduce

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

class SAGate(nn.Module):
    def __init__(self, channels, out_ch, reduction=16):
        super(SAGate, self).__init__()
        self.channels = channels

        self.fusion1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.gate = nn.Sequential(
            nn.Conv2d(channels , channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

        self.fusion2 = nn.Conv2d(channels * 2, out_ch, kernel_size=1)

    def forward(self, sar, opt):
        b, c, h, w = sar.size()
        output = [sar, opt]

        fea_U = self.fusion1(torch.cat([sar, opt], dim=1))
        fea_s = self.avg_pool(fea_U) + self.max_pool(fea_U)
        attention_vector = self.gate(fea_s)
        attention_vector = attention_vector.reshape(b, 2, self.channels, -1)
        attention_vector = self.softmax(attention_vector)
        attention_vector = list(attention_vector.chunk(2, dim=1))
        attention_vector = list(map(lambda x: x.reshape(b, self.channels, 1, 1), attention_vector))
        V = list(map(lambda x, y: x * y, output, attention_vector))
        # concat + conv
        V = reduce(lambda x, y: self.fusion2(torch.cat([x, y], dim=1)), V)

        return V



class SOLCV7(nn.Module):
    def __init__(self, num_classes, atrous_rates=[6,12,18]):
        super(SOLCV7, self).__init__()

        d = [1, 1, 1, 2]

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

        self.aspp = BasicRFB(256 * 2, 256)

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.low_level_down = SAGate(256, 48)

        self.sar_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.opt_high_level_down = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        self.final = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
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

        low_level_features = self.low_level_down(sar_en2, opt_en2)

        high_level_features = torch.cat([self.sar_high_level_down(sar_en5), self.opt_high_level_down(opt_en5)], 1)

        print(opt_en5.shape)

        high_level_features = self.aspp(high_level_features)

        high_level_features = F.upsample(high_level_features, sar_en2.size()[2:], mode='bilinear')

        low_high = torch.cat([low_level_features, high_level_features], 1)

        sar_opt_decoder = self.decoder(low_high)
        final = self.final(sar_opt_decoder)
        return F.upsample(final, sar.size()[2:], mode='bilinear')


if __name__ == "__main__":
    model = SOLCV7(num_classes=8)
    model.train()
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 4, 256, 256)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)
