"""
The U-Net and its two Siamese variants.

All of them end in the same decoder and differ only in how the pre and post
image enter: stacked as input channels, or through a shared encoder.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Four encoder/decoder stages with skip connections and dropout in the bottleneck."""

    def __init__(self, in_channels=3, out_channels=2, base_filters=64, dropout_p=0.2):
        super().__init__()
        f = base_filters

        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.dropout = nn.Dropout2d(dropout_p)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.dropout(self.bottleneck(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output_conv(d1)


class SiameseUNet(nn.Module):
    """Shared encoder run twice, pre and post features concatenated at every scale.

    Both images pass through the same encoder weights. The decoder receives the
    concatenated feature maps of both on every level, so its input channels are
    doubled compared to the plain U-Net.
    """

    def __init__(self, in_channels=3, out_channels=5, base_filters=64, dropout_p=0.2):
        super().__init__()
        f = base_filters

        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.dropout = nn.Dropout2d(dropout_p)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(f * 16 * 2, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 8 + f * 8 * 2, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 4 + f * 4 * 2, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 2 + f * 2 * 2, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f + f * 2, f)

        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.dropout(self.bottleneck(self.pool(e4)))
        return e1, e2, e3, e4, b

    def forward(self, pre, post):
        pre_features = self.encode(pre)
        post_features = self.encode(post)
        m1, m2, m3, m4, mb = [torch.cat([p, q], dim=1)
                              for p, q in zip(pre_features, post_features)]

        d4 = self.dec4(torch.cat([self.up4(mb), m4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), m3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), m2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), m1], dim=1))

        return self.output_conv(d1)


class SiameseResNetUNet(nn.Module):
    """Siamese U-Net with a pretrained ResNet encoder, merged once at the end.

    The decoder is built from encoder.out_channels, so switching the encoder is
    a single argument. The six encoder stages run at strides 1, 2, 4, 8, 16, 32;
    stage 0 is the input image itself and is used as the finest skip connection,
    which the ResNet stem would otherwise skip over.
    """

    def __init__(self, out_channels=5, encoder_name="resnet50",
                 encoder_weights="imagenet", decoder_channels=config.DECODER_CHANNELS):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, weights=encoder_weights, in_channels=3)
        channels = self.encoder.out_channels

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_channels = channels[5]
        for i, out_ch in enumerate(decoder_channels):
            self.ups.append(nn.ConvTranspose2d(in_channels, out_ch, kernel_size=2, stride=2))
            self.decs.append(DoubleConv(out_ch + channels[4 - i], out_ch))
            in_channels = out_ch

        self.head = nn.Sequential(DoubleConv(in_channels * 2, in_channels),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def body(self, x):
        features = self.encoder(x)
        d = features[5]
        for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
            d = dec(torch.cat([up(d), features[4 - i]], dim=1))
        return d

    def forward(self, pre, post):
        return self.head(torch.cat([self.body(pre), self.body(post)], dim=1))


def load_encoder_weights(model, checkpoint_path):
    """Copy the encoder of the localization U-Net. The decoder stays random,
    its doubled input channels do not match the plain U-Net."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prefixes = ("enc1", "enc2", "enc3", "enc4", "bottleneck", "dropout")
    target_state = model.state_dict()

    transferred = 0
    for key, value in state.items():
        if (key.split(".")[0] in prefixes and key in target_state
                and target_state[key].shape == value.shape):
            target_state[key] = value
            transferred += 1

    model.load_state_dict(target_state)
    print(f"transferred {transferred} encoder tensors from the localization model")
    return model
