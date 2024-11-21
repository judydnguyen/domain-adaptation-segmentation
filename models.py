import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3)
        self.attention1 = nn.Identity()  # Placeholder for attention mechanism
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)
        self.attention2 = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:])
        ])

    def forward(self, features):
        x = features[0]
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Sigmoid()
        )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b7(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.features.children()))  # Encoder layers

        decoder_channels = [640, 384, 224, 160, 128, 32, 16]  # Match your architecture
        self.decoder = UNetDecoder(decoder_channels)

        self.segmentation_head = SegmentationHead(16, 1)

    def forward(self, x):
        # Extract features from encoder
        features = [x]
        for layer in self.encoder:
            features.append(layer(features[-1]))

        # Decoder
        x = self.decoder(features[::-1])  # Pass reversed features for skip connections

        # Segmentation head
        x = self.segmentation_head(x)
        return x

if __name__ == '__main__':
    model = UNet()
    input_tensor = torch.randn(1, 3, 512, 512)  # Example input tensor
    output = model(input_tensor)  # Forward pass
    print(output.shape)  # Expected: (1, 1, 512, 512)
