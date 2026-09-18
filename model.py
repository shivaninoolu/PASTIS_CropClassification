import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, channels=32, n_classes=19):
        super().__init__()
        self.enc1 = ConvBlock(channels, channels)
        self.down = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.mid = ConvBlock(channels * 2, channels * 2)

        self.up = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec = ConvBlock(channels * 2, channels)
        self.classifier = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        m = self.mid(e2)
        u = self.up(m)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec(torch.cat([u, e1], dim=1))
        return self.classifier(d)


class TemporalSpatialCNNUNet(nn.Module):
    """
    Lightweight baseline:
    shared temporal-spatial Conv3D encoder -> temporal aggregation -> 2D U-Net.
    Input: B,T,C,H,W
    """
    def __init__(self, n_bands, n_classes=19, temporal_features=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(n_bands, temporal_features, (3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(temporal_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(temporal_features, temporal_features, (3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(temporal_features),
            nn.ReLU(inplace=True),
        )
        self.decoder = UNetDecoder(temporal_features, n_classes)

    def forward(self, x):
        # B,T,C,H,W -> B,C,T,H,W
        x = x.permute(0, 2, 1, 3, 4)
        z = self.encoder(x)
        # Temporal mean after temporal convolution.
        z = z.mean(dim=2)
        return self.decoder(z)


class UTAE(nn.Module):
    """
    Lightweight U-TAE-style semantic segmentation model.

    Each date is encoded with the same spatial CNN. A learned temporal
    attention score weights the date-wise feature maps before the U-Net
    decoder. This is an attention-based temporal encoder inspired by the
    U-TAE design, but it is a compact implementation for this assignment,
    not a byte-for-byte reproduction of the official repository.
    """
    def __init__(self, n_bands, n_classes=19, temporal_features=32):
        super().__init__()

        self.temporal_features = temporal_features

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(n_bands, temporal_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(temporal_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(temporal_features, temporal_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(temporal_features),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.Sequential(
            nn.Linear(temporal_features, temporal_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_features // 2, 1),
        )

        self.decoder = UNetDecoder(temporal_features, n_classes)

    def forward(self, x):
        # x: B,T,C,H,W
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        z = self.spatial_encoder(x)
        z = z.reshape(B, T, self.temporal_features, H, W)

        # Global temporal descriptors.
        descriptor = z.mean(dim=(-1, -2))  # B,T,F
        scores = self.attention(descriptor).squeeze(-1)  # B,T
        weights = torch.softmax(scores, dim=1)

        weighted = z * weights[:, :, None, None, None]
        fused = weighted.sum(dim=1)

        return self.decoder(fused)


def build_model(model_name, n_bands, n_classes, temporal_features=32):
    if model_name == "baseline":
        return TemporalSpatialCNNUNet(
            n_bands=n_bands,
            n_classes=n_classes,
            temporal_features=temporal_features,
        )

    if model_name == "utae":
        return UTAE(
            n_bands=n_bands,
            n_classes=n_classes,
            temporal_features=temporal_features,
        )

    raise ValueError(f"Unknown model: {model_name}")
