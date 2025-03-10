import torch.nn as nn
from ..builder import NECKS


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


@NECKS.register_module()
class ConvMixer(nn.Module):
    def __init__(
        self, in_channel, dim, depth, kernel_size=9, patch_size=7, n_classes=1000
    ):
        super(ConvMixer, self).__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=patch_size, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=4),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for i in range(depth)
            ]
        )

        # self.out_conv=nn.conv

    def forward(self, input):

        out = self.mixer(input)

        return out


# def ConvMixer(in_channel, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(in_channel, dim, kernel_size=patch_size, padding=3),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[
#             nn.Sequential(
#                 Residual(
#                     nn.Sequential(
#                         nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=4),
#                         nn.GELU(),
#                         nn.BatchNorm2d(dim),
#                     )
#                 ),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim),
#             )
#             for _ in range(depth)
#         ]
#     )
