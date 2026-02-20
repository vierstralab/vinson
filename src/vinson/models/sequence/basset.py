import torch
from typing import List

from vinson.models.shared import MLPBlock

class BassetTrunk(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=4, out_channels=300, kernel_size=19, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=300, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, padding=(3 - 1) // 2),
        )
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=300, out_channels=200, kernel_size=11, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=200, out_channels=200, kernel_size=7, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu3 = torch.nn.ReLU()

        self.output_dim = 200

        # self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        # flatten
        # x = self.flatten(x)
        x = torch.flatten(x, start_dim=1)

        return x


class BassetTrunkEmbed(BassetTrunk):
    def __init__(self, embeds: List[MLPBlock], **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(embeds, MLPBlock):
            embeds = [embeds] * 2

        assert len(embeds) == 2, f"Number of embedding MLPs must match the number of blocks in the trunk. Expected 2, got {len(embeds)}"

        self.bias2 = torch.nn.Sequential(
            embeds[0],
            torch.nn.Linear(embeds[0].output_dim, self.layer2[0].out_channels)
        )

        self.bias3 = torch.nn.Sequential(
            embeds[1],
            torch.nn.Linear(embeds[1].output_dim, self.layer3[0].out_channels)
        )


    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)

        x_conv = self.layer2(x)
        x_bias = self.bias2(embed)[:, :, None]
        x = self.relu2(x_conv + x_bias)

        x_conv = self.layer3(x)
        x_bias = self.bias3(embed)[:, :, None]
        x = self.relu3(x_conv + x_bias)

        # Flatten features
        x = torch.flatten(x, start_dim=1)

        return x
