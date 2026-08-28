import torch
import torch.nn as nn


class GrainNum(nn.Module):
    """1D-CNN for grain number regression."""

    def __init__(
        self,
        in_channels=16,
        seq_len=7,
        n_layers=3,
        base_channels=48,
        kernel_size=7,
        dropout=0.096,
        fc_ratio=1.0
    ):
        super().__init__()

        pad = kernel_size // 2
        layers = []
        C_in = in_channels
        C = base_channels

        for _ in range(n_layers):
            conv = nn.Conv1d(C_in, C, kernel_size=kernel_size, padding=pad)
            bn = nn.BatchNorm1d(C)
            layers.append(nn.Sequential(conv, bn, nn.ReLU()))
            C_in = C
            C = C * 2

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc_in = C_in
        self.fc_hidden = int(self.fc_in * fc_ratio)
        self.fc1 = nn.Linear(self.fc_in, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, 16 features, 7 days)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        x = self.gap(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class GrainDemand(nn.Module):
    """MLP for daily grain biomass prediction."""

    def __init__(self, in_features=18, hidden_sizes=(64, 32), dropout=0.1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return torch.relu(self.net(x)) * 10


class AgroEvoDeepYield(nn.Module):
    """Connect grain number prediction with daily grain biomass accumulation."""

    def __init__(self, grain_number_mean, grain_number_std):
        super().__init__()
        self.grain_number_model = GrainNum()
        self.grain_biomass_model = GrainDemand()
        self.grain_number_mean = grain_number_mean
        self.grain_number_std = grain_number_std

    def forward(self, grain_number_input, daily_features):
        # Predict grain number from the final seven-day sequence.
        grain_number = self.grain_number_model(grain_number_input)

        # Standardize and insert grain number as the 14th MLP feature.
        grain_number_norm = (
            (grain_number - self.grain_number_mean) / self.grain_number_std
        )
        num_days = daily_features.size(1)
        grain_number_daily = grain_number_norm.unsqueeze(1).expand(-1, num_days, -1)
        grain_biomass_input = torch.cat(
            [
                daily_features[:, :, :13],
                grain_number_daily,
                daily_features[:, :, 13:],
            ],
            dim=2,
        )

        # Predict daily grain biomass and accumulate it into final yield.
        batch_size = grain_biomass_input.size(0)
        daily_grain_biomass = self.grain_biomass_model(
            grain_biomass_input.reshape(-1, 18)
        ).reshape(batch_size, num_days)
        final_yield = daily_grain_biomass.sum(dim=1)

        return final_yield, daily_grain_biomass, grain_number.squeeze(-1)


def knowledge_guided_loss(
    final_yield_pred,
    final_yield_true,
    daily_grain_biomass_pred,
    daily_grain_biomass_true,
    auxiliary_weight=80.0,
):
    """Combine final-yield loss with daily grain biomass auxiliary loss."""

    mse = nn.MSELoss()
    yield_loss = mse(final_yield_pred, final_yield_true)
    daily_grain_biomass_loss = mse(
        daily_grain_biomass_pred,
        daily_grain_biomass_true,
    )
    total_loss = yield_loss + auxiliary_weight * daily_grain_biomass_loss

    return total_loss, yield_loss, daily_grain_biomass_loss
