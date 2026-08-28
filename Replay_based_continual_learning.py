import copy
import math

import torch
import torch.nn as nn


class GrainDemand(nn.Module):
    """Original single-head model used to load pretrained weights."""

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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return torch.relu(self.net(x)) * 10.0


class GradReverseFn(torch.autograd.Function):
    """Keep features unchanged in forward and reverse gradients in backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd):
    return GradReverseFn.apply(x, lambd)


class SharedExtractor(nn.Module):
    def __init__(self, in_features=18, h1=64, h2=32, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_dim=32):
        super().__init__()
        self.fc_out = nn.Linear(in_dim, 1)

    def forward(self, z):
        out = self.fc_out(z)
        return torch.relu(out) * 10.0


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=32, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z)


class MultiHeadDANN(nn.Module):
    """Shared extractor with field, replay, and domain-specific heads."""

    def __init__(self, in_features=18, h1=64, h2=32, dropout=0.1):
        super().__init__()
        self.extractor = SharedExtractor(in_features, h1, h2, dropout)
        self.head_m1 = Head(in_dim=h2)
        self.head_m2 = Head(in_dim=h2)
        self.disc = DomainDiscriminator(
            in_dim=h2,
            hidden=h2,
            dropout=min(0.5, dropout + 0.1),
        )

    def forward_m1(self, x):
        z = self.extractor(x)
        y = self.head_m1(z)
        return y, z

    def forward_m2(self, x):
        z = self.extractor(x)
        y = self.head_m2(z)
        return y, z

    def domain_logits(self, z, lambd):
        z_rev = grl(z, lambd)
        return self.disc(z_rev)


def load_pretrained_weights(model, checkpoint_path):
    """Map a pretrained GrainDemand model to the shared extractor and heads."""

    base = GrainDemand(in_features=18, hidden_sizes=(64, 32), dropout=0.1)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    base.load_state_dict(state_dict)

    with torch.no_grad():
        model.extractor.fc1.weight.copy_(base.net[0].weight)
        model.extractor.fc1.bias.copy_(base.net[0].bias)

        model.extractor.fc2.weight.copy_(base.net[3].weight)
        model.extractor.fc2.bias.copy_(base.net[3].bias)

        model.head_m1.fc_out.weight.copy_(base.net[6].weight)
        model.head_m1.fc_out.bias.copy_(base.net[6].bias)

        model.head_m2.load_state_dict(copy.deepcopy(model.head_m1.state_dict()))

    return model


def composite_loss(preds, y_daily, group_idx, w_daily=80.0):
    """Combine daily grain biomass loss with final-yield loss."""

    mse = nn.MSELoss()
    loss_daily = mse(preds, y_daily)

    group_count = int(group_idx.max().item()) + 1
    sum_pred = torch.zeros(group_count, device=preds.device)
    sum_true = torch.zeros(group_count, device=preds.device)
    sum_pred.index_add_(0, group_idx, preds)
    sum_true.index_add_(0, group_idx, y_daily)

    loss_group = mse(sum_pred, sum_true)
    loss = w_daily * loss_daily + loss_group
    return loss, loss_daily, loss_group


def dann_lambda_schedule(progress, lambda_max):
    value = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return float(value * lambda_max)


def train_replay_based_continual_learning(
    model,
    x_field,
    y_field,
    field_group_idx,
    x_replay,
    y_replay,
    replay_group_idx,
    epochs_m2_pretrain=400,
    epochs_align=800,
    epochs_m1_finetune=10,
    lr_pretrain=5e-2,
    lr_align=5e-3,
    lr_finetune=5e-3,
    weight_decay=0.0,
    w_daily=80.0,
    alpha_m2=1.0,
    beta_domain=1.0,
    lambda_grl_max=1.0,
):
    """
    Train the model using replay-head pretraining, GRL alignment, and field fine-tuning.

    Inputs are expected to be standardized tensors. Group indices must be
    zero-based contiguous integers identifying rows from the same field sample.
    """

    # Stage A: pretrain the replay head without changing shared field knowledge.
    model.extractor.requires_grad_(False)
    model.head_m1.requires_grad_(False)
    model.head_m2.requires_grad_(True)
    model.disc.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.head_m2.parameters(),
        lr=lr_pretrain,
        weight_decay=weight_decay,
    )

    for _ in range(epochs_m2_pretrain):
        model.train()
        optimizer.zero_grad()
        pred_replay, _ = model.forward_m2(x_replay)
        pred_replay = pred_replay.squeeze(-1)
        loss_replay, _, _ = composite_loss(
            pred_replay,
            y_replay,
            replay_group_idx,
            w_daily=w_daily,
        )
        loss_replay.backward()
        optimizer.step()

    # Stage B: jointly optimize prediction and adversarial domain objectives.
    model.extractor.requires_grad_(True)
    model.head_m1.requires_grad_(True)
    model.head_m2.requires_grad_(True)
    model.disc.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_align,
        weight_decay=weight_decay,
    )
    bce_logits = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs_align + 1):
        model.train()
        optimizer.zero_grad()

        pred_field, z_field = model.forward_m1(x_field)
        pred_replay, z_replay = model.forward_m2(x_replay)
        pred_field = pred_field.squeeze(-1)
        pred_replay = pred_replay.squeeze(-1)

        loss_field, _, _ = composite_loss(
            pred_field,
            y_field,
            field_group_idx,
            w_daily=w_daily,
        )
        loss_replay, _, _ = composite_loss(
            pred_replay,
            y_replay,
            replay_group_idx,
            w_daily=w_daily,
        )

        progress = epoch / float(epochs_align)
        lambd = dann_lambda_schedule(progress, lambda_grl_max)

        shared_features = torch.cat([z_field, z_replay], dim=0)
        domain_targets = torch.cat(
            [
                torch.zeros((z_field.shape[0], 1), device=x_field.device),
                torch.ones((z_replay.shape[0], 1), device=x_field.device),
            ],
            dim=0,
        )
        domain_logits = model.domain_logits(shared_features, lambd=lambd)
        loss_domain = bce_logits(domain_logits, domain_targets)

        loss = (
            loss_field
            + alpha_m2 * loss_replay
            + beta_domain * loss_domain
        )
        loss.backward()
        optimizer.step()

    # Stage C: refine the shared extractor and field head using field data only.
    model.extractor.requires_grad_(True)
    model.head_m1.requires_grad_(True)
    model.head_m2.requires_grad_(False)
    model.disc.requires_grad_(False)

    finetune_parameters = list(model.extractor.parameters())
    finetune_parameters.extend(model.head_m1.parameters())
    optimizer = torch.optim.Adam(
        finetune_parameters,
        lr=lr_finetune,
        weight_decay=weight_decay,
    )

    for _ in range(epochs_m1_finetune):
        model.train()
        optimizer.zero_grad()
        pred_field, _ = model.forward_m1(x_field)
        pred_field = pred_field.squeeze(-1)
        loss_field, _, _ = composite_loss(
            pred_field,
            y_field,
            field_group_idx,
            w_daily=w_daily,
        )
        loss_field.backward()
        optimizer.step()

    return model
