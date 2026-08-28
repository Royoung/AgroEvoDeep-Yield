# AgroEvoDeep-Yield

This repository provides the core implementation associated with the manuscript:

**A continually evolving knowledge-guided deep learning framework for daily maize yield formation**

The released code focuses on the principal methodological components of **AgroEvoDeep-Yield**, including:

* the core neural network architecture for grain number and daily grain biomass simulation;
* the knowledge-guided loss function used for continual learning;
* the replay-based continual learning framework with shared feature extraction, dual prediction heads, gradient reversal, and domain discrimination.

This repository is intended to improve the transparency of the proposed framework and provide the main implementation needed to understand and reproduce its core learning procedures. The released scripts assume that the input tensors have been preprocessed and standardized as described in the manuscript.

## Repository contents

### `AgroEvoDeep_Yield_architecture_and_knowledge_guided_loss.py`

This file contains the core AgroEvoDeep-Yield architecture.

The model consists of two sequentially connected modules:

1. **Grain number module (`GrainNum`)**

   * implemented as a 1D convolutional neural network;
   * uses a 7-day sequence as input;
   * contains three convolutional blocks by default;
   * predicts grain number during the flowering period.
2. **Daily grain biomass module (`GrainDemand`)**

   * implemented as a multilayer perceptron;
   * receives daily environmental, soil, management, crop-growth, and predicted grain-number information;
   * predicts non-negative daily grain biomass accumulation.
3. **Integrated model (`AgroEvoDeepYield`)**

   * couples the grain number and grain biomass modules;
   * inserts the predicted grain number into the daily grain-biomass inputs;
   * accumulates daily grain biomass predictions to obtain final yield.

The file also includes the **knowledge-guided loss**:

```text
Total loss = final-yield loss
           + auxiliary_weight × daily grain-biomass loss
```

The default auxiliary weight is `80.0`, consistent with the configuration used in the manuscript.

---

### `Replay_based_continual_learning.py`

This file contains the replay-based continual learning implementation used for the grain biomass module.

The framework includes:

* a shared feature extractor;
* a field-data prediction head;
* an APSIM-replay prediction head;
* a domain discriminator;
* a Gradient Reversal Layer (GRL);
* loading and mapping of pretrained grain-biomass model weights;
* a composite loss combining daily grain-biomass and final-yield supervision;
* a progressive GRL coefficient schedule.

Replay-based continual learning is implemented in three stages:

1. **APSIM-head adaptation**  
   The shared extractor, field head, and domain discriminator are frozen, while only the APSIM replay head is trained using APSIM replay samples.
2. **GRL-based adversarial alignment**  
   The shared extractor, field head, APSIM replay head, and domain discriminator are jointly optimized. The GRL encourages the shared extractor to learn representations that are less specific to the field or APSIM data domain.
3. **Field-only fine-tuning**  
   The shared extractor and field head are further refined using field observations only, while the APSIM replay head and domain discriminator are frozen.

## Training configuration

| Training stage | Epochs | Learning rate | Training objective |
| --- | ---: | ---: | --- |
| APSIM-head adaptation | 400 | 0.05 | `Loss_M2` |
| GRL-based adversarial alignment | 800 | 0.005 | `Loss_M1 + Loss_M2 + Loss_domain` |
| Field-only fine-tuning | 10 | 0.005 | `Loss_M1` |

`Loss_M1` and `Loss_M2` denote the mechanism-guided prediction losses for field and APSIM replay samples, respectively. `Loss_domain` denotes the domain classification loss. During adversarial alignment, the weights assigned to `Loss_M2` and `Loss_domain` and the maximum GRL coefficient are all set to `1.0`. The daily grain-biomass auxiliary weight is set to `80.0`.

## Requirements

The released code requires:

```text
Python 3.x
PyTorch
```

A typical installation is:

```bash
pip install torch
```

No additional third-party Python packages are required by the two released scripts.

## Input conventions

### AgroEvoDeep-Yield architecture

The integrated `AgroEvoDeepYield` model expects:

* `grain_number_input`: a tensor with shape

```text
(batch_size, 16, 7)
```

where the final dimension represents the 7-day flowering-period sequence;

* `daily_features`: a tensor with shape

```text
(batch_size, number_of_days, 17)
```

before the predicted grain number is inserted into the daily feature vector.

The model internally standardizes the predicted grain number using the supplied training-set mean and standard deviation, inserts it into the daily feature sequence, predicts daily grain biomass, and sums the daily predictions to obtain final yield.

### Replay-based continual learning

The function:

```python
train_replay_based_continual_learning(...)
```

expects standardized PyTorch tensors for field and replay samples.

`field_group_idx` and `replay_group_idx` should be zero-based contiguous integer tensors identifying daily records belonging to the same sample. These indices are used to aggregate daily grain-biomass predictions into sample-level final yield for the composite loss.

## Example: model initialization

```python
from AgroEvoDeep_Yield_architecture_and_knowledge_guided_loss import AgroEvoDeepYield

model = AgroEvoDeepYield(
    grain_number_mean=grain_number_mean,
    grain_number_std=grain_number_std,
)
```

The forward pass returns:

```python
final_yield, daily_grain_biomass, grain_number = model(
    grain_number_input,
    daily_features,
)
```

## Example: knowledge-guided loss

```python
from AgroEvoDeep_Yield_architecture_and_knowledge_guided_loss import knowledge_guided_loss

loss, yield_loss, daily_loss = knowledge_guided_loss(
    final_yield_pred,
    final_yield_true,
    daily_grain_biomass_pred,
    daily_grain_biomass_true,
    auxiliary_weight=80.0,
)
```

## Example: replay-based continual learning

```python
from Replay_based_continual_learning import (
    MultiHeadDANN,
    load_pretrained_weights,
    train_replay_based_continual_learning,
)

model = MultiHeadDANN(
    in_features=18,
    h1=64,
    h2=32,
    dropout=0.1,
)

model = load_pretrained_weights(
    model,
    checkpoint_path="pretrained_grain_biomass_model.pth",
)

model = train_replay_based_continual_learning(
    model=model,
    x_field=x_field,
    y_field=y_field,
    field_group_idx=field_group_idx,
    x_replay=x_replay,
    y_replay=y_replay,
    replay_group_idx=replay_group_idx,
)
```
