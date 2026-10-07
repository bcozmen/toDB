# Synthea-to-PyTorch Database Transformer

This document describes the training pipeline implemented in the repository, especially the configuration used by [train.ipynb](train.ipynb).

## 1. End-to-end pipeline

1. Synthea tables are normalized and exported through DuckDB to Parquet files.
2. Patients are assigned to train, validation, and test partitions by the exported dataset metadata.
3. `PatientEventsDataset` loads patient rows and event rows from DuckDB-backed Parquet data.
4. `SameFileBatchSampler` groups indices by event file. Every batch is drawn from one event file because the dataset fetches that file using the first patient in the batch.
5. The dataset pairs the first half of a batch with the second half. Each pair produces one positive sequence and one corrupted negative sequence.
6. Sequences are truncated to at most `context_size - 7` event tokens, padded to `context_size`, and dynamically sliced to the longest active sequence in the batch.
7. `Embedding` converts the sequence into feature embeddings, prepends a learned empty token, and returns tokens plus the shifted padding mask.
8. `CausalMaskedTransformer` produces prefix-only hidden states using causal self-attention.
9. `PredictionDecoder` produces per-position corruption-classification logits, factorized next-event predictions, and future-important-encounter predictions.
10. `DBTransformer.get_loss()` sums one classification loss, three next-event losses, and two future-encounter losses. `LearnFrame` performs optimization, validation, checkpointing, and metric logging.

## 2. Dataset representation

The event and patient tensors contain nine semantic channels:

| Channel | Meaning | Current use |
|---:|---|---|
| 0 | Code vocabulary ID | Categorical embedding and next/future code targets |
| 1 | Numeric value, or `NaN` | Numeric projection |
| 2 | Numeric-value presence | Missing-value mask for channel 1 |
| 3 | Birth-relative event age, or `inf` | Age embedding and future-time targets |
| 4 | Duration, or `inf` | Present in tensors, not embedded |
| 5 | Time since the previous event | Time embedding and next-event time target |
| 6 | Source table vocabulary ID | Categorical embedding and next-table target |
| 7 | Source column vocabulary ID | Categorical embedding |
| 8 | Event type index | Selects event positions for next-event loss and encounter positions for future loss |

The loader appends two runtime channels:

- Channel 9 is a source/target or replacement-role marker. It is currently **not embedded**.
- Channel 10 is a shared content-mask indicator. During training one mask signature is sampled for both members of a pair; masked positions use learned mask embeddings for all six embedded feature slots.

There are seven patient-token positions followed by event tokens. Patient-token age values are set to `inf` before embedding so the model cannot directly read the patient's maximum observed age. Channels 3–5 are divided by `TIME_SCALE = 31,536,000` seconds per year. For a truncated context, channel 5 at the first retained event is replaced with that event's age since birth.

## 3. Positive and negative sequences

For each positive patient, the negative sequence starts as a copy of the positive sequence. A donor patient is selected from the same event-file batch and donor events are matched to positive events by event type and nearest age/time. The donor is filtered to events before the positive patient's observed age cutoff.

The default notebook configuration uses `poison_fraction=[0.01, 0.2]`; the actual swap fraction is sampled uniformly in that interval and at least one replacement is requested. Replacements are limited by available donor events.

Replaced channels are `(0, 1, 2, 6, 7, 8)`: code, numeric value, value-presence, table, column, and event type. Channels `(3, 4, 5)`—age, duration, and inter-event time—remain from the positive sequence. Thus this is a field-level corruption task, not an independent draw from $p(X)p(Y)$; the negative can contain semantic/time inconsistencies by design.

The replacement marker is shared by the positive and negative members. In the current implementation, the normal event-match path always has a fallback match when both sequences contain events, so patient-token swapping is generally not reached. Both sequences receive the same random content mask. Labels are broadcast over positions: positive sequence `1`, negative sequence `0`; padding is masked separately.

`SameFileBatchSampler` requires an even batch size. With `drop_last=False`, an odd final batch leaves one index unused because pairing uses the first and second halves.

## 4. Model and causal alignment

### Embedding

Six feature slots are embedded:

1. code (categorical);
2. numeric value (frequency-aware numeric projection);
3. inter-event time (time-aware numeric projection);
4. birth-relative age (time-aware numeric projection);
5. source table (categorical);
6. source column (categorical).

Duration, event type, the role marker, and the content-mask channel are not separate transformer features. The six slots are concatenated and projected to `d_model`, layer-normalized, and dropout is applied. A learned empty token is prepended, so transformer length is input length plus one.

### Transformer

`CausalMaskedTransformer` is a batch-first Transformer encoder using explicit scaled dot-product attention, causal attention, and a final layer norm. The padding mask marks right-padding; the empty token is always valid.

### Decoder alignment

If `latent` has shape `(batch, input_length + 1, d_model)`:

- `latent[..., :-1, :]` predicts the corresponding original input position. The empty state predicts the first original token, and later states predict the next event. This is used for next-event and future-encounter heads.
- `latent[..., 1:, :]` is aligned with the current original token and is used for causal classification. The first original position is excluded from classification by `classification_task()`.

The next-event head factorizes training and generation as

$$
p(\Delta t, \text{table}, \text{code}\mid h)
=p(\Delta t\mid h)
 p(\text{table}\mid h,\Delta t)
 p(\text{code}\mid h,\Delta t,\text{table}).
$$

Training uses the predicted mixture mean to condition table and code heads; targets are still teacher-forced for the loss. `generate()` samples or selects time, then table, then code without ground-truth fields.

## 5. Training objectives

### A. Causal corruption classification

`class_logits` are binary per-position logits. BCE is averaged over non-padding positions after excluding the first original position. Positive prefixes have label 1 and corrupted prefixes have label 0. This objective can learn patient/event dependency, but it may also exploit corruption artifacts, shared patient tokens, marker statistics, or impossible code/time combinations.

### B–D. Next-event prediction

Next-event losses are computed only on positive, non-padding event positions where `x[..., 8, :] > 0` (all non-patient event types, not only encounters):

- **Time:** a 10-component Gaussian mixture negative log likelihood for channel 5;
- **Table:** categorical cross entropy for channel 6;
- **Code:** categorical cross entropy for channel 0.

Table and code logits are produced from the time-conditioned factorized head. Logged metrics include time MAE/RMSE and table/code top-1, top-5, top-10, and top-20 accuracy/AUC.

### E–F. Future important-encounter prediction

For each observed encounter position (`event_type == 1`), the code suffix is searched for the closest current-or-later code in `IMPORTANT_ENCOUNTER_INDICES`. The model predicts:

- a 10-component GMM for delay;
- the important encounter code.

The delay is calculated from birth-relative ages:

$$
\Delta t = a_{\text{target}} - a_{\text{prediction}}.
$$

Future losses are applied only when an important target is visible in the supplied context. Positions without a target contribute zero. The future code head is conditioned on its predicted time. Numeric value prediction is not implemented.

The total loss is:

$$
L = L_{\text{classification}}
 + L_{\text{next time}} + L_{\text{next table}} + L_{\text{next code}}
 + L_{\text{future time}} + L_{\text{future code}}.
$$

## 6. Current limitations and risks

1. **No future-occurrence objective.** The future head learns “given that a target is visible, predict its delay and code.” It does not learn whether an important encounter will occur. `FUTURE_ENCOUNTER_HORIZON` is defined but currently unused. Add a censored survival/discrete-hazard or occurrence head if risk estimation is required.
2. **Artificial context censoring.** Future targets are searched after `enforce_context_size()` selects a random contiguous event window. Events outside the window are invisible, so a boundary can look like a no-event case. Construct future targets from the complete timeline before truncation or explicitly track right-censoring.
3. **Corruption is not an independent negative distribution.** Patient tokens and time fields are retained from the positive, while selected semantic fields come from a donor. Interpret classification as discrimination against this implemented corruption process, not automatically as PMI.
4. **Unused/dead channels.** Duration and event type are not embedded; the role marker is created but not embedded. Adding them changes the embedding layout and requires retraining.
5. **Marker leakage is possible.** The marker is not embedded, but it remains in the input tensor and should stay deliberately excluded unless role semantics are defined for every token, including patient tokens.
6. **Sampling and loader assumptions.** Batches must stay within one event file, batch size must be even, and `num_workers=0` is used so DuckDB threads are not multiplied by data-loader workers.
7. **Autoregressive evaluation is separate from teacher forcing.** Training conditions table/code on predicted time-conditioned features, while generation samples or selects its own time and table. Error accumulation should be measured without teacher forcing.

## 7. Training hyperparameters (`train.ipynb`)

### Data and runtime

| Hyperparameter | Value |
|---|---|
| Dataset path | `/home/baris/database_transformer/dataset/ml` |
| Dictionary | `dataset/ml/vocab.pt` |
| Batch size | 32 (paired as 16 positive + 16 negative sequences) |
| Context size | 4096 total positions, including 7 patient tokens |
| `poison_fraction` | `[0.01, 0.2]` (dataset default) |
| `mask_rate` | 0.15 during training; disabled for validation/test modes |
| DuckDB threads | 32 |
| Data-loader workers | 0 |
| `pin_memory` | True |
| Batch sampler | `SameFileBatchSampler` |
| `drop_last` | True |
| Training budget | 1.6 epochs over 4,000,000 samples |
| Planned steps | `int((4_000_000 / 32) * 1.6)` = 200,000 |

### Model

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 (`8 × 64`) |
| Attention heads | 8 |
| Transformer layers | 12 |
| Feed-forward dimension | 2048 (`4 × d_model`) |
| Dropout | 0.1 |
| Activation | GELU |
| Feature slots | 6 |
| Numeric/time frequency count | 10 |
| GMM components | 10 |
| Final layer norm | True |
| Gradient checkpointing | True |
| Attention | Causal scaled dot-product attention |

### Optimization and scheduling

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (`fused=True`) |
| Peak/max learning rate | `1e-4` |
| Weight decay | `1e-2` |
| Gradient clipping | `clip_grad_norm_`, max norm 5.0 |
| Gradient accumulation | 1 |
| Scheduler | `OneCycleLR` |
| Scheduler warm-up fraction | `pct_start=0.2` |
| Annealing | Cosine |
| `div_factor` | 25 (initial LR approximately `4e-6`) |
| `final_div_factor` | 10,000 (final LR approximately `1e-8`) |
| Momentum cycling | Enabled |
| Base/max momentum | 0.85 / 0.95 |
| AMP | Enabled (`dp.env_config.use_amp=True`) |
| Torch compile | Enabled with `dynamic=True` |
| Matmul precision | `torch.set_float32_matmul_precision('high')` |

## 8. Recommended evaluation

1. Plot classification metrics by causal position, not only as one aggregate.
2. Log replacement counts, successful matches, padding, selected context-window boundaries, and age cutoffs by class.
3. Evaluate next-event likelihood and calibration in addition to mixture-mean MAE/RMSE, stratified by patient age.
4. Add occurrence/survival calibration for future encounters, including right-censored trajectories.
5. Report future code top-k accuracy separately from future-event occurrence probability.
6. Evaluate autoregressive generation without teacher forcing and report error accumulation.
7. Test on unseen patients and on a separately generated corruption distribution.
