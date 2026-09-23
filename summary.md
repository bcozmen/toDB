# Synthea-to-PyTorch database transformer

## 1. Learning objective

The model receives one complete patient/event sequence and one binary label:

- $Z=1$: a real positive patient sequence;
- $Z=0$: a corrupted sequence formed from a positive sequence and another patient.

The label is sequence-level, not token-level. Every valid sequence position predicts the same $Z$, but position $i$ can only use the prefix visible through the causal mask. Early positions therefore make a decision from less information; later positions can use more of the sequence.

If a sequence is decomposed into context $Y$ and event content $X$, and negatives are sampled from $p(X)p(Y)$ with equal class priors, the optimal discriminator logit is

$$
\log \frac{p(X,Y)}{p(X)p(Y)} = \operatorname{PMI}(X,Y).
$$

In this project the result is generally a density-ratio estimate relative to the actual corruption distribution. It is PMI only when the negative sampler really implements the product of marginals.

The second objective is causal next-event prediction. Numeric-value prediction is deferred for now. Given a hidden representation $h$ of the observed prefix, the current model factorizes the next event as

$$
\begin{aligned}
p(\Delta t, \text{table}, \text{code}\mid h)
={}&p(\Delta t\mid h)\\
&\cdot p(\text{table}\mid h,\Delta t)\\
&\cdot p(\text{code}\mid h,\Delta t,\text{table}).
\end{aligned}
$$

This supports both local next-token reconstruction and autoregressive generation.

## 2. Data and runtime representation

Synthea CSV tables are normalized with DuckDB and exported to Parquet. Patients are assigned to deterministic train/validation/test hash buckets. Event part files contain chronologically ordered events, and a patient index contains patient features and the event-file location.

The stored event and patient records contain nine channels:

| Channel | Meaning |
|---:|---|
| 0 | Concept/code vocabulary ID |
| 1 | Continuous value, or `NaN` |
| 2 | Continuous-value presence mask |
| 3 | Start age/time, or `inf` |
| 4 | Duration, or `inf` |
| 5 | Time since the previous event |
| 6 | Source table vocabulary ID |
| 7 | Source column vocabulary ID |
| 8 | Event type index |

The loader appends runtime channel 9 as a source/target marker. It is initialized to zero and set to one at sampled positions. This marker identifies sampled positions; it is not the sequence label.

The first seven sequence positions are patient features. Time channels are normalized by `TIME_SCALE`. Positive examples retain their original tokens and mark sampled positions. Negative examples begin as copies of the positive event sequence, replace selected fields from another patient, and use the same marker mechanism. The current `REPLACED_CHANNELS` set replaces code, numeric value, value-presence, source table, source column, and event type, but leaves the time, duration, and inter-event-time channels from the positive sequence unchanged.

The runtime content mask and observation missingness are distinct concepts. The current mask channel hides feature content while retaining the event token. A future missing-observation augmentation should hide events from the model input while constructing targets from the complete timeline; it should not redefine the next event or observation endpoint from the shortened input. A separate event-observed indicator is preferable if the missingness mechanism is known to the model.

The current dataset pairs the first half of each input index batch with the second half. For each pair, the positive sequence retains its event tokens and the negative sequence starts from the same positive event sequence, replacing selected channels at matched event positions with events from the donor patient. The replacement mask is shared by the positive and negative outputs, so the source/target marker has the same positions in both classes. The swap fraction is sampled uniformly between `poison_fraction[0]` and `poison_fraction[1]`; the resulting count is capped by the positive event count and donor event count, and matching can reduce the final number further. It can also be zero for short sequences or small fractions. Donor events are first restricted to those before the positive patient's age cutoff.

This implementation does not currently perform the previously described three-way grouping, age/`n_events` rank matching, or shared replacement-budget sampling across multiple pairs. `SameFileBatchSampler` instead groups indices by event file and emits groups sized for positive/negative pairing plus a donor pool; the dataset's current `assign_pairs()` uses only the first and second halves of whatever index batch it receives. Event loading also reads the event file named by the first patient in a batch, so batches must remain within one event file.

## 3. Current model

The embedding contains categorical code/table/column features, numeric-value projections, a start-time projection, and a source/target marker embedding. The numeric projection reaches the concatenated token representation; missing or non-finite values use the dictionary's missing-value embedding. Event type channel 8 is still not separately embedded, although table and column features remain available.

The embedding prepends a learned empty token and shifts the mask consistently. The transformer uses a causal attention mask and padding mask. There are two deliberately different alignments after this prepend:

- `latent[..., :-1, :]` is used for next-token prediction. Its first position is the empty-token representation, which predicts the first original token (the first patient feature token). Its remaining positions predict the corresponding later original tokens from their causal prefixes. The final latent position is the state after the last observed token and is used for next-event generation.
- `latent[..., 1:, :]` is used for classification. The empty-token position is excluded, and each classification logit is aligned with the corresponding original token position. Thus classification uses the current token's causal transformer representation rather than the empty representation used to predict the first token.

The old question-slot decoder has been replaced by `PredictionDecoder` and a factorized event head. Time uses a Gaussian-mixture head. Table and code use categorical logits. `table_condition_embedding` and `code_condition_embedding` convert categorical IDs into dense condition vectors; the integer IDs are never concatenated directly with hidden states.

Training uses teacher forcing: the true target time, table, and code are passed to the later factorized condition heads. This correctly trains the conditional factorization, but it must not be used as the generation path because future ground-truth fields are unavailable at inference time.

The model also provides an autoregressive `generate_next()` path. It predicts fields in order:

1. Predict the time mixture and use its mixture mean.
2. Predict the table and select the highest-probability table.
3. Condition on that predicted time and table, then predict the code.
4. Return the generated time, table, and code. Numeric-value generation is deferred.

The current generation path is deterministic: the mixture mean for time and argmax for categorical fields. Sampling can replace these choices without changing the architecture.

The classification head continues to use the causal hidden representation and is trained with a sequence-level label broadcast across valid positions. This is intentional for the current density-ratio/classification objective; it is not a claim that each token independently has a positive or negative label.

During training, the embedding layer also applies uniform random token masking with configurable `mask_rate` (currently 0.15 by default). Masking can affect the seven patient/demographic tokens and event tokens, but never padding. The source/target marker remains visible because it is a sampling-role indicator shared by positive and negative examples rather than part of token content. Masking is disabled in evaluation mode.

The decoder also exposes a discrete-time survival predictor for the next encounter after each encounter landmark. By default the interval boundaries are one week, one month, six months, one year, five years, and ten years. The hazard for interval $k$ is conditional on no earlier encounter, and cumulative risk is derived as

$$
S_k = \prod_{j=1}^{k}(1-\lambda_j),
\qquad
F_k = 1-S_k.
$$

The encounter-code head produces a code distribution only for the interval in which the next encounter occurs. Earlier intervals receive observed negative hazard targets; later intervals are excluded after an event. If no encounter is observed, an interval is supervised negatively only when follow-up reaches its upper boundary; otherwise it is censored and masked. The targets are built with a vectorized reverse cumulative search for the first encounter strictly after each landmark. Horizons use the loader's normalized ten-year time units. `FutureEncounterPredictor.cumulative_risk()` converts interval hazard logits into monotone cumulative risk estimates.

The current `_future_targets()` implementation derives the follow-up boundary from the final non-padding token in the supplied context. Therefore, context truncation or removal of future encounter groups can create artificial censoring or skip the true next encounter. A correct missing-observation or fixed-context pipeline must construct targets from the complete patient timeline, preserve the true observation endpoint, and apply context truncation only to the model input. This separation is not yet implemented in the current loader.

## 4. Classification difficulty and possible shortcuts

Fast classification can be legitimate: later positions have more evidence, and patient/event histories may be strongly dependent. It can also be exploiting the corruption process.

Important checks:

1. **Corruption artifacts:** event replacement can create impossible temporal, table/code, or patient/event combinations.
2. **Patient identity:** demographic features and repeated patient patterns may identify the patient without modeling general event dependence.
3. **Marker nuisance:** exposing channel 9 does not directly leak $Z$ if positive and negative examples use exactly the same marker distribution. It can still be a shortcut if marker counts or locations differ between classes.
4. **Negative distribution:** same-event-file negatives are not automatically $p(X)p(Y)$, so PMI interpretation requires measuring or correcting the actual negative distribution. The current negative also keeps the positive sequence's time-related channels, so this is a field-level replacement distribution rather than an independent event-sequence draw.
5. **Causal position:** the apparent accuracy should be plotted by position. A model that becomes accurate only after seeing many events may be behaving as expected.

The classifier using the marker is not automatically invalid. Removing the marker can change the task because the model no longer knows which positions were deliberately sampled for the reconstruction/corruption experiment. A useful comparison is to keep it for the main model and run a marker-ablated model as a control.

## 5. Corruption-budget analysis

The positive and negative paths use the same nominal fraction parameters and the same randomized budget rule, but their effective maximum budgets are not necessarily identical.

For a negative sample, donor positions are filtered by the positive patient's age before event matching. The donor pool can therefore be smaller. The current `get_num_swap()` caps the requested replacements by both the positive and donor event counts, while `_match_event_indices()` may find fewer valid same-type matches. The actual number and locations of marked positions can therefore vary even though the configured fractions are equal. The positive and negative versions of a pair still receive the same final replacement mask.

The randomized count reduces sensitivity to one fixed corruption strength. The final replacement mask is nevertheless identical for the positive and negative members of each pair: `swap()` constructs one `replacement_mask`, passes that same mask to `_add_role_row()` for both outputs, and pads both sequences from the same positive event sequence. Donor filtering and same-type matching can change how many replacements are possible before the mask is created, and can change the distribution across pairs, but they do not give the positive and negative versions of one pair different effective budgets. Measure requested swaps, matched swaps, marker counts, sequence length, padding, and age cutoffs across pairs to detect batch-level distribution differences.

## 6. Recommended experiments

1. Plot classification loss/AUC separately for every causal position.
2. Log marker count, event count, padding count, and age cutoff by class.
3. Run marker-only, patient-prefix-only, event-only, and shuffled-label baselines.
4. Run a marker-ablated classifier; do not assume marker removal is required for validity.
5. Use the exact same sampled positions for positive and negative versions of a pair, changing only the event contents for the negative version.
6. Construct matched negatives that preserve table, column, event type, time bucket, duration, and missingness, while changing semantic event identity.
7. Compare arbitrary cross-patient replacement against in-batch negatives and conditional negatives.
8. Evaluate on unseen patients and on a separately generated corruption distribution.
9. Evaluate autoregressive generation without teacher-forcing inputs; report calibration and error accumulation separately from teacher-forced likelihood.
10. For future encounter prediction, evaluate time-bin calibration, censoring-aware likelihood, encounter-type top-$k$ accuracy, and simulated event-count calibration.

The first priority is not to make the task artificially harder. It is to verify that the positive and negative distributions differ only in the intended dependency structure and that performance is not explained by marker counts, padding, patient identity, or obvious temporal inconsistencies.
