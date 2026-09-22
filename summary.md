# Synthea-to-PyTorch database transformer

## 1. Learning objective

The model receives one complete patient/event sequence and one binary label:

- $Z=1$: a real positive patient sequence;
- $Z=0$: a corrupted sequence formed from a positive sequence and another patient.

The label is sequence-level, not token-level. Every valid sequence position predicts the same $Z$, but position $i$ can only use the prefix visible through the causal mask. Early positions therefore make a decision from less information; later positions can use more of the sequence.

If a sequence is decomposed into context $Y$ and event content $X$, and negatives are sampled from $p(X)p(Y)$ with equal class priors, the optimal discriminator logit is

$$\log \frac{p(X,Y)}{p(X)p(Y)} = \operatorname{PMI}(X,Y).$$

In this project the result is generally a density-ratio estimate relative to the actual corruption distribution. It is PMI only when the negative sampler really implements the product of marginals.

The second objective is reconstruction: answering conditional questions such as predicting a code or time value from causal context. This targets conditional distributions such as $p(X\mid Y)$.

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

The current dataset pairs the first half of each input index batch with the
second half. For each pair, the positive sequence retains its event tokens and
the negative sequence starts from the same positive event sequence, replacing
selected channels at matched event positions with events from the donor
patient. The replacement mask is shared by the positive and negative outputs,
so the source/target marker has the same positions in both classes. The swap
fraction is sampled uniformly between `poison_fraction[0]` and
`poison_fraction[1]`; the resulting count is capped by the positive event
count and donor event count, and matching can reduce the final number further.
It can also be zero for short sequences or small fractions. Donor events are
first restricted to those before the positive patient's age cutoff.

This implementation does not currently perform the previously described
three-way grouping, age/`n_events` rank matching, or shared replacement-budget
sampling across multiple pairs. `SameFileBatchSampler` instead groups indices
by event file and emits groups sized for positive/negative pairing plus a donor
pool; the dataset's current `assign_pairs()` uses only the first and second
halves of whatever index batch it receives. Event loading also reads the event
file named by the first patient in a batch, so batches must remain within one
event file.

## 3. Current model

The embedding contains categorical code/table/column features, numeric-value
projections, a start-time projection, and a source/target marker embedding. The
numeric projection now reaches the concatenated token representation; missing
or non-finite values use the dictionary's missing-value embedding. Event type
channel 8 is still not separately embedded, although table and column features
remain available. The embedding prepends a learned empty token before the
sequence and shifts the mask consistently. Reconstruction outputs exclude the
final shifted position so their length matches the original input sequence;
classification outputs exclude the empty position. Consequently, the
reconstruction heads use causal preceding context rather than the token they
are asked to predict.

The decoder's raw-feature questions use the six embedding feature slots in
this order: code, numeric value, start time, source table, source column, and
source/target marker. The table/time and time/table questions must therefore
read slot 3, while time-dependent questions read slot 2. Using slot 6 would be
out of bounds and was corrected.
The Transformer uses a causal mask and padding mask, then the decoder creates
learned question embeddings for:

- time reconstruction;
- code reconstruction;
- numeric-value reconstruction, currently scaffolded;
- sequence-level binary classification.

The classification head is trained with a global label broadcast across valid causal positions. This is intentional for the current sequence-level objective. It is not a claim that every individual token is independently positive or negative.

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

The positive and negative paths use the same nominal fraction parameters and
the same randomized budget rule, but their effective maximum budgets are not
necessarily identical.

For a negative sample, donor positions are filtered by the positive patient's
age before event matching. The donor pool can therefore be smaller. The
current `get_num_swap()` caps the requested replacements by both the positive
and donor event counts, while `_match_event_indices()` may find fewer valid
same-type matches. The actual number and locations of marked positions can
therefore vary even though the configured fractions are equal. The positive
and negative versions of a pair still receive the same final replacement mask.

The randomized count reduces sensitivity to one fixed corruption strength, but
it does not magically make the class-conditional distributions identical when
their available event pools differ. Measure the number of marked positions,
sequence length, padding, and age cutoff separately for $Z=1$ and $Z=0$.

## 6. Recommended experiments

1. Plot classification loss/AUC separately for every causal position.
2. Log marker count, event count, padding count, and age cutoff by class.
3. Run marker-only, patient-prefix-only, event-only, and shuffled-label baselines.
4. Run a marker-ablated classifier; do not assume marker removal is required for validity.
5. Use the exact same sampled positions for positive and negative versions of a pair, changing only the event contents for the negative version.
6. Construct matched negatives that preserve table, column, event type, time bucket, duration, and missingness, while changing semantic event identity.
7. Compare arbitrary cross-patient replacement against in-batch negatives and conditional negatives.
8. Evaluate on unseen patients and on a separately generated corruption distribution.
9. Keep the constant sequence-level label for the current objective; use token-level labels only for a separate candidate/event-ranking experiment.

The first priority is not to make the task artificially harder. It is to verify that the positive and negative distributions differ only in the intended dependency structure and that performance is not explained by marker counts, padding, patient identity, or obvious temporal inconsistencies.
