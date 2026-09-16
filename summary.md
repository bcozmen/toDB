## Synthea to PyTorch data pipeline

### 1. Synthea database and source files

Synthea generates synthetic healthcare records. The generated CSV files in
`dataset/csv/` are the source data. They contain patient demographics and
clinical events such as encounters, conditions, medications, observations,
procedures, allergies, care plans, imaging studies, devices, supplies, and
immunizations.

The pipeline uses DuckDB as the intermediate database. DuckDB can query the
CSV files directly and provides a convenient SQL layer for filtering,
renaming, casting, and joining the data.

### 2. Normalize CSV files to Parquet

`toparquet.py` defines one export specification per source table. For each
table it:

1. Reads the corresponding Synthea CSV with DuckDB.
2. Renames columns to a consistent schema, for example `PATIENT` to
	`patient_id` and `START` to `start_time`.
3. Casts dates to `TIMESTAMP` and numeric fields such as income and observation
	values to `DOUBLE` using safe casts.
4. Removes rows that cannot be associated with a patient, and where required,
	an encounter.
5. Writes compressed Zstandard Parquet files to `dataset/parquet/`.

Parquet is used as the durable, columnar representation because it is smaller
than CSV and can be scanned efficiently by DuckDB.

### 3. Build the DuckDB healthcare database

`todb.py` opens `dataset/healthcare.duckdb` and creates one DuckDB view for
each Parquet file. The views retain the normalized table names while avoiding
an unnecessary full data copy. Later SQL queries can therefore treat the
Parquet exports as ordinary tables such as `patients`, `encounters`, and
`observations`.

### 4. Build the vocabulary

`build_dictionary.py` creates the shared vocabulary used by both tokenization
and model loading. It assigns a stable integer ID to:

- Special values: missing (`None`), positive infinity, and `NaN`.
- Table names, such as `patients` and `observations`.
- Categorical values, represented as namespaced tokens such as
  `patients.gender.M` or `observations.code.8302-2`.
- Column names used as token metadata, such as `patients.income`.

Numeric values are not added to the vocabulary. They remain continuous
features. The resulting mappings are saved to `dataset/ml/vocab.pt`.

### 5. Convert records into model-ready event data

`totorch.py` constructs a unified event view from the clinical tables. Each
event is reduced to a common shape containing a patient ID, encounter ID,
start and stop times, event type, code, and an optional numeric value.
Observations are restricted to numeric observations so their `value` can be
used as a continuous feature. The imaging body-site code is used as the
event code for imaging records.

Each event is exported as eight token channels:

1. Categorical code ID.
2. Continuous value.
3. Indicator showing whether a continuous value is present.
4. Start timestamp.
5. Stop timestamp.
6. Table ID.
7. Column ID.
8. Event-type ID.

Events are sorted chronologically per patient and receive a zero-based index.
Patients are assigned to one of 20 deterministic hash buckets. Buckets 0-15
form the training set, 16-17 form validation, and 18-19 form test. Each
bucket is written as a compressed Parquet part file. A separate
`patient_index.parquet` file stores patient metadata, event counts, file
locations, and precomputed six-column patient tokens for efficient lookup.

Patient tokens encode gender, race, ethnicity, birthplace, city, and income.
The first five are categorical channels; income is stored as a continuous
value with a presence indicator. Patient records use the same eight-channel
layout as events, with six positions corresponding to these features.

### 6. Load batches for training

`load.py` provides `PatientEventsDataset`, a PyTorch `Dataset`. At startup it
loads the selected split's patient index into memory. For a batch, it uses the
index to identify the relevant Parquet part file and queries only the required
patient rows and event ranges with DuckDB.

The loader converts patient and event rows into tensors shaped as
`(8, sequence_length)`, concatenates the six patient positions with the event
positions, and pads shorter sequences to the configured context length. A
padding mask distinguishes real tokens from padding. Timestamps are scaled
from the stored epoch representation before being passed to the model.

The result is a compact, deterministic pipeline:

```text
Synthea CSV
	 -> normalized Parquet
	 -> DuckDB views
	 -> vocabulary + unified events
	 -> bucketed token Parquet
	 -> patient index
	 -> PyTorch Dataset batches
```

## Modeling objective and proposed training design

### 7. What the model is trying to learn

The main goal is to learn dependencies between a patient context and an event
or group of events. Let `Y` denote the patient context and `X` denote a selected
event payload, such as a medication, observation, or event block. The
conditional modeling objective is

$$
p(X \mid Y),
$$

which asks what event is plausible given the rest of the patient's
information. The decoder can approximate this distribution by predicting the
event type/code, its continuous value when present, and its time features.

The discrimination objective uses two kinds of pairs:

- Positive: `X` and `Y` came from the same patient context.
- Negative: `Y` came from patient A, while `X` was replaced by an event from
  independently sampled patient B.

If the negative construction approximates independent sampling, then the
positive and negative distributions are approximately

$$
p_{+}(x,y) = p(x,y),
\qquad
p_{-}(x,y) = p(x)p(y).
$$

With equal positive and negative sampling probabilities, the optimal binary
discriminator satisfies

$$
\operatorname{logit} D^*(x,y)
 = \log \frac{p(x,y)}{p(x)p(y)}
 = \operatorname{PMI}(x,y).
$$

Thus the discriminator logit estimates pointwise mutual information: positive
values indicate that `X` and `Y` co-occur more often than expected from their
individual frequencies, while negative values indicate less-than-expected
co-occurrence. Mutual information is the expectation of this quantity over
the positive joint distribution:

$$
I(X;Y)
 = \mathbb{E}_{p(x,y)}
   \left[\log \frac{p(x,y)}{p(x)p(y)}\right]
 = \mathbb{E}_{p(x,y)}[\operatorname{PMI}(x,y)].
$$

The classifier output itself is not automatically `p(X | Y)`. Under the ideal
independent-negative setup,

$$
p(x \mid y) = p(x)\exp(\operatorname{PMI}(x,y)).
$$

In practice, the model learns a density ratio through the discriminator, and
the decoder learns the conditional distribution directly. Keeping both
objectives is useful: the decoder models what can occur given a context, while
the discriminator measures whether the complete context has the dependency
structure of a real patient.

This interpretation depends on the negative sampler. If events are sampled
from a restricted distribution, for example the same event type or the same
time slot, the classifier estimates

$$
\log \frac{p_{+}(x,y \mid r)}{q_{-}(x,y \mid r)},
$$

where `r` describes the sampling rule. It is then a conditional density-ratio
or conditional-PMI-like score, not unrestricted PMI. This is still useful, but
the sampling policy must be documented and kept consistent between training
and evaluation.

Demographic information is intentionally part of `Y`. Age, sex, and other
patient attributes may be genuine causes of event probabilities. The risk is
not using these variables; the risk is allowing them to solve the task without
learning event relationships. Matched negative patients and same-type event
replacements are therefore needed to test whether the model uses more than
demographic shortcuts.

### 8. Architecture and sampling policies

The canonical Parquet data remains chronological. Randomization and corruption
are applied while constructing training examples, so the exported data stays
reproducible and can support multiple experiments.

#### Context construction

For a patient A, choose a context of `C` events, where `C` is the number of
events presented to the model. This is not a randomly selected conditioning
subset. It is the complete sampled context for that example. The initial
policy should be a chronological window, such as the latest `C` events or a
random contiguous window of `C` events.

The patient features are prepended to the context. The resulting positive
sequence is

```text
patient information, event_A1, event_A2, ..., event_AC
```

If permutation-based subset representations are being tested, randomly
permute the event order after choosing the context while retaining each
event's actual time features. This changes the order used by the causal mask,
not the clinical time associated with an event.

#### Negative construction

Sample a second patient B and a swap mask over the `C` context positions. The
negative sequence contains the same number of positions as the positive
sequence:

```text
positive: patient_A, event_A1, event_A2, event_A3, event_A4
negative: patient_A, event_A1, event_B2, event_A3, event_B4
```

Swap complete event records by default: event type, code, value, presence
indicator, time, table ID, and column ID. Swapping individual channels creates
artificially inconsistent records and can make the task trivial. A later hard
negative experiment can keep the target time fixed while replacing only the
event payload.

Do not expose the swap mask, source patient ID, or a corruption marker to the
discriminator. Otherwise it can identify a negative example without learning
patient-event compatibility. The event's normal type and metadata remain
visible because those are part of the intended problem.

Use a distribution of swap counts rather than one fixed percentage. For
context size `C`, sample a ratio from values such as 5%, 10%, 20%, and 30%,
then replace at least one event. Low corruption rates provide hard examples;
higher rates provide a stronger learning signal. Very high rates should be
avoided initially because the discriminator can detect that most of the
context came from another patient.

The negative-patient policy should progress from easy to hard:

1. Random patient B.
2. Patient B matched on age and broad demographics.
3. A same-type replacement, such as medication for medication or observation
   for observation.
4. A semantically similar replacement, such as one plausible drug or numeric
   value for another.

#### Encoder and discriminator

Embed each token using its categorical metadata, continuous value, and time
features. Use a causal Transformer encoder if the experiment is intended to
produce representations for prefixes of the sampled order. A final valid
state, or an explicit classification token placed after the context, is passed
to an MLP discriminator:

```text
patient context -> causal encoder -> final context representation -> real/fake MLP
```

The discriminator should classify the whole context. Applying the same real or
fake label to every prefix is incorrect when only later positions were
swapped, because early prefixes are identical in the positive and negative
examples.

#### Decoder and prediction heads

The encoder output can also be used as memory for a decoder or prediction
heads. The decoder receives a shifted version of the clean positive sequence
and predicts the next event representation. Separate heads are appropriate
because event identity, continuous values, and time have different domains:

```text
encoder memory -> decoder
				  ├── event type/code head
				  ├── value head
				  └── time head
```

The decoder objective is distinct from discrimination. Initially compute the
decoder loss on clean positive contexts. Use the corrupted contexts for the
real/fake objective; otherwise the decoder may be trained to treat corrupted
events as valid clinical targets.

#### Time policy

Use the patient's birth date as a natural time anchor. For event time `t_i` and
birth time `t_birth`, define age at event as

$$
a_i = \frac{t_i - t_{birth}}{365.25\ \text{days}}.
$$

Age at event is more portable and clinically interpretable than a raw epoch
timestamp. It should be supplemented, when useful, by local intervals such as

$$
\Delta a_i = a_i - \max_{j \in Y} a_j,
$$

the target age relative to the latest context event. A practical time input
can therefore contain age, log-scaled time since the previous event, and an
explicit missing-time indicator. Randomizing sequence order must not randomize
these actual event times.

### 9. Small first example

Suppose patient A has the following context:

```text
patient A: age 64, diabetes, hypertension

events:
	encounter: endocrinology
	observation: HbA1c = 8.1
	medication: metformin
```

Let

```text
Y = age 64 + diabetes + hypertension + endocrinology encounter
X+ = medication: metformin
```

The positive example is the real pair `(X+, Y)`.

Now sample patient B and replace the medication event with a same-type event:

```text
patient B event:
	medication: albuterol

X- = medication: albuterol
```

The negative example is `(X-, Y)`. Both examples contain a medication, so the
model cannot solve the task only by detecting the event type. It must decide
whether the specific medication is compatible with the context.

The discriminator receives:

```text
positive: Y + metformin
negative: Y + albuterol
```

If it assigns `D = 0.8` to the positive pair and the positive/negative class
priors are equal, the positive logit is

$$
\log \frac{0.8}{1-0.8} = \log 4.
$$

Under the ideal independent-negative assumption, this means the model
estimates a density ratio of four for that observed pair, or a PMI of
`log(4)` nats. It does not mean that the probability of metformin is 0.8;
the discriminator probability is a real-versus-corrupted probability. The
decoder's conditional head is the component that estimates which event is
likely given `Y`.

For a context size of four, a complete training example could be:

```text
positive context:
	[encounter_A, diabetes_A, HbA1c_A, metformin_A]

negative context, 50% corruption:
	[encounter_A, diabetes_A, HbA1c_B, metformin_B]

discriminator target:
	positive = 1, negative = 0

decoder target on the clean sequence:
	predict each shifted event's type/code, value, and age
```

The first experiment should compare chronological and randomly permuted
contexts using the same discriminator, corruption policy, and decoder losses.
This isolates whether random ordering improves relational learning or merely
changes the next-event prediction problem.
