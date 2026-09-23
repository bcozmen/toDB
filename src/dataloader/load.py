from collections import defaultdict
import duckdb
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

NUM_CHANNELS = 9
NUM_PATIENT_TOKENS = 7
MASK_CHANNEL = NUM_CHANNELS + 1  # Runtime channel after the source/target marker.
TIME_SCALE = (60 * 60 * 24 * 365) * 10
# Event and patient tensors contain raw epoch seconds until ``swap`` scales
# their time channels.  Keep this cutoff in the same units as those tensors.
MAX_DATE = 1790812800

EVENT_TYPES = {
    "encounter": "encounters", "condition": "conditions", "medication": "medications",
    "procedure": "procedures", "observation": "observations", "immunization": "immunizations",
    "allergy": "allergies", "careplan": "careplans", "imaging_study": "imaging_studies",
}
EVENT_TO_INDEX = {"patients": 0, **{table: i + 1 for i, table in enumerate(EVENT_TYPES.values())}}
REPLACED_CHANNELS = (0, 1, 2, 6, 7, 8)

from .helper import NegativeSampler, PatientRepository

class PatientEventsDataset(Dataset):
    def __init__(self, path, context_size, shuffle=True,
            poison_fraction=[0.1, 0.3], mask_rate=0.15, threads = 1, lazy_index = False):
        self.context_size = context_size
        self.lazy_index = lazy_index
        self.shuffle = shuffle
        self.poison_fraction = poison_fraction
        self.mask_rate = mask_rate

        self.repository = PatientRepository(base_path=path, num_channels=NUM_CHANNELS, threads=threads, lazy_index=lazy_index)
        #self.sampler = NegativeSampler(max_poison_fractions=max_poison_fractions, num_patient_tokens=NUM_PATIENT_TOKENS)

        self.mode = "train"
        self.index = []
        self._dataset_length = 0
        self.init_mode(self.mode)

    def init_mode(self, mode: str):
        if mode == "valid":
            mode = "val"
        self.repository.close()
        self.mode = mode
        
        if self.lazy_index:
            # Instant initialization: only read metadata count
            self._dataset_length = self.repository.get_patient_count(mode)
            self.index = []
        else:
            # Eager initialization: pre-load all index rows into RAM
            self.index = self.repository.load_patient_index(mode)
            self._dataset_length = len(self.index)


    def get_event_file_groups(self) -> list[list[int]]:
        """Returns row index groups grouped by event_file."""
        # Eager mode: build directly from RAM if index is already populated
        if self.index:
            groups = defaultdict(list)
            for index, row in enumerate(self.index):
                groups[row["event_file"]].append(index)
            return list(groups.values())
        return self.repository.fetch_event_file_groups(self.mode)

        # Lazy mode: execute fast 1-column DuckDB projection query
    def __len__(self) -> int:
        return self._dataset_length

    def __getitems__(self, indices: list[int]) -> list:
        if not indices:
            return []

        batch_patient_data, batch_events_by_patient = self.get_data_from_database(indices)
        paired_indices = self.assign_pairs(indices, batch_patient_data, batch_events_by_patient)

        
        ret_arr = [
            sample
            for pos, neg in paired_indices
            for sample in self.get_sample(pos, neg, batch_patient_data, batch_events_by_patient)
        ]

        return ret_arr

    def get_tokens(self, ix, batch_patient_data, batch_events_by_patient):
        patient = batch_patient_data[ix]
        patient_tokens = self.repository.patient_to_tensor(patient)
        event_tokens = batch_events_by_patient[patient["patient_id"]]
        return patient_tokens, event_tokens

    def get_num_swap(self, pos_event_tokens, neg_event_tokens):
        swap_fraction = torch.rand(1).item() * (self.poison_fraction[1] - self.poison_fraction[0]) + self.poison_fraction[0]
        num_swap = int(pos_event_tokens.shape[1] * swap_fraction)
        num_swap = min(num_swap, neg_event_tokens.shape[1])

        return num_swap

    def _match_event_indices(self, pos_event_tokens, neg_event_tokens, num_swap):
        """Match donor events to unique positive events of the same type."""
        pos_types = pos_event_tokens[8].tolist()
        neg_types = neg_event_tokens[8].tolist()
        pos_times = pos_event_tokens[3].tolist()
        neg_times = neg_event_tokens[3].tolist()

        # Build the candidate lists once.  Calling torch.where() and then
        # converting/filtering its result for every donor repeatedly scans all
        # positive events and causes many Python/Torch boundary crossings.
        candidates_by_type = defaultdict(list)
        for target_index, event_type in enumerate(pos_types):
            candidates_by_type[event_type].append(target_index)

        donor_indices = torch.randperm(len(neg_types))
        used_targets = set()
        matches = []

        for donor_index in donor_indices.tolist():
            if len(matches) == num_swap:
                break

            candidates = [
                index for index in candidates_by_type.get(neg_types[donor_index], [])
                if index not in used_targets
            ]
            if not candidates:
                continue

            donor_time = neg_times[donor_index]
            target_index = min(
                candidates,
                key=lambda index: abs(pos_times[index] - donor_time),
            )

            used_targets.add(target_index)
            matches.append((target_index, donor_index))

        return matches

    @staticmethod
    def _prefix_mask(replacement_mask):
        """Return a mask that starts at the first replaced position."""
        prefix_mask = torch.ones_like(replacement_mask, dtype=torch.float32)
        replaced_indices = torch.where(replacement_mask)[0]
        if replaced_indices.numel():
            prefix_mask[replaced_indices.min():] = 0.0
        return prefix_mask

    @staticmethod
    def _add_role_row(patient_tokens, event_tokens, event_role):
        """Append the source/target row and concatenate patient and event tokens."""
        patient_role = torch.zeros_like(patient_tokens[:1])
        patient_tokens = torch.cat((patient_tokens, patient_role), dim=0)
        event_tokens = torch.cat((event_tokens, event_role.unsqueeze(0)), dim=0)
        return torch.cat((patient_tokens, event_tokens), dim=1)

    def add_masks(self, pos_tokens, neg_tokens):
        """Append the same content-mask row to both paired sequences."""
        token_mask = self.sample_mask(pos_tokens.shape[1])
        pos_mask = token_mask.to(device=pos_tokens.device, dtype=pos_tokens.dtype)
        neg_mask = token_mask.to(device=neg_tokens.device, dtype=neg_tokens.dtype)
        pos_tokens = torch.cat((pos_tokens, pos_mask.unsqueeze(0)), dim=0)
        neg_tokens = torch.cat((neg_tokens, neg_mask.unsqueeze(0)), dim=0)
        return pos_tokens, neg_tokens

    def enforce_context_size(self, tokens):
        max_events = self.context_size - NUM_PATIENT_TOKENS
        if tokens.shape[1] <= max_events:
            return tokens

        encounter_type = EVENT_TO_INDEX["encounters"]
        encounter_indices = torch.where(tokens[8] == encounter_type)[0]

        if encounter_indices.numel() == 0:
            return tokens[:, :max_events]

        encounter_starts = tokens[3, encounter_indices].contiguous()
        event_starts = tokens[3].contiguous()
        event_groups = torch.searchsorted(encounter_starts, event_starts, right=True) - 1
        associated = event_groups >= 0

        group_sizes = torch.bincount(event_groups[associated], minlength=encounter_indices.numel())
        excess = tokens.shape[1] - max_events
        encounter_order = torch.randperm(encounter_indices.numel(), device=tokens.device)
        groups_to_remove = encounter_order[
            :torch.searchsorted(
                group_sizes[encounter_order].cumsum(0),
                torch.tensor(excess, device=tokens.device),
            ).item() + 1
        ]

        remove_group_mask = torch.zeros(
            encounter_indices.numel(), dtype=torch.bool, device=tokens.device
        )
        remove_group_mask[groups_to_remove] = True
        remove_mask = associated & remove_group_mask[event_groups.clamp_min(0)]
        remaining = tokens[:, ~remove_mask]
        return remaining[:, :max_events]
        
    def pad_to_context_size(self, tokens, labels):
        """Pad directly into the final-size tensors instead of concatenating."""
        seq_len = tokens.shape[1]
        if seq_len >= self.context_size:
            return tokens[:, :self.context_size], torch.zeros(self.context_size, dtype=torch.bool), labels[:self.context_size]

        padded_tokens = torch.zeros(
            (tokens.shape[0], self.context_size),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        padded_tokens[:, :seq_len] = tokens
        padding_mask = torch.zeros(
            self.context_size, dtype=torch.bool, device=tokens.device
        )
        padding_mask[seq_len:] = 1

        padded_labels = torch.zeros(
            self.context_size, dtype=labels.dtype, device=labels.device
        )
        padded_labels[:seq_len] = labels
        return padded_tokens, padding_mask, padded_labels

    def sample_mask(self, sequence_length):
        """Sample one content-mask signature for both members of a pair."""
        if self.mode != "train" or self.mask_rate <= 0:
            return torch.zeros(sequence_length, dtype=torch.bool)
        if self.mask_rate >= 1:
            return torch.ones(sequence_length, dtype=torch.bool)
        return torch.rand(sequence_length) < self.mask_rate

    def scale_time_columns(self, tokens):
        tokens[3:6] = tokens[3:6] / TIME_SCALE  # Normalize time columns
        return tokens

    def swap(self, pos_patient_tokens, pos_event_tokens, neg_patient_tokens, neg_event_tokens):
        pos_event_tokens = self.enforce_context_size(pos_event_tokens)
        neg_tokens = pos_event_tokens.clone()
        num_swap = self.get_num_swap(pos_event_tokens, neg_event_tokens)

        replacement_mask = torch.zeros(pos_event_tokens.shape[1])
        matches = self._match_event_indices(pos_event_tokens, neg_event_tokens, num_swap)

        for target_index, donor_index in matches:
            neg_tokens[list(REPLACED_CHANNELS), target_index] = neg_event_tokens[list(REPLACED_CHANNELS), donor_index]
            replacement_mask[target_index] = 1.0

        replacement_mask = torch.cat((torch.zeros(NUM_PATIENT_TOKENS), replacement_mask), dim=0)
        #neg_labels = self._prefix_mask(replacement_mask)
        neg_labels = torch.zeros_like(replacement_mask)
        pos_labels = torch.ones_like(neg_labels)

        pos = self._add_role_row(pos_patient_tokens, pos_event_tokens, replacement_mask[NUM_PATIENT_TOKENS:])
        neg = self._add_role_row(pos_patient_tokens, neg_tokens, replacement_mask[NUM_PATIENT_TOKENS:])
        pos, neg = self.add_masks(pos, neg)
        
        pos, pos_padding_mask, pos_labels = self.pad_to_context_size(pos, pos_labels)
        pos = self.scale_time_columns(pos)
        neg, neg_padding_mask, neg_labels = self.pad_to_context_size(neg, neg_labels)
        neg = self.scale_time_columns(neg)
        return (pos, pos_padding_mask, pos_labels), (neg, neg_padding_mask, neg_labels)
    def cut_negative_events(self, pos_patient_tokens, pos_event_tokens, neg_event_tokens):
        pos_age = pos_patient_tokens[4,0] if torch.isfinite(pos_patient_tokens[4,0]) else MAX_DATE - pos_patient_tokens[3,0]
        #delete all events in neg_event_tokens that are after pos_age
        neg_event_tokens = neg_event_tokens[:, neg_event_tokens[3, :] < pos_age]
        return neg_event_tokens
    
    def get_sample(self, pos, neg, batch_patient_data, batch_events_by_patient):
        pos_patient_tokens, pos_event_tokens = self.get_tokens(pos, batch_patient_data, batch_events_by_patient)
        _, neg_event_tokens = self.get_tokens(neg, batch_patient_data, batch_events_by_patient)

        neg_event_tokens = self.cut_negative_events(pos_patient_tokens, pos_event_tokens, neg_event_tokens)
        return self.swap(pos_patient_tokens, pos_event_tokens, pos_patient_tokens, neg_event_tokens)
    def get_data_from_database(self, indices: list[int]):
        if self.lazy_index:
            batch_patient_data = self.repository.fetch_patients_by_indices(self.mode, indices)
        else:
            batch_patient_data = [self.index[index] for index in indices]
        batch_events_by_patient = self.repository.fetch_events_by_patients_vectorized(self.mode, batch_patient_data)
        return batch_patient_data, batch_events_by_patient

    def assign_pairs(self, indices, patient_data, batch_events_by_patient):
        positives = list(range(len(indices)//2))
        negatives = list(range(len(indices)//2, len(indices)))
        return [(pos, neg) for pos, neg in zip(positives, negatives)]
