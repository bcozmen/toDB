import torch

class NegativeSampler:
    def __init__(self, max_poison_fractions, num_patient_tokens):
        self.max_poison_fractions = max_poison_fractions
        self.num_patient_tokens = num_patient_tokens

    def assign_positive_negative_pairs(self, indices, patient_data, MAX_DATE):
        num_events = [patient.get("n_events", 0) for patient in patient_data]
        age = [
            patient["patient_stop_time"] if torch.isfinite(patient["patient_stop_time"]) else MAX_DATE - patient["patient_start_time"]
            for patient in patient_data
        ]

        positives = sorted(range(len(indices)//2), key=lambda i: (age[i], num_events[i]))
        negatives = sorted(range(len(indices)//2, len(indices)), key=lambda i: (age[i], num_events[i]))
        new_indices = [(pos, pos) for pos in positives] + [(neg, pos) for neg, pos in zip(negatives, positives)]
        labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dtype=torch.float32)
        info = [ 
            (num_events[pos], num_events[neg], age[pos], age[neg])
            for pos, neg in zip(positives, negatives)] * 2

        return new_indices, labels, info

    

    def construct_sample(self, p_tokens, min_age, event_count, n_tokens = None):
        p_max_select_token_length

    def construct_negative_sample(self, p_tokens, n_tokens, min_age, event_count=None):
        p_max_select_token_length = int((p_tokens.shape[1] - self.num_patient_tokens) * self.max_poison_fractions[0])
        n_tokens = self._ensure_time_order(n_tokens, min_age)
        return self._sample_negative(
            p_tokens, n_tokens, p_max_select_token_length,
            swap=True, event_count=event_count
        )

    def construct_positive_sample(self, p_tokens, min_age, event_count=None):
        """Mark sampled target positions without replacing their source tokens."""
        p_max_select_token_length = int((p_tokens.shape[1] - self.num_patient_tokens) * self.max_poison_fractions[0])
        candidate_tokens = self._ensure_time_order(p_tokens, min_age)
        return self._sample_negative(
            p_tokens, candidate_tokens, p_max_select_token_length,
            swap=False, event_count=event_count
        )

    def _ensure_time_order(self, n_tokens, min_age):
        start_age = n_tokens[3, :]
        duration = n_tokens[4, :]
        end_age = start_age + duration
        keep_mask = (start_age < min_age) & (torch.isinf(duration) | (end_age <= min_age))
        keep_mask[:self.num_patient_tokens] = True
        return n_tokens[:, keep_mask]

    def _sample_negative(self, p_tokens, n_tokens, p_max_select_token_length, swap, event_count=None):
        p_tokens = p_tokens.clone()
        # The final channel is a role marker: 0 = source/kept token,
        # 1 = target token. It must be assigned from sampled positions only.
        p_tokens[-1].zero_()
        selected_indices, mapping = self._sample_patient_tokens()
        patient_swap_length = len(selected_indices)

        n_max_select_token_length = int((n_tokens.shape[1] - self.num_patient_tokens) * self.max_poison_fractions[1])
        max_select_token_length = min(
            p_max_select_token_length - patient_swap_length, 
            n_max_select_token_length - patient_swap_length
        )
        select_token_length = event_count if event_count is not None else (
            torch.randint(1, max_select_token_length + 1, (1,)).item()
            if max_select_token_length > 0 else 0
        )
        select_token_length = min(
            select_token_length,
            p_tokens.shape[1] - self.num_patient_tokens,
            n_tokens.shape[1] - self.num_patient_tokens,
        )

        if select_token_length > 0:
            new_selected_indices, new_mapping = self._sample_event_tokens(p_tokens, n_tokens, select_token_length)
            selected_indices = torch.cat((selected_indices, new_selected_indices))
            mapping = torch.cat((mapping, new_mapping))

        if swap:
            p_tokens[:, mapping] = n_tokens[:, selected_indices]
        p_tokens[-1, mapping] = 1
        return p_tokens

    def _sample_patient_tokens(self):
        swap = torch.randint(self.num_patient_tokens // 2, (1,))
        selected_indices = torch.randperm(self.num_patient_tokens)[:swap.item()]
        return selected_indices, selected_indices.clone()

    def _sample_event_tokens(self, p_tokens, n_tokens, select_token_length):
        selected_indices = torch.randperm(n_tokens.shape[1] - self.num_patient_tokens)[:select_token_length] + self.num_patient_tokens
        mapping = torch.randperm(p_tokens.shape[1] - self.num_patient_tokens)[:select_token_length] + self.num_patient_tokens
        return selected_indices, mapping