import torch
class EventTensorizer:
    def __init__(self, num_channels):
        self.num_channels = num_channels
    def patient_to_tensor(self, patient):
        patient_tokens = torch.tensor([patient[f"patient_token_{i}"] for i in range(self.num_channels)], dtype=torch.float32)
        return patient_tokens  # Add a new dimension for concatenation with event tokens

    def events_to_tensor(self, events):
        if not events:
            return torch.empty((self.num_channels, 0), dtype=torch.float32)
        
        event_tokens = torch.tensor([[event[f"token_{i}"] for event in events] for i in range(self.num_channels)], dtype=torch.float32)
        event_tokens[3:6] 
        return event_tokens