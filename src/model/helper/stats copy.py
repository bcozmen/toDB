import torch
import torch.nn as nn


class StatLogger(nn.Module):
    """Stateless, batch-local metric logger."""

    def __init__(self, logger, num_classes=None, device=None, auc_bins=64):
        super().__init__()
        self.logger = logger
        self.auc_bins = auc_bins

    @torch.no_grad()
    def binary(self, probabilities, predictions, labels):
        """Compute binary metrics without sorting or retaining batch state."""
        probabilities = probabilities.reshape(-1)
        predictions = predictions.reshape(-1)
        labels = labels.long().reshape(-1)
        positive = labels == 1
        predicted_positive = predictions == 1
        true_positive = (positive & predicted_positive).sum().float()
        precision = true_positive / predicted_positive.sum().float().clamp_min(1)
        recall = true_positive / positive.sum().float().clamp_min(1)
        accuracy = (predictions == labels).float().mean()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8)) +
                    (1 - probabilities) * torch.log((1 - probabilities).clamp_min(1e-8))).mean()
        bins = (probabilities.clamp(0, 1) * self.auc_bins).long().clamp_max(self.auc_bins - 1)
        positive_hist = torch.bincount(bins[positive], minlength=self.auc_bins).float()
        negative_hist = torch.bincount(bins[~positive], minlength=self.auc_bins).float()
        positive_count = positive_hist.sum()
        negative_count = negative_hist.sum()
        negative_above = torch.cumsum(negative_hist.flip(0), 0).flip(0) - negative_hist
        concordant = (positive_hist * (negative_above + 0.5 * negative_hist)).sum()
        auc_roc = torch.where((positive_count > 0) & (negative_count > 0),
                              concordant / (positive_count * negative_count),
                              probabilities.new_tensor(0.5))
        return torch.stack([auc_roc, precision, recall, accuracy, entropy])

    @torch.no_grad()
    def categorical(self, probabilities, predictions, labels):
        num_classes = probabilities.size(-1)
        probabilities = probabilities.reshape(-1, num_classes)
        predictions = predictions.reshape(-1).long()
        labels = labels.reshape(-1).long()
        correct = predictions == labels
        true_positive = torch.bincount(labels[correct], minlength=num_classes).float()
        predicted_count = torch.bincount(predictions, minlength=num_classes).float()
        label_count = torch.bincount(labels, minlength=num_classes).float()
        precision = (true_positive / predicted_count.clamp_min(1)).mean()
        recall = (true_positive / label_count.clamp_min(1)).mean()
        accuracy = correct.float().mean()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(-1).mean()
        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
        bins = (confidence.clamp(0, 1) * self.auc_bins).long().clamp_max(self.auc_bins - 1)
        positive_hist = torch.bincount(bins[correct], minlength=self.auc_bins).float()
        negative_hist = torch.bincount(bins[~correct], minlength=self.auc_bins).float()
        positive_count = positive_hist.sum()
        negative_count = negative_hist.sum()
        negative_above = torch.cumsum(negative_hist.flip(0), 0).flip(0) - negative_hist
        concordant = (positive_hist * (negative_above + 0.5 * negative_hist)).sum()
        auc_roc = torch.where((positive_count > 0) & (negative_count > 0),
                              concordant / (positive_count * negative_count),
                              probabilities.new_tensor(0.5))
        return torch.stack([auc_roc, precision, recall, accuracy, entropy])

    def log_classification(self, logits, labels, log_name, task=None):
        with torch.no_grad():
            if task == "binary" or logits.size(-1) == 1:
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).long()
                metrics = self.binary(probabilities, predictions, labels)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                metrics = self.categorical(probabilities, predictions, labels)
            self.logger.add(log_name, metrics.cpu())

    def log_loss(self, loss):
        self.logger.add("Loss", loss.detach().cpu())
