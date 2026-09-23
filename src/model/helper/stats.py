import torch
import torch.nn as nn

class StatLogger(nn.Module):
    """Stateless, batch-local metric logger."""
    def __init__(self, logger, num_classes=None, device=None, auc_bins=64):
        super().__init__()
        self.logger = logger
        self.auc_bins = auc_bins
        self.entropy_names = (
            "Classification", "Next Table", "Next Code",
            "Future Hazard", "Future Code",
        )
        self._batch_entropies = {}

    def _binned_auc(self, scores, positive):
        """Approximate AUROC in O(N) using fixed score bins."""
        scores = torch.nan_to_num(
            scores.reshape(-1), nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0, 1)
        positive = positive.reshape(-1).bool()
        if scores.numel() != positive.numel():
            raise ValueError("scores and positive must contain the same number of elements")

        bins = (scores * self.auc_bins).long().clamp(0, self.auc_bins - 1)
        positive_hist = torch.bincount(bins[positive], minlength=self.auc_bins).float()
        negative_hist = torch.bincount(bins[~positive], minlength=self.auc_bins).float()
        positive_count = positive_hist.sum()
        negative_count = negative_hist.sum()
        negative_below = torch.cumsum(negative_hist, 0) - negative_hist
        concordant = (positive_hist * (negative_below + 0.5 * negative_hist)).sum()
        # A top-k hit metric can legitimately contain only one class.  For
        # example, if every target is present in the top-k predictions, there
        # are no negative (miss) examples.  The ROC curve is mathematically
        # undefined in that case, but for this metric the observed ranking is
        # perfect, so report 1.0.  Conversely, all misses report 0.0.  Keep
        # 0.5 only for an empty sample, where there is no evidence either way.
        has_positive = positive_count > 0
        has_negative = negative_count > 0
        auc = concordant / (positive_count * negative_count).clamp_min(1)
        return torch.where(
            has_positive & has_negative,
            auc,
            torch.where(
                has_positive,
                scores.new_tensor(1.0),
                torch.where(
                    has_negative,
                    scores.new_tensor(0.0),
                    scores.new_tensor(0.5),
                ),
            ),
        )

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
        entropy = -(probabilities * torch.log(probabilities) +
                (1 - probabilities) * torch.log(1 - probabilities)) #.mean()
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).mean()

        auc_roc = self._binned_auc(probabilities, positive)
        return torch.stack([auc_roc, precision, recall, accuracy, entropy])

    @torch.no_grad()
    def categorical(self, probabilities, predictions, labels):
        """Compute entropy, top-k accuracy, and top-k AUC."""
        num_classes = probabilities.size(-1)
        probabilities = probabilities.reshape(-1, num_classes)
        predictions = predictions.reshape(-1).long()
        labels = labels.reshape(-1).long()
        entropy = -(probabilities * torch.log(probabilities)).sum(-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).mean()

        requested_k = (1, 5, 10, 20)
        top_k = min(max(requested_k), num_classes)
        top_predictions = probabilities.topk(top_k, dim=-1).indices
        top_k_accuracy = []
        top_k_auc = []
        top_probabilities = probabilities.gather(1, top_predictions)
        for k in requested_k:
            effective_k = min(k, num_classes)
            hits = top_predictions[:, :effective_k].eq(labels[:, None]).any(dim=1)
            top_k_accuracy.append(hits.float().mean())
            confidence = top_probabilities[:, :effective_k].sum(dim=1)
            top_k_auc.append(self._binned_auc(confidence, hits))
        top_k_accuracy = torch.stack(top_k_accuracy)
        top_k_auc = torch.stack(top_k_auc)

        return (
            torch.nan_to_num(top_k_accuracy),
            torch.nan_to_num(top_k_auc),
            torch.nan_to_num(entropy),
        )

    def log_classification(self, logits, labels, log_name, task=None):
        with torch.no_grad():
            if logits.numel() == 0:
                missing = torch.full((4,), float("nan"))
                if task == "binary" or logits.size(-1) == 1:
                    self.logger.add(log_name, missing)
                else:
                    top_k = torch.full((4,), float("nan"))
                    self.logger.add(f"{log_name} Top-K Accuracy", top_k)
                    self.logger.add(f"{log_name} Top-K AUC", top_k)
                self._batch_entropies[log_name] = torch.tensor(float("nan"))
                return

            if task == "binary" or logits.size(-1) == 1:
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).long()
                metrics = self.binary(probabilities, predictions, labels)
                self.logger.add(log_name, metrics[:4].cpu())
                self._batch_entropies[log_name] = metrics[4]
            else:
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                top_k_accuracy, top_k_auc, entropy = self.categorical(
                    probabilities, predictions, labels
                )
                self.logger.add(f"{log_name} Top-K Accuracy", top_k_accuracy.cpu())
                self.logger.add(f"{log_name} Top-K AUC", top_k_auc.cpu())
                self._batch_entropies[log_name] = entropy

    def log_entropy(self):
        """Log all classification entropies as one fixed-width metric vector."""
        reference = next(iter(self._batch_entropies.values()), torch.tensor(0.0))
        missing = reference.new_tensor(float("nan"))
        values = [
            self._batch_entropies.get(name, missing)
            for name in self.entropy_names
        ]
        self.logger.add("Entropy", torch.stack(values).cpu())
        self._batch_entropies.clear()

    def log_loss(self, loss):
        self.logger.add("Loss", loss.detach().cpu())
            
            