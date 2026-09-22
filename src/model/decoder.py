import torch
import torch.nn as nn
from .helper import GaussianMixtureEstimator

class QuestionDecoder(nn.Module):
    def __init__(self, embedding_dim, question_dim, dictionary_size, num_questions=6, dropout=0.1, n_gaussians=10):
        super(QuestionDecoder, self).__init__()
        self.num_questions = num_questions
        self.question_dim = question_dim
        self.questions = nn.Parameter(torch.randn(num_questions, 1,question_dim) * 0.02)
        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size

        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim + question_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim + question_dim),
            nn.LayerNorm(embedding_dim + question_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim + question_dim, 1)  # Binary classification for contrastive

        self.time_estimator = GaussianMixtureEstimator(input_dim=embedding_dim + question_dim, n_gaussians=n_gaussians)
        self.code_estimator = nn.Linear(embedding_dim + question_dim, dictionary_size)  # Categorical distribution over codes
        self.value_estimator = GaussianMixtureEstimator(input_dim=embedding_dim + question_dim, n_gaussians=n_gaussians)  # For numeric values
    #batched
    def generate_questions(self, e_raw):
        batch_size, num_features, seq_length, question_dim = e_raw.shape
        question_embeddings = torch.zeros(batch_size, seq_length, self.num_questions, self.question_dim, device=e_raw.device, dtype=torch.float32)

        question_embeddings[..., 0, :] = self.questions[0]  # None -> Time
        question_embeddings[..., 1, :] = self.questions[1] + e_raw[..., 3, :, :]  # Table -> Time
        question_embeddings[..., 2, :] = self.questions[2] + e_raw[..., 0, :, :]  # Code -> Time
        question_embeddings[..., 3, :] = self.questions[3] + e_raw[..., 2, :, :] + e_raw[..., 3, :, :]  # Time + Table -> Code
        question_embeddings[..., 4, :] = self.questions[4] + e_raw[..., 0, :, :] + e_raw[..., 2, :, :]  # Time + Code -> Value
        question_embeddings[..., 5, :] = self.questions[5] # Classification question (contrastive learning)
        return question_embeddings

    def forward(self, X):
        #latent shape = (batch_size, context length, embedding_dim)
        #e_raw shape = (batch_size, context length, 6, question_dim)
        latent, e_raw = X #Embedding results

        

        #Create questions based on the raw embeddings and the latent representation
        questions = self.generate_questions(e_raw)
        #expand latent to match the shape of questions for concatenation
        latent_expanded = latent.unsqueeze(-2).expand(-1, -1, self.num_questions, -1)  # Shape: (batch_size, context length, num_questions, embedding_dim)
        questions = torch.cat([latent_expanded, questions], dim=-1)
        questions = self.trunk(questions)  # Shape: (batch_size, context length, num_questions, embedding_dim + question_dim)

        recon_questions = questions[..., :-1, :, :] # Exclude last token for reconstruction tasks
        time_params_none = self.time_estimator(recon_questions[..., [0], :])  # Shape: (batch_size, context length, num_questions, 3 * n_gaussians)
        time_params_table = self.time_estimator(recon_questions[..., [1], :])  # Shape: (batch_size, context length, num_questions, 3 * n_gaussians)
        time_params_code = self.time_estimator(recon_questions[..., [2], :])
        code_logits = self.code_estimator(recon_questions[..., [3], :])  # Shape: (batch_size, context length, num_questions, dictionary_size)
        #value_params = self.value_estimator(recon_questions[..., [4], :])  # Shape: (batch_size, context length, num_questions, 3 * n_gaussians)
        
        class_questions = questions[..., 1:, :, :]  # Exclude the first token and empty set for classification tasks
        class_logits = self.classifier(class_questions[..., [5], :])  # Shape: (batch_size, context length, 1)
        
        return class_logits, time_params_none, time_params_table, time_params_code, code_logits
        
        
