import torch
import torch.nn as nn
from .helper import GaussianMixtureEstimator

class QuestionDecoder(nn.Module):
    def __init__(self, embedding_dim, question_dim, dictionary_size, num_questions=6, dropout=0.1, n_gaussians=10):
        super(QuestionDecoder, self).__init__()
        self.num_questions = num_questions
        self.question_dim = question_dim
        self.questions = nn.Parameter(torch.randn(num_questions, 1, question_dim) * 0.02)
        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size

        # Keep the question identity and its conditioning features separate
        # until they have been projected together. Adding embeddings is a
        # lossy way to combine them because different features can cancel
        # before the decoder sees them.
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim * 3, question_dim),
            nn.LayerNorm(question_dim),
            nn.GELU(),
        )

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

        # These tasks have the same target, but represent different
        # conditionals. Separate output heads give each conditional its own
        # output parameterization.
        self.time_estimators = nn.ModuleList([
            GaussianMixtureEstimator(
                input_dim=embedding_dim + question_dim,
                n_gaussians=n_gaussians,
            )
            for _ in range(3)
        ])
        self.code_estimator = nn.Linear(embedding_dim + question_dim, dictionary_size)  # Categorical distribution over codes
        #self.value_estimator = GaussianMixtureEstimator(input_dim=embedding_dim + question_dim, n_gaussians=n_gaussians)  # For numeric values
    #batched
    def generate_questions(self, e_raw):
        batch_size, num_features, seq_length, question_dim = e_raw.shape
        if question_dim != self.question_dim:
            raise ValueError(
                f"Expected question dimension {self.question_dim}, got {question_dim}"
            )

        # Feature slots are concatenated, not summed. The layout is
        # [learned question, feature 1, feature 2]; unused slots are zero.
        question_inputs = torch.zeros(
            batch_size,
            seq_length,
            self.num_questions,
            question_dim * 3,
            device=e_raw.device,
            dtype=e_raw.dtype,
        )
        question_features = {
            0: (),              # None -> Time
            1: (3,),            # Table -> Time
            2: (0,),            # Code -> Time
            3: (2, 3),         # Time + Table -> Code
            4: (0, 2),         # Time + Code -> Value
            5: (),              # Classification
        }

        for question_index, features in question_features.items():
            question_inputs[:, :, question_index, :question_dim] = self.questions[
                question_index
            ].expand(batch_size, seq_length, -1)
            for slot, feature_index in enumerate(features, start=1):
                question_inputs[
                    :, :, question_index, slot * question_dim:(slot + 1) * question_dim
                ] = e_raw[:, feature_index, :, :]

        return self.question_projection(question_inputs)

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
        time_params_none = self.time_estimators[0](recon_questions[..., [0], :])
        time_params_table = self.time_estimators[1](recon_questions[..., [1], :])
        time_params_code = self.time_estimators[2](recon_questions[..., [2], :])
        code_logits = self.code_estimator(recon_questions[..., [3], :])  # Shape: (batch_size, context length, num_questions, dictionary_size)
        #value_params = self.value_estimator(recon_questions[..., [4], :])  # Shape: (batch_size, context length, num_questions, 3 * n_gaussians)
        
        class_questions = questions[..., 1:, :, :]  # Exclude the first token and empty set for classification tasks
        class_logits = self.classifier(class_questions[..., [5], :])  # Shape: (batch_size, context length, 1)
        
        return class_logits, time_params_none, time_params_table, time_params_code, code_logits
        
        
