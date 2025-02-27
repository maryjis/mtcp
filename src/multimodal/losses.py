import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable_temp=False):
        super().__init__()
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, embeddings_a, embeddings_b):
        """
        Args:
            embeddings_a: (batch_size, token_length_a, embedding_dim)
            embeddings_b: (batch_size, token_length_b, embedding_dim)
            
        Returns:
            clip_loss: scalar
        """

        
        # Aggregate tokens via mean pooling
        emb_a = embeddings_a.mean(dim=1)  # (B, E)
        emb_b = embeddings_b.mean(dim=1)  # (B, E)

        # Normalize embeddings
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        # Compute similarity matrix
        temperature = 0.07
        logits = torch.matmul(emb_a, emb_b.T) / temperature  # (B, B)

        # Create target labels (identity matrix indicating positive pairs)
        batch_size = emb_a.size(0)
        labels = torch.arange(batch_size, device=emb_a.device)

        # Symmetric cross-entropy loss
        
        total_loss = F.cross_entropy(logits.T, labels)
       

        return total_loss

    def get_similarity_matrix(self, embeddings_a, embeddings_b):
        """Utility method to get similarity matrix after pooling/normalization"""
        with torch.no_grad():
            emb_a = embeddings_a.mean(dim=1)
            emb_b = embeddings_b.mean(dim=1)
            return torch.matmul(
                F.normalize(emb_a, p=2, dim=-1),
                F.normalize(emb_b, p=2, dim=-1).T
            )
