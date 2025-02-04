import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    def __init__(self, cogvideo_dim, dino_dim, num_heads):
        """
        Handles the cross-modal fusion between DINO and CogVideoX features.
        
        Uses DINO features as queries and temporally-weighted CogVideoX 
        features as keys/values.
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dino_dim // num_heads
        
        # Projection layers
        self.q_proj = nn.Linear(dino_dim, dino_dim)
        self.k_proj = nn.Linear(cogvideo_dim, dino_dim)
        self.v_proj = nn.Linear(cogvideo_dim, dino_dim)
        self.output_proj = nn.Linear(dino_dim, dino_dim)
        
        # Normalization
        self.norm_q = nn.LayerNorm(dino_dim)
        self.norm_kv = nn.LayerNorm(cogvideo_dim)

    def forward(self, dino_query, cogvideo_keys, temporal_weights):
        """
        Fuse features using temporally-weighted cross-attention.
        
        Args:
            dino_query: DINO features for current frame [B, C, H, W]
            cogvideo_keys: CogVideoX features [B, T, H, W, C]
            temporal_weights: Weights for each CogVideoX frame [T]
            
        Returns:
            Fused features in DINO format [B, C, H, W]
        """
        print("\n=== Cross-Modal Fusion ===")
        print(f"DINO query shape: {dino_query.shape}")
        print(f"CogVideo keys shape: {cogvideo_keys.shape}")
        print(f"Temporal weights shape: {temporal_weights.shape}")
        
        # 1. Prepare inputs
        B, C, H, W = dino_query.shape
        dino_query = dino_query.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 2. Normalize and project
        q = self.q_proj(self.norm_q(dino_query))
        k = self.k_proj(self.norm_kv(cogvideo_keys))
        v = self.v_proj(self.norm_kv(cogvideo_keys))
        print("\nCrossModal Projections:")
        print(f"q Mean: {q.mean():.4f}")
        print(f"q Std: {q.std()():.4f}")
        print(f"q Norm: {torch.norm(q).item():.4f}\n")

        print(f"k Mean: {k.mean():.4f}")
        print(f"k Std: {k.std()():.4f}")
        print(f"k Norm: {torch.norm(k).item():.4f}\n")

        print(f"v Mean: {v.mean():.4f}")
        print(f"v Std: {v.std()():.4f}")
        print(f"v Norm: {torch.norm(v).item():.4f}\n")
        
        # 3. Reshape for multi-head attention
        q = q.reshape(B, H*W, self.num_heads, self.head_dim)
        k = k.reshape(B, -1, self.num_heads, self.head_dim)
        v = v.reshape(B, -1, self.num_heads, self.head_dim)
        print("\nCrossModal Reshaped:")
        print(f"q Mean: {q.mean():.4f}")
        print(f"q Std: {q.std()():.4f}")
        print(f"q Norm: {torch.norm(q).item():.4f}\n")

        print(f"k Mean: {k.mean():.4f}")
        print(f"k Std: {k.std()():.4f}")
        print(f"k Norm: {torch.norm(k).item():.4f}\n")

        print(f"v Mean: {v.mean():.4f}")
        print(f"v Std: {v.std()():.4f}")
        print(f"v Norm: {torch.norm(v).item():.4f}\n")


        # 4. Apply scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 5. Apply temporal weighting
        attention_scores = attention_scores * temporal_weights.view(1, 1, -1, 1)
        
        # 6. Get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        print("\nCrossModal Attention Scores:")
        print(f"Mean: {attention_scores.mean():.4f}")
        print(f"Std: {attention_scores.std()():.4f}")
        print(f"Norm: {torch.norm(attention_scores).item():.4f}\n")
        print("\nCrossModal Attention Probs:")
        print(f"Probs mean: {attention_probs.mean():.4f}")
        print(f"Std: {attention_probs.std()():.4f}")
        print(f"Norm: {torch.norm(attention_probs).item():.4f}\n")

        # 7. Apply attention to values
        output = torch.matmul(attention_probs, v)
        
        # 8. Reshape and project to output space
        output = output.reshape(B, H, W, C)
        output = self.output_proj(output)
        
        # 9. Restore DINO format
        output = output.permute(0, 3, 1, 2)

        print("\nCrossModal Output:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std()():.4f}")
        print(f"Norm: {torch.norm(output).item():.4f}\n")
        print("="*10)
        
        return output