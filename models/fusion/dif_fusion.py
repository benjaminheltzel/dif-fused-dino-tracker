import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBlockFusion:
    def __init__(self, feature_dim=1920, num_heads=8):
        """
        Initialize the spatial fusion mechanism.
        
        The attention uses multiple heads to capture different aspects of the 
        spatial relationships between blocks. We keep feature_dim=1920 to match
        CogVideoX's native dimension.
        """
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Create attention components
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Add layer normalization for stability
        self.norm = nn.LayerNorm(feature_dim)
        
        # Initialize all weights using Xavier initialization
        # This helps with stable gradient flow early in training
        self._init_weights()

    def process_single_frame(self, frame_features):
        """
        Process features from a single temporal position across blocks.
        
        Input shape: [B, 3, H, W, C] where:
            - B is batch size
            - 3 is number of blocks (0, 14, 29)
            - H=30, W=45 are spatial dimensions
            - C=1920 is feature dimension
            
        This maintains spatial structure while fusing information across blocks.
        """
        print("\n=== Spatial Block Fusion ===")

        B, num_blocks, H, W, C = frame_features.shape
        print(f"Input shape: {frame_features.shape}")
        
        # Reshape to treat each spatial position as a token
        # [B, 3, H, W, C] -> [B, 3, H*W, C]
        features = frame_features.reshape(B, num_blocks, H*W, C)
        print(f"Reshaped: {features.shape}")
        
        # Split into heads
        # [B, 3, H*W, C] -> [B, 3, H*W, num_heads, head_dim]
        features = features.reshape(
            B, num_blocks, H*W, self.num_heads, self.head_dim
        )
        print(f"Multi-head shape: {features.shape}")
        
        # Compute Q, K, V projections
        # Each spatial position attends to the same position in other blocks
        Q = self.q_proj(features)
        K = self.k_proj(features)
        V = self.v_proj(features)
        print("\nProjections:")
        print(f"Q Mean: {Q.mean():.4f}")
        print(f"Q Std: {Q.std()():.4f}")
        print(f"Q Norm: {torch.norm(Q).item():.4f}\n")

        print(f"K Mean: {K.mean():.4f}")
        print(f"K Std: {K.std()():.4f}")
        print(f"K Norm: {torch.norm(K).item():.4f}\n")

        print(f"V Mean: {V.mean():.4f}")
        print(f"V Std: {V.std()():.4f}")
        print(f"V Norm: {torch.norm(V).item():.4f}\n")

        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        print("\nAttention Scores:")
        print(f"Mean: {attention_scores.mean():.4f}")
        print(f"Std: {attention_scores.std():.4f}")
        print(f"Norm: {torch.norm(attention_scores).item():.4f}\n")
        print("\nAttention Probs:")
        print(f"Mean: {attention_probs.mean():.4f}")
        print(f"Std: {attention_probs.std():.4f}")
        print(f"Norm: {torch.norm(attention_probs).item():.4f}\n")

        # Apply attention to values
        attended_values = torch.matmul(attention_probs, V)
        
        # Restore spatial dimensions
        # [B, 3, H*W, C] -> [B, 3, H, W, C]
        output = attended_values.reshape(B, num_blocks, H, W, C)
        
        # Project to output space
        output = self.output_proj(output)

        print("\nOutput:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {attention_probs.std():.4f}")
        print("="*10)
        
        return output

    def forward(self, features):
        """
        Process all frames while maintaining temporal independence.
        
        Input shape: [B, 3, T, H, W, C] where T=8 is temporal dimension
        Output shape: Same as input, but with fused block information
        """
        B, num_blocks, T, H, W, C = features.shape
        output = []
        
        # Process each frame independently to maintain temporal structure
        for t in range(T):
            frame_output = self.process_single_frame(features[:, :, t])
            output.append(frame_output)
            
        # Stack outputs along temporal dimension
        output = torch.stack(output, dim=2)
        
        return output


class TemporalFusion:
    def __init__(self, feature_dim=1920, num_heads=8, window_size=3):
        """
        Initialize temporal fusion with sliding window attention.
        
        The window_size parameter determines how many frames are processed
        together. We use window_size=3 to capture local motion patterns
        while keeping computation manageable.
        """
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Attention components
        self.qkv_proj = nn.Linear(feature_dim, 3 * feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        self._init_weights()

    def create_window_mask(self, window_size, device):
        """
        Create an attention mask for the sliding window.
        
        This ensures each frame only attends to frames within its window,
        maintaining local temporal consistency.
        """
        mask = torch.zeros(window_size, window_size)
        for i in range(window_size):
            left = max(0, i - self.window_size // 2)
            right = min(window_size, i + self.window_size // 2 + 1)
            mask[i, left:right] = 1
        return mask.to(device)

    def process_window(self, window_features):
        """
        Process a single temporal window while maintaining spatial structure.
        
        Input shape: [B, W, H, W, C] where:
            - W is window_size
            - H, W are spatial dimensions
            - C is feature dimension
        """
        print("\n=== Temporal Block Fusion ===")
        B, W, H, W_spatial, C = window_features.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(window_features)
        print("\nWindow Projected:")
        print(f"Shape: {qkv.shape}")
        print(f"Mean: {qkv.mean():.4f}")
        print(f"Std: {qkv.std()():.4f}")
        print(f"Norm: {torch.norm(qkv).item():.4f}\n")

        qkv = qkv.reshape(B, W, H*W_spatial, 3, self.num_heads, self.head_dim)
        print("\nWindow Reshaped:")
        print(f"Shape: {qkv.shape}")
        print(f"Mean: {qkv.mean():.4f}")
        print(f"Std: {qkv.std()():.4f}")
        print(f"Norm: {torch.norm(qkv).item():.4f}\n")

        q, k, v = qkv.unbind(dim=3)
        
        # Compute attention with window mask
        mask = self.create_window_mask(W, window_features.device)

        # Scale dot product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print("\nAttention Scores:")
        print(f"Shape: {attention_scores.shape}")
        print(f"Mean: {attention_scores.mean():.4f}")
        print(f"Std: {attention_scores.std()():.4f}")
        print(f"Norm: {torch.norm(attention_scores).item():.4f}\n")

        attention_scores = attention_scores * mask.unsqueeze(0).unsqueeze(0)
        print("\nMasked Attention Scores:")
        print(f"Shape: {attention_scores.shape}")
        print(f"Mean: {attention_scores.mean():.4f}")
        print(f"Std: {attention_scores.std()():.4f}")
        print(f"Norm: {torch.norm(attention_scores).item():.4f}\n")

        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attention_probs, v)
        print("\nOutput:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std()():.4f}")
        print(f"Norm: {torch.norm(output).item():.4f}\n")
        
        # Restore spatial dimensions and project
        output = output.reshape(B, W, H, W_spatial, C)
        print("\nReshaped Output:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std()():.4f}")
        print(f"Norm: {torch.norm(output).item():.4f}\n")

        output = self.output_proj(output)
        print("\nProjected Output:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std()():.4f}")
        print(f"Norm: {torch.norm(output).item():.4f}\n")
        print("="*10)

        return output

    def forward(self, features):
        """
        Process entire sequence using sliding windows.
        
        Input shape: [B, T, H, W, C]
        Uses stride=1 to ensure smooth transitions between windows.
        """
        B, T, H, W, C = features.shape
        output = []
        
        # Create overlapping windows
        for t in range(0, T - self.window_size + 1):
            window = features[:, t:t+self.window_size]
            window_output = self.process_window(window)
            output.append(window_output[:, self.window_size//2])
        
        # Handle edge cases at sequence boundaries
        output = torch.stack(output, dim=1)
        
        return output