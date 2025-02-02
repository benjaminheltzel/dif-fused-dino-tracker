import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.diffusion_utils import ReshapeCogVideoFeatures, TemporalInterpolator
from models.fusion.dif_fusion import SpatialBlockFusion, TemporalFusion
from models.fusion.cross_modal_fusion import CrossModalFusion


class MainFusion(nn.Module):
    def __init__(
        self,
        cogvideo_dim=1920,
        dino_dim=1024,
        num_heads=8,
        dropout=0.1
    ):
        """
        Main fusion module that combines CogVideoX and DINO features.
        
        This class manages the complete fusion pipeline:
        1. Processes CogVideoX features from multiple blocks
        2. Fuses them spatially and temporally
        3. Aligns them with DINO features using 4-microtick system
        4. Produces final DINO-compatible features
        """
        super().__init__()
        
        # Store configuration
        self.cogvideo_dim = cogvideo_dim
        self.dino_dim = dino_dim
        self.num_heads = num_heads
        
        # Initialize components
        # Feature processing
        self.cogvideo_reshaper = ReshapeCogVideoFeatures()
        self.temporal_interpolator = TemporalInterpolator()
        
        # Attention mechanisms
        self.spatial_fusion = SpatialBlockFusion(
            feature_dim=cogvideo_dim,
            num_heads=num_heads
        )
        self.temporal_fusion = TemporalFusion(
            feature_dim=cogvideo_dim,
            num_heads=num_heads
        )
        
        # Dimension reduction for CogVideoX features
        self.cogvideo_projection = self._build_projection_mlp()
        
        # Cross-attention for feature fusion
        self.cross_attention = CrossModalFusion(
            cogvideo_dim=dino_dim,  # After projection
            dino_dim=dino_dim,
            num_heads=num_heads
        )
        
        # Final feature refinement
        self.final_norm = nn.LayerNorm(dino_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_projection_mlp(self):
        """
        Creates the MLP that projects CogVideoX features to DINO dimension.

        - Layer normalization for stable training
        - Linear projection with GELU activation
        - Final layer norm for output stability
        """
        return nn.Sequential(
            nn.LayerNorm(self.cogvideo_dim),
            nn.Linear(self.cogvideo_dim, self.dino_dim),
            nn.GELU(),
            nn.LayerNorm(self.dino_dim)
        )

    def forward(self, cogvideo_features, dino_features):
        """
        Full forward pass of the fusion pipeline.
        
        Args:
            cogvideo_features: Dict containing features from blocks 0, 14, 29
                             Each with shape [1, 10800, 1920]
            dino_features: DINO features with shape [B, 32, dino_dim, H, W]
            
        Returns:
            Fused features in DINO's format: [B, 32, dino_dim, H, W]
        """
        print("\n=== MainFusion Forward Pass ===")
        print(f"DINO Features: 
              Shape={dino_features.shape}, 
              Mean={dino_features.mean():.4f}, 
              Std={dino_features.std():.4f}, 
              Norm: {torch.norm(block_features).item():.4f}")

        # 1. Process CogVideoX features from each block
        processed_blocks = {}
        for block_idx in [0, 14, 29]:
            block_features = cogvideo_features[f'block_{block_idx}_hidden']
            print(f"\nBlock {block_idx} Features:")
            print(f"Shape: {block_features.shape}")
            print(f"Mean: {block_features.mean():.4f}")
            print(f"Std: {block_features.std():.4f}")
            print(f"Norm: {torch.norm(block_features).item():.4f}")
            processed_blocks[block_idx] = self.cogvideo_reshaper(block_features)
        
        # Stack processed blocks: [B, 3, 8, 30, 45, 1920]
        stacked_features = torch.stack(
            [processed_blocks[idx] for idx in [0, 14, 29]], 
            dim=1
        )
        print("\nStacked Features:")
        print(f"Shape: {stacked_features.shape}")
        print(f"Mean: {stacked_features.mean():.4f}")
        print(f"Std: {stacked_features.std():.4f}")
        print(f"Norm: {torch.norm(stacked_features).item():.4f}")
        
        # 2. Apply spatial and temporal fusion
        block_fused = self.spatial_fusion(stacked_features)
        print("\nAfter Spatial Fusion:")
        print(f"Shape: {block_fused.shape}")
        print(f"Mean: {block_fused.mean():.4f}")
        print(f"Std: {block_fused.std():.4f}")
        print(f"Norm: {torch.norm(block_fused).item():.4f}")

        temporal_fused = self.temporal_fusion(block_fused)
        print("\nAfter Temporal Fusion:")
        print(f"Shape: {temporal_fused.shape}")
        print(f"Mean: {temporal_fused.mean():.4f}")
        print(f"Std: {temporal_fused.std():.4f}")
        print(f"Norm: {torch.norm(temporal_fused).item():.4f}")
        
        # 3. Project to DINO dimension
        projected_features = self.cogvideo_projection(temporal_fused)
        print("\nAfter Projection:")
        print(f"Shape: {projected_features.shape}")
        print(f"Mean: {projected_features.mean():.4f}")
        print(f"Std: {projected_features.std():.4f}")
        print(f"Norm: {torch.norm(projected_features).item():.4f}")
        
        # 4. Prepare for cross-modal fusion
        B, T_dino, C, H, W = dino_features.shape
        assert C == self.dino_dim, f"Expected DINO dim {self.dino_dim}, got {C}"
        fused_features = torch.zeros_like(dino_features)
        
        # 5. Apply cross-modal fusion with temporal interpolation
        print("\nCross-Modal Fusion:")
        for t in range(T_dino):
            # Get interpolation weights for current DINO frame
            weights = self.temporal_interpolator.get_interpolation_weights(t)
            if t < 3:  # Print first few frames as example
                print(f"\nFrame {t} temporal weights:")
                print(f"Min: {weights.min():.4f}")
                print(f"Max: {weights.max():.4f}")
                print(f"Mean: {weights.mean():.4f}")
                print(f"Std: {weights.std():.4f}")

            # Apply fusion with temporal weighting
            frame_fused = self.cross_attention(
                dino_query=dino_features[:, t],
                cogvideo_keys=projected_features,
                temporal_weights=weights
            )
            
            fused_features[:, t] = frame_fused
        
        # 6. Final refinement
        output = self.final_norm(fused_features)
        output = self.dropout(output)
        
        print("\nFinal Output:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std():.4f}")
        print(f"Norm: {torch.norm(output).item():.4f}")
        print("="*10)

        return output