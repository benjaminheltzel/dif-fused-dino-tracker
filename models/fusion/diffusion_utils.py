import torch

class ReshapeCogVideoFeatures:
    def __init__(self):
        # Define constants that describe our feature dimensions
        self.frames = 8  # Number of CogVideoX frames
        self.height = 30  # Height after patchification 
        self.width = 45  # Width after patchification
        self.cogvideo_dim = 1920  # Feature dimension
        
        # Validate our understanding of the total token count
        self.spatial_tokens = self.height * self.width  # Should be 1350
        self.total_tokens = self.frames * self.spatial_tokens  # Should be 10800

    def validate_input(self, features):
        """
        Before processing, we must ensure our input matches expected dimensions.
        The 10800 dimension represents 8 frames x (30x45) spatial tokens.
        """
        if len(features.shape) != 3:
            raise ValueError(f"Expected 3D tensor [B, 10800, 1920], got shape {features.shape}")
        
        B, tokens, C = features.shape
        if tokens != self.total_tokens:
            raise ValueError(
                f"Expected {self.total_tokens} tokens (8 frames x 1350 spatial tokens), "
                f"got {tokens}"
            )
        if C != self.cogvideo_dim:
            raise ValueError(f"Expected {self.cogvideo_dim} channels, got {C}")

    def reshape_features(self, features):
        """
        Transform CogVideoX features while preserving their spatial-temporal structure.
        
        Input shape: [B, 10800, 1920] where:
            - B is batch size (typically 1)
            - 10800 is 8 frames x (30x45) spatial tokens
            - 1920 is the CogVideoX feature dimension
            
        Output shape: [B, 8, 30, 45, 1920] where:
            - 8 is the temporal dimension
            - 30x45 preserves spatial relationships
            - 1920 remains unchanged
        """
        print("\n=== Reshaping CogVideo Features ===")
        self.validate_input(features)
        print(f"Input shape: {features.shape}")

        B = features.shape[0]

        # Step 1: Separate temporal dimension
        # [B, 10800, 1920] -> [B, 8, 1350, 1920]
        features = features.reshape(B, self.frames, self.spatial_tokens, self.cogvideo_dim)
        print(f"After temporal split: {features.shape}")

        # Step 2: Unfold spatial dimensions
        # [B, 8, 1350, 1920] -> [B, 8, 30, 45, 1920]
        features = features.reshape(B, self.frames, self.height, self.width, self.cogvideo_dim)
        print(f"After spatial unfolding: {features.shape}")
        print("="*10)

        # Ensure memory layout is contiguous for efficient operations
        return features.contiguous()

    def __call__(self, features):
        return self.reshape_features(features)


class TemporalInterpolator:
    def __init__(self):
        # Define temporal constants
        self.dino_frames = 32  # Number of DINO frames
        self.cogvideo_frames = 8  # Number of CogVideoX frames
        self.microticks_per_macrotick = 4  # Relationship between DINO and CogVideoX frames
        
        # Pre-compute the interpolation weights matrix
        self.weights = self._compute_interpolation_weights()
    
    def _compute_interpolation_weights(self):
        """
        Compute the interpolation weights between DINO and CogVideoX frames.
        
        For each DINO frame i, we compute weights for CogVideoX frames that overlap
        with its 4-microtick window [i, i+3].
        
        Returns:
            weights: Tensor of shape [32, 8] containing interpolation weights
                    Each row sums to 1.0 and represents weights for one DINO frame
        """
        weights = torch.zeros(self.dino_frames, self.cogvideo_frames)
        
        for dino_frame in range(self.dino_frames):
            # Calculate the microtick window for this DINO frame
            window_start = dino_frame
            window_end = dino_frame + 3  # 4-microtick window
            
            # Convert to CogVideoX frame indices
            cogvideo_start = window_start // self.microticks_per_macrotick
            cogvideo_end = window_end // self.microticks_per_macrotick
            
            # Handle boundary cases
            cogvideo_start = min(cogvideo_start, self.cogvideo_frames - 1)
            cogvideo_end = min(cogvideo_end, self.cogvideo_frames - 1)
            
            if cogvideo_start == cogvideo_end:
                # DINO frame falls entirely within one CogVideoX frame
                weights[dino_frame, cogvideo_start] = 1.0
            else:
                # Calculate overlap proportions
                start_overlap = (
                    self.microticks_per_macrotick - 
                    (window_start % self.microticks_per_macrotick)
                ) / self.microticks_per_macrotick
                
                end_overlap = (
                    (window_end % self.microticks_per_macrotick) + 1
                ) / self.microticks_per_macrotick
                
                # Assign weights based on overlap
                weights[dino_frame, cogvideo_start] = start_overlap
                weights[dino_frame, cogvideo_end] = end_overlap
        
        # Normalize weights to sum to 1.0 for each DINO frame
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights
    
    def get_interpolation_weights(self, dino_frame_idx):
        """
        Get interpolation weights for a specific DINO frame.
        
        Args:
            dino_frame_idx: Index of the DINO frame (0-31)
            
        Returns:
            weights: Tensor of shape [8] containing weights for each CogVideoX frame
        """
        print(f"\n=== Temporal Interpolation for Frame {dino_frame_idx} ===")
        
        weights = self.weights[dino_frame_idx]
        print(f"Weight distribution: {weights.numpy()}")
        print(f"Sum: {weights.sum():.4f}")
        print("="*10)

        return weights