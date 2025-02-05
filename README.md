# 🦖 Dif-fused DINO-tracker: Zero-shot Point Tracking Enhanced with Video Diffusion 

## Motivation: 
- Point tracking enables key computer vision tasks, but tracking through occlusions remains challenging
- DINO-tracker achieves strong spatial understanding via semantic priors but lacks inherent temporal reasoning
- CogVideoX learns rich temporal dynamics through its jointly-trained 3D VAE and full 3D attention 

## Contributions:
- First framework combining pre-trained video diffusion features with point tracking
- Transformer block fusion architecture combining multiple Cog VideoX features through normalized self-attention
- Cross-model fusion pipeline maintaining temporal consistency while projecting from CogVideoX to DINO feature space

## Use:
1. Extract CogVideoX diffusion features from video w/ notebook: diffusion_feature_extractor.ipynb
2. Save features to folder ./diffusion/<video-id>
3. Run point tracking model w/ notebook: dif-fused-dino-tracker.ipynb
4. Watch your colab credits get Thanos-snapped 🫡
