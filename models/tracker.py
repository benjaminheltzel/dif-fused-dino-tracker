import torch
import torch.nn as nn
from einops import rearrange
import os
import gc
from pathlib import Path
from models.networks.tracker_head import TrackerHead
from models.networks.delta_dino import DeltaDINO
from models.utils import load_pre_trained_model
from data.dataset import RangeNormalizer
from utils import bilinear_interpolate_video


EPS = 1e-08


class Tracker(nn.Module):
    def __init__(
        self,
        video=None,
        #video_id=None,
        ckpt_path="",
        dino_embed_path="",
        dino_patch_size=14,
        stride=7,
        device="cuda:0",
        
        cyc_n_frames=4,
        cyc_batch_size_per_frame=256,
        cyc_fg_points_ratio=0.7,
        cyc_thresh=4
        ):
        super().__init__()

        self.stride = stride
        self.dino_patch_size = dino_patch_size
        self.device = device
        self.refined_features = None
        self.dino_embed_path = dino_embed_path
        self.ckpt_path = ckpt_path
        self.cyc_n_frames = cyc_n_frames
        self.cyc_batch_size_per_frame = cyc_batch_size_per_frame
        self.cyc_fg_points_ratio = cyc_fg_points_ratio
        self.cyc_thresh = cyc_thresh
        
        self.video = video
        #self.video_id = video_id
        
        # DINO embed
        self.load_dino_embed_video()

        # Delta-DINO
        self.delta_dino = DeltaDINO(vit_stride=self.stride).to(device)

        # CNN-Refiner
        t, c, h, w = self.video.shape
        self.cmap_relu = nn.ReLU(inplace=True)
        self.tracker_head = TrackerHead(use_cnn_refiner=True,
                                        patch_size=dino_patch_size,
                                        step_h=stride,
                                        step_w=stride,
                                        video_h=h,
                                        video_w=w).to(device)
        self.range_normalizer = RangeNormalizer(shapes=(w, h, self.video.shape[0]))


        ################################################
        #           Diffusion Feature Fusion           #
        ################################################

        # Define transformer blocks for feature pyramid
        self.transformer_blocks = [0, 7, 14, 21, 29]

        # Load CogVideoX features
        self.cogvideo_features = torch.load(f"./diffusion/29/cogvideox_features.pt")
        self.cogvideo_spatial = {}
        assert all(f'block_{idx}_hidden' in self.cogvideo_features for idx in self.transformer_blocks), \
            "Not all required transformer blocks are present in features"

        # Verify and reshape features for each block
        for block_idx in self.transformer_blocks:
            block_features = self.cogvideo_features[f'block_{block_idx}_hidden'].float()
            assert block_features.shape == (1, 10800, 1920), f"Expected block {block_idx} features shape (1, 10800, 1920), got {block_features.shape}"
            
            # Reshape features for each block [1, 10800, 1920] -> [8, 1350, 1920]
            self.cogvideo_spatial[block_idx] = block_features[0].reshape(
                8, 10800 // 8, 1920
            ).to(self.device)

        # Calculate spatial dimensions
        # The 10800 dimension should be divided into 8 temporal groups since we have 8 compressed frames
        self.frames_per_group = 4  # Since 32 original frames compressed to 8 frames
        self.cogvideo_feat_len = 10800 // 8  # Features per frame group
        self.cogvideo_dim = 1920
        self.dino_dim = 1024

        # Individual normalization layers
        self.dino_norm = nn.LayerNorm(self.dino_dim)
        self.cogvideo_norms = nn.ModuleDict({
            f'block_{i}': nn.InstanceNorm1d(self.cogvideo_dim)
            for i in self.transformer_blocks
        })

        # Create feature pyramid components
        self.block_projectors = nn.ModuleDict({
            f'block_{i}': nn.Sequential(
                nn.LayerNorm(self.cogvideo_dim),
                nn.Linear(self.cogvideo_dim, self.dino_dim),
                nn.ReLU()
            ) for i in self.transformer_blocks
        }).to(device)
        # Feature pyramid normalization
        self.pyramid_norm = nn.LayerNorm(self.dino_dim)

        # Lateral connections for feature pyramid
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dino_dim, self.dino_dim, 1, bias=False),
                nn.BatchNorm2d(self.dino_dim)
            ) for _ in range(len(self.transformer_blocks) - 1) 
        ]).to(device)
        # Lateral normalization layers 
        self.lateral_norms = nn.ModuleList([
            nn.BatchNorm2d(self.dino_dim)
            for _ in range(len(self.transformer_blocks) - 1)
        ])

        # Two-stage fusion architecture
        # Stage 1: Pyramid feature fusion
        self.pyramid_fusion = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.dino_dim),
            nn.ReLU()
        ).to(self.device)    
        # Stage 2: Final DINO fusion
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(self.dino_dim * 2),  # For concatenated DINO and pyramid features
            nn.Linear(self.dino_dim * 2, self.dino_dim),
            nn.ReLU()
        ).to(self.device)
        # Final normalization layer
        self.final_norm = nn.LayerNorm(self.dino_dim)

        # Gate layers for DINO and CogVideoX features
        self.dino_gate = nn.Linear(self.dino_dim, 1)
        self.cogvideo_gate = nn.Linear(self.dino_dim, 1)

        # Initialize gates to favor DINO features initially
        with torch.no_grad():
            self.dino_gate.bias.fill_(2.0)  # Initially biased towards DINO
            self.cogvideo_gate.bias.fill_(0.0)


    @torch.no_grad()
    def load_dino_embed_video(self):
        """
        video: T x 3 x H' x W'
        self.dino_embed_video: T x C x H x W
        """
        assert os.path.exists(self.dino_embed_path)
        self.dino_embed_video = torch.load(self.dino_embed_path, map_location=self.device)
  
    def get_dino_embed_video(self, frames_set_t):
        dino_emb = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)] if frames_set_t.device != self.dino_embed_video.device else self.dino_embed_video[frames_set_t]
        return dino_emb
    
    def normalize_points_for_sampling(self, points):
        t, c, vid_h, vid_w = self.video.shape
        h = vid_h
        w = vid_w
        patch_size = self.dino_patch_size
        stride = self.stride
        
        last_coord_h =( (h - patch_size) // stride ) * stride + (patch_size / 2)
        last_coord_w =( (w - patch_size) // stride ) * stride + (patch_size / 2)
        ah = 2 / (last_coord_h - (patch_size / 2))
        aw = 2 / (last_coord_w - (patch_size / 2))
        bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
        bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
        
        a = torch.tensor([[aw, ah, 1]]).to(self.device)
        b = torch.tensor([[bw, bh, 0]]).to(self.device)
        normalized_points = a * points + b
        return normalized_points
    
    def sample_embeddings(self, embeddings, source_points):
        """embeddings: T x C x H x W. source_points: B x 3, where the last dimension is (x, y, t), x and y are in [-1, 1]"""
        t, c, h, w = embeddings.shape
        sampled_embeddings = bilinear_interpolate_video(video=rearrange(embeddings, "t c h w -> 1 c t h w"),
                                                               points=source_points,
                                                               h=h,
                                                               w=w,
                                                               t=t,
                                                               normalize_w=False,
                                                               normalize_h=False,
                                                               normalize_t=True)
        sampled_embeddings = sampled_embeddings.squeeze()
        if len(sampled_embeddings.shape) == 1:
            sampled_embeddings = sampled_embeddings.unsqueeze(1)
        sampled_embeddings = sampled_embeddings.permute(1,0)
        return sampled_embeddings

    def get_refined_embeddings(self, frames_set_t, return_raw_embeddings=False, print_diagnostics=False):
        # Get DINO embeddings and residuals
        frames_dino_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        refiner_input_frames = self.video[frames_set_t]

        # Print initial DINO feature statistics
        if print_diagnostics:
            print("\nInitial DINO Features:")
            print(f"Mean: {frames_dino_embeddings.mean().item():.4f}")
            print(f"Std: {frames_dino_embeddings.std().item():.4f}")
            print(f"Norm: {torch.norm(frames_dino_embeddings).item():.4f}")

        # Compute residuals in batches
        batch_size = 8
        n_frames = frames_set_t.shape[0]
        residual_embeddings = torch.zeros_like(frames_dino_embeddings)
        for i in range(0, n_frames, batch_size):
            end_idx = min(i+batch_size, n_frames)
            residual_embeddings[i:end_idx] = self.delta_dino(
                refiner_input_frames[i:end_idx], 
                frames_dino_embeddings[i:end_idx]
            )

        refined_dino = frames_dino_embeddings + residual_embeddings
        refined_dino = self.dino_norm(refined_dino.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Print refined DINO statistics before fusion
        if print_diagnostics:
            print("\nRefined DINO Features (before fusion):")
            print(f"Mean: {refined_dino.mean().item():.4f}")
            print(f"Std: {refined_dino.std().item():.4f}")
            print(f"Norm: {torch.norm(refined_dino).item():.4f}")

        B, C, H, W = refined_dino.shape


        ################################################
        #       Diffusion Feature Fusion - Start       #
        ################################################

        # Get frame indices for CogVideoX features & verify bounds
        cogvideo_indices = frames_set_t // self.frames_per_group
        assert torch.all(cogvideo_indices >= 0) and torch.all(cogvideo_indices < 8), \
            f"Invalid temporal indices: {cogvideo_indices.min()}-{cogvideo_indices.max()}"
        
        ################################################
        # Stage 1: Build CogVideoX Feature Pyramid
        pyramid_features = []
        for block_idx in self.transformer_blocks:
            # Get block features - shape: [n_frames, 1350, 1920]
            block_feat = self.cogvideo_spatial[block_idx][cogvideo_indices]
            block_feat = self.cogvideo_norms[f'block_{block_idx}'](block_feat.unsqueeze(1)).squeeze(1)

            if print_diagnostics:
                print(f"\nCogVideoX Block {block_idx} Features:")
                print(f"Mean: {block_feat.mean().item():.4f}")
                print(f"Std: {block_feat.std().item():.4f}")
                print(f"Norm: {torch.norm(block_feat).item():.4f}")

            # Reshape to [n_frames, 1920, 1350, 1] for interpolation
            block_feat = block_feat.permute(0, 2, 1).unsqueeze(-1)
            
            # Interpolate to match spatial dimensions [n_frames, 1920, H*W, 1]
            block_feat = torch.nn.functional.interpolate(
                block_feat,
                size=(H * W, 1),
                mode='bilinear',
                align_corners=True
            )

            # Reshape to [n_frames, H*W, 1920] then to [B, H, W, cogvideo_dim]
            block_feat = block_feat.squeeze(-1).permute(0, 2, 1)
            block_feat = block_feat.reshape(B, H, W, self.cogvideo_dim)

            # Flatten for projection to DINO dimension
            block_feat_flat = block_feat.reshape(-1, self.cogvideo_dim)
            block_feat_projected = self.block_projectors[f'block_{block_idx}'](block_feat_flat)
            block_feat_spatial = block_feat_projected.reshape(B, H, W, self.dino_dim)

            pyramid_features.append(block_feat_spatial)


        # Apply lateral connections from top to bottom
        for i in range(len(pyramid_features)-1, 0, -1):
            # Convert to channel-first format for conv [B, C, H, W]
            top_feat = pyramid_features[i].permute(0, 3, 1, 2)
            lateral_feat = pyramid_features[i-1].permute(0, 3, 1, 2)

            # Apply lateral connection
            lateral_conv = self.lateral_convs[i-1](top_feat)

            # Normalize and fuse
            fused = self.lateral_norms[i-1](lateral_feat + lateral_conv)

            # Convert back to channel-last format [B, H, W, C]
            pyramid_features[i-1] = fused.permute(0, 2, 3, 1)


        # Get final pyramid features
        final_pyramid = pyramid_features[0]

        # Flatten for pyramid fusion
        pyramid_flat = final_pyramid.reshape(-1, self.dino_dim)

        # Apply pyramid fusion
        pyramid_features = self.pyramid_fusion(pyramid_flat)

        pyramid_features = self.pyramid_norm(pyramid_features)

        # Reshape to DINO format: [B, H, W, dino_dim]
        pyramid_features = pyramid_features.reshape(B, H, W, self.dino_dim)

        ################################################
        # Final fusion
        # Convert DINO features to channel-last format [B, H, W, C]
        refined_spatial = refined_dino.permute(0, 2, 3, 1)

        # Flatten both feature sets for fusion [B*H*W, dino_dim]
        refined_flat = refined_spatial.reshape(-1, self.dino_dim)
        pyramid_flat = pyramid_features.reshape(-1, self.dino_dim)

        # Attention-based fusion
        dino_gate = torch.sigmoid(self.dino_gate(refined_flat))
        cogvideo_gate = torch.sigmoid(self.cogvideo_gate(pyramid_flat))

        # Apply gates to features
        combined = torch.cat([
            dino_gate * refined_flat,
            cogvideo_gate * pyramid_flat
        ], dim=-1)

        # Fuse features
        fused_features = self.final_fusion(combined)
        fused_features = self.final_norm(fused_features)

        if print_diagnostics:
            print("\nFinal Fused Features:")
            print(f"Mean: {fused_features.mean().item():.4f}")
            print(f"Std: {fused_features.std().item():.4f}")
            print(f"Norm: {torch.norm(fused_features).item():.4f}")
            print(f"DINO Gate Mean: {dino_gate.mean().item():.4f}")
            print(f"CogVideo Gate Mean: {cogvideo_gate.mean().item():.4f}")
            print("-" * 50)

        # Reshape back to DINO format
        fused_features = fused_features.reshape(B, H, W, self.dino_dim)
        fused_features = fused_features.permute(0, 3, 1, 2)

        if return_raw_embeddings:
            return fused_features, residual_embeddings, frames_dino_embeddings
        return fused_features, residual_embeddings

        ################################################
        #        Diffusion Feature Fusion - End        #
        ################################################


    def cache_refined_embeddings(self, move_dino_to_cpu=False):
        refined_features, _ = self.get_refined_embeddings(torch.arange(0, self.video.shape[0]))
        self.refined_features = refined_features
        if move_dino_to_cpu:
            self.dino_embed_video = self.dino_embed_video.to("cpu")
    
    def uncache_refined_embeddings(self, move_dino_to_gpu=False):
        self.refined_features = None
        torch.cuda.empty_cache()
        gc.collect()
        if move_dino_to_gpu:
            self.dino_embed_video = self.dino_embed_video.to("cuda")
    
    def save_weights(self, iter):
        torch.save(self.tracker_head.state_dict(), Path(self.ckpt_path) / f"tracker_head_{iter}.pt")
        torch.save(self.delta_dino.state_dict(), Path(self.ckpt_path) / f"delta_dino_{iter}.pt")
        
    def load_weights(self, iter):
        self.tracker_head = load_pre_trained_model(
            torch.load(os.path.join(self.ckpt_path, f"tracker_head_{iter}.pt")),
            self.tracker_head
        )
        self.delta_dino = load_pre_trained_model(
            torch.load(os.path.join(self.ckpt_path, f"delta_dino_{iter}.pt")),
            self.delta_dino
        )
    
    def get_corr_maps_for_frame_set(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps_set = torch.einsum("bc,nchw->bnhw", source_embeddings, frame_embeddings_set)
        corr_maps = corr_maps_set[torch.arange(source_embeddings.shape[0]), target_frame_indices.int(), :, :]
        
        embeddings_norm = frame_embeddings_set.norm(dim=1)
        target_embeddings_norm = embeddings_norm[target_frame_indices.int()]
        source_embeddings_norm = source_embeddings.norm(dim=1).unsqueeze(-1).unsqueeze(-1)
        corr_maps_norm = (source_embeddings_norm * target_embeddings_norm)
        corr_maps = corr_maps / torch.clamp(corr_maps_norm, min=EPS)
        corr_maps = rearrange(corr_maps, "b h w -> b 1 h w")
        
        return corr_maps
    
    def get_point_predictions_from_embeddings(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps = self.get_corr_maps_for_frame_set(source_embeddings, frame_embeddings_set, target_frame_indices)
        coords = self.tracker_head(self.cmap_relu(corr_maps))
        return coords
    
    def get_point_predictions(self, inp, frame_embeddings):
        source_points_unnormalized, source_frame_indices, target_frame_indices, _ = inp
        source_points = self.normalize_points_for_sampling(source_points_unnormalized)
        source_embeddings = self.sample_embeddings(frame_embeddings, torch.cat([ source_points[:, :-1], source_frame_indices[:, None] ], dim=1)) # B x C
        return self.get_point_predictions_from_embeddings(source_embeddings, frame_embeddings, target_frame_indices)
    
    @torch.no_grad()
    def get_cycle_consistent_coords(self, frames_set_t, fg_masks):
        source_selector = torch.randint(frames_set_t.shape[0], (self.cyc_n_frames,), device=frames_set_t.device)
        target_selector = torch.randint(frames_set_t.shape[0], (self.cyc_n_frames,), device=frames_set_t.device)
        
        # create 2D meshgrid for size fg_masks and join them to a single tensor of coordinates [x,y]
        h, w = fg_masks.shape[-2:]
        x = torch.arange(w, device=fg_masks.device).float()
        y = torch.arange(h, device=fg_masks.device).float()
        yy, xx = torch.meshgrid(y, x)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        grid_coords = torch.stack([xx, yy], dim=-1)
        
        source_points = []
        target_points = []
        cycle_points = []
        cyc_source_frame_indices = []
        cyc_target_frame_indices = []
        source_times = []
        target_times = []
        
        BATCH_SIZE_PER_FRAME = self.cyc_batch_size_per_frame
        BATCH_SIZE_FG = int(BATCH_SIZE_PER_FRAME * self.cyc_fg_points_ratio)
        BATCH_SIZE_BG = BATCH_SIZE_PER_FRAME - BATCH_SIZE_FG
        
        for source_idx, target_idx in zip(source_selector, target_selector):
            source_t = frames_set_t[source_idx]
            target_t = frames_set_t[target_idx]

            frame_fg_mask = fg_masks[source_t] > 0
            frame_bg_mask = ~frame_fg_mask
            frame_coords_fg = grid_coords[frame_fg_mask.reshape(-1)]
            frame_coords_fg = frame_coords_fg[torch.randperm(frame_coords_fg.shape[0])[:BATCH_SIZE_FG]]
            frame_coords_bg = grid_coords[frame_bg_mask.reshape(-1)]
            frame_coords_bg = frame_coords_bg[torch.randperm(frame_coords_bg.shape[0])[:BATCH_SIZE_BG]]
            frame_coords = torch.cat([frame_coords_fg, frame_coords_bg], dim=0)
            
            frame_coords = torch.cat([frame_coords, torch.ones((frame_coords.shape[0], 1), device=frame_coords.device)*source_t], dim=-1)
            
            source_frame_indices = torch.tensor([source_idx]*frame_coords.shape[0], device=frames_set_t.device)
            target_frame_indices = torch.tensor([target_idx]*frame_coords.shape[0], device=frames_set_t.device)
            inp = frame_coords, source_frame_indices, target_frame_indices, frames_set_t
            
            # cycle-consistency filtering
            with torch.no_grad():
                target_coords = self.get_point_predictions(inp, self.frame_embeddings)                
                target_coords = self.range_normalizer.unnormalize(target_coords, src=(-1, 1), dims=[0, 1])
                target_coords = torch.cat([target_coords, torch.ones((target_coords.shape[0], 1), device=target_coords.device)*target_t], dim=-1)
                source_frame_indices = torch.tensor([target_idx]*target_coords.shape[0], device=frames_set_t.device)
                target_frame_indices = torch.tensor([source_idx]*target_coords.shape[0], device=frames_set_t.device)
                inp = target_coords, source_frame_indices, target_frame_indices, frames_set_t
                
                coords = self.get_point_predictions(inp, self.frame_embeddings)
                
                coords = self.range_normalizer.unnormalize(coords, src=(-1, 1), dims=[0, 1])
            filtered_source_indices = torch.norm(frame_coords[:, :2] - coords[:, :2], dim=1) <= self.cyc_thresh
            filtered_source_coords = frame_coords[filtered_source_indices]
            filtered_target_coords = target_coords[filtered_source_indices]
            filtered_cycle_coords = coords[filtered_source_indices]
            
            source_points.append(filtered_source_coords)
            target_points.append(filtered_target_coords)
            cycle_points.append(filtered_cycle_coords)
            cyc_source_frame_indices.append(torch.tensor([source_idx]*filtered_source_coords.shape[0], device=frames_set_t.device))
            cyc_target_frame_indices.append(torch.tensor([target_idx]*filtered_source_coords.shape[0], device=frames_set_t.device))  
            source_times.append(torch.tensor([source_t]*filtered_source_coords.shape[0], device=frames_set_t.device))
            target_times.append(torch.tensor([target_t]*filtered_source_coords.shape[0], device=frames_set_t.device))
        
        source_points = torch.cat(source_points, dim=0)
        target_points = torch.cat(target_points, dim=0)
        cycle_points = torch.cat(cycle_points, dim=0)
        cyc_source_frame_indices = torch.cat(cyc_source_frame_indices, dim=0)
        cyc_target_frame_indices = torch.cat(cyc_target_frame_indices, dim=0)
        source_times_normalized = self.range_normalizer(torch.cat(source_times, dim=0).unsqueeze(1).repeat(1, 3).float(), dst=(-1, 1), dims=[2])[:, 2]
        target_times_normalized = self.range_normalizer(torch.cat(target_times, dim=0).unsqueeze(1).repeat(1, 3).float(), dst=(-1, 1), dims=[2])[:, 2]
        
        return {
            "source_points": source_points,
            "target_points": target_points,
            "cycle_points": cycle_points,
            "source_frame_indices": cyc_source_frame_indices,
            "target_frame_indices": cyc_target_frame_indices,
            "source_times_normalized": source_times_normalized,
            "target_times_normalized": target_times_normalized,
        }
        
    def get_cycle_consistent_preds(self, frames_set_t, fg_masks):
        found_cycle_consistency_coords = False
        while not found_cycle_consistency_coords:
            cycle_consistency_coords =\
                self.get_cycle_consistent_coords(frames_set_t, fg_masks)
            found_cycle_consistency_coords = cycle_consistency_coords["source_points"].shape[0] > 0
        
        source_target_input = (cycle_consistency_coords["source_points"],
                                cycle_consistency_coords["source_frame_indices"],
                                cycle_consistency_coords["target_frame_indices"],
                                frames_set_t)
        target_source_input = (cycle_consistency_coords["target_points"],
                                cycle_consistency_coords["target_frame_indices"],
                                cycle_consistency_coords["source_frame_indices"],
                                frames_set_t)
        source_target_coords = self.get_point_predictions(source_target_input, self.frame_embeddings)
        target_source_coords = self.get_point_predictions(target_source_input, self.frame_embeddings)
        cycle_consistency_dists = torch.norm(cycle_consistency_coords["cycle_points"][:, :2] - cycle_consistency_coords["source_points"][:, :2], dim=1)
        
        cycle_source_points_normalized =\
            self.range_normalizer(cycle_consistency_coords["source_points"], dst=[-1, 1])
        cycle_target_points_normalized =\
            self.range_normalizer(cycle_consistency_coords["target_points"], dst=[-1, 1])
        cycle_consistency_preds = {
            "source_coords": cycle_source_points_normalized,
            "target_coords": cycle_target_points_normalized,
            "source_target_coords": source_target_coords[:, :2],
            "target_source_coords": target_source_coords[:, :2],
            "cycle_consistency_dists": cycle_consistency_dists,
            "cycle_points": cycle_consistency_coords["cycle_points"]
        }

        return cycle_consistency_preds

    def forward(self, inp, print_diagnostics=False, use_raw_features=False):
        """
        inp: source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t; where
        source_points_unnormalized: B x 3. ((x, y, t) in image scale - NOT normalized)
        source_frame_indices: the indices of frames of source points in frames_set_t
        target_frame_indices: the indices of target frames in frames_set_t
        frames_set_t: N, 0 to T-1 (NOT normalized)
        print_diagnostics: whether to print feature statistics and gradients
        """
        frames_set_t = inp[-1]

        if use_raw_features:
            frame_embeddings = raw_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        elif self.refined_features is not None: # load from cache
            frame_embeddings = self.refined_features[frames_set_t]
            raw_embeddings = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)]
        else:
            # Pass through the print_diagnostics flag to get_refined_embeddings
            frame_embeddings, residual_embeddings, raw_embeddings = self.get_refined_embeddings(
                frames_set_t, 
                return_raw_embeddings=True,
                print_diagnostics=print_diagnostics
            )
            self.residual_embeddings = residual_embeddings
        self.frame_embeddings = frame_embeddings
        self.raw_embeddings = raw_embeddings
        coords = self.get_point_predictions(inp, frame_embeddings)

        return coords
