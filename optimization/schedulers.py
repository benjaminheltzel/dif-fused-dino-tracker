from torch.optim.lr_scheduler import LambdaLR

def get_cnn_refiner_scheduler(optimizer, gamma=0.999, apply_every=40, warmup_steps=1000):
    """
    Creates a learning rate scheduler that handles different components of the network:
    
    1. DINO feature refinement (delta_dino): Gradual decay after warmup
    2. Tracker head: Constant learning rate for stable point tracking
    3. Fusion components: Slower warmup and gentler decay to allow careful feature integration
    4. Lateral connections: Similar to fusion but with faster decay
    5. Final fusion layers: Careful warmup to prevent early overfitting
    
    Args:
        optimizer: The optimizer containing parameter groups for each component
        gamma: The decay factor for learning rates
        apply_every: Number of steps between learning rate updates
        warmup_steps: Number of steps for warmup period
    """
    def create_schedule(warmup_factor, decay_factor):
        def schedule(step):
            # Warmup phase
            if step < warmup_steps:
                return warmup_factor * (step / warmup_steps)
            # Decay phase    
            else:
                decay_power = ((step - warmup_steps) // apply_every)
                return (gamma ** (decay_power * decay_factor))
        return schedule

    # Create specific schedules for each component
    schedulers = []
    
    # Group 0: delta_dino (feature refinement)
    # Quick warmup but gentle decay to maintain DINO feature quality
    schedulers.append(create_schedule(warmup_factor=0.5, decay_factor=0.7))
    
    # Group 1: tracker_head
    # Constant learning rate for stable tracking behavior
    schedulers.append(lambda step: 1.0)
    
    # Group 2: block_projectors (CogVideo feature processing)
    # Slower warmup to carefully learn feature transformations
    schedulers.append(create_schedule(warmup_factor=0.3, decay_factor=0.8))
    
    # Group 3: lateral_convs (feature pyramid connections)
    # Medium warmup with standard decay
    schedulers.append(create_schedule(warmup_factor=0.4, decay_factor=1.0))
    
    # Group 4: pyramid_fusion
    # Careful warmup to establish good feature integration
    schedulers.append(create_schedule(warmup_factor=0.2, decay_factor=0.9))
    
    # Group 5: final_fusion
    # Very careful warmup with minimal decay to maintain fusion quality
    schedulers.append(create_schedule(warmup_factor=0.1, decay_factor=0.6))

    return LambdaLR(optimizer, lr_lambda=schedulers)