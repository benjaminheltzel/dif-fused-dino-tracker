from torch.optim.lr_scheduler import LambdaLR
import math

def get_cnn_refiner_scheduler(optimizer, gamma=0.999, apply_every=40, warmup_steps=1000):
    def get_warmup_schedule(warmup_factor):
        def schedule(step):
            if step < warmup_steps:
                return step / warmup_steps * warmup_factor
            decay_steps = (step - warmup_steps) // apply_every
            return gamma ** decay_steps
        return schedule
    
    # Different schedules for different components
    schedules = [
        # Delta-DINO: Standard warmup and decay
        get_warmup_schedule(1.0),
        
        # Tracker head: Constant learning rate
        lambda step: 1.0
    ]
    
    return LambdaLR(optimizer, lr_lambda=schedules)

def get_fusion_scheduler(optimizer, warmup_steps=1000, gamma=0.999):
    def get_fusion_schedule(warmup_factor, decay_factor):
        def schedule(step):
            if step < 0:
                raise ValueError(f"Invalid step value: {step}")
            if step < warmup_steps:
                # Add smoother warmup curve
                warmup_progress = step / warmup_steps
                return warmup_factor * (1 - math.cos(math.pi * warmup_progress)) / 2
            decay_steps = (step - warmup_steps) // 100
            return gamma ** (decay_steps * decay_factor)
        return schedule
    
    # Schedules for fusion components
    schedules = [
        # Pyramid projectors: careful warmup
        get_fusion_schedule(0.7, 0.5),
        # Pyramid weights: very careful warmup
        get_fusion_schedule(0.5, 0.3),
        # Pyramid combiner: moderate warmup
        get_fusion_schedule(0.8, 0.6),
        # Final fusion: careful warmup
        get_fusion_schedule(0.6, 0.4)
    ]
    
    return LambdaLR(optimizer, lr_lambda=schedules)