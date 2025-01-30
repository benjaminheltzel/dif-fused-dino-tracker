from torch.optim.lr_scheduler import LambdaLR


def get_cnn_refiner_scheduler(optimizer, gamma=0.999, apply_every=40):
    
    
    def decay_scheduler(epoch):
        return gamma ** (epoch // apply_every)
    
    schedulers = []

    for i in range(len(optimizer.param_groups)):
        # delta_dino and all fusion components use decayed learning rate
        if i in [0, 2, 3, 4, 5]:  
            schedulers.append(decay_scheduler)
        # tracker_head uses constant learning rate
        else:  
            schedulers.append(lambda epoch: 1)
    
    return LambdaLR(optimizer, lr_lambda=schedulers)
