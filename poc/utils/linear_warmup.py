def linear_warmup(current_step, warmup_steps, base_lr, init_lr=1e-4):
    if current_step < warmup_steps:
        return init_lr + (base_lr - init_lr) * (current_step / warmup_steps)
    return base_lr