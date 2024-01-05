import math

class LambdaScheduler():
    """
    The model converges in 100 steps, so setting total_step based on
    epoch and batch_size is unhelpful.

    Use 150 instead.

    """
    def __init__(self, initial_rate, strategy, **kwargs) -> None:
        self.initial_rate = initial_rate
        self.strategy = strategy
        self.kwargs = kwargs

    def update(self, current_step, total_step):
        if self.strategy == "linear_decay":
            this_step_lambda_ = self.linear_decay(current_step, total_step)
            return this_step_lambda_
        elif self.strategy == "cosine_decay":
            return self.cosine_decay(current_step, total_step)
        elif self.strategy == "identify":
            return self.initial_rate

    def linear_decay(self, current_step, total_step):
        residual = self.kwargs.get("residual", 0)
        if current_step <= total_step:
            return ((self.initial_rate - residual) * (1 - current_step / total_step)) + residual
        else:
            return residual

    def cosine_decay(self, current_step, total_step):
        residual = self.kwargs.get("residual", 0)
        if current_step <= total_step:
            cosine_decay_value = 0.5 * (1 + math.cos(math.pi * current_step / total_step))
            return (self.initial_rate - residual) * cosine_decay_value + residual
        else:
            return residual


scheduler_8_linear_1 = LambdaScheduler(0.8, "linear_decay", residual=0.1)
scheduler_9_identify = LambdaScheduler(0.9, "identify", residual=0.1)
__all__ = ["scheduler_8_linear_1", "scheduler_9_identify"]