############
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L75
############

from typing import Iterable, Union
import contextlib

import torch


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext

        for s_param, param in zip(self.shadow_params, parameters):
            with context_manager():
                if param.requires_grad:
                    s_param.sub_(one_minus_decay * (s_param - param))
                else:
                    s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None