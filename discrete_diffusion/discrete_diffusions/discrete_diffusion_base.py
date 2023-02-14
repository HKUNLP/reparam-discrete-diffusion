import math
import numpy as np
import torch
import torch.nn as nn
from discrete_diffusions.utils import (
    topk_masking
)

class DiscreteDiffusion(nn.Module):
    """
    The parent class for discrete denoising diffusion probabilistic models.

    It should support the following methods:
    - q_sample()
        Sample x_t ~ q(x_t | x_0) to construct noisy Transformer inputs.
    - compute_losses()
        Compute the loss L_t = KL(q||p) at t-th time step.
    - sample_step()
        Sample x_t ~ p(x_{t-1} | x_t, x_0) at t-th time step.
    """
    
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps

    def q_sample(self, x_0, t, **kwargs):
        """

        Sample from q(x_t | x_0), which is used as the model inputs.

        Args:
            x_0: token ids with shape [B, N]
            t: current time step, tensor with shape [B]

        Returns:
            would possibly return a dict of relevant outputs, including x_t.
            
        """
        raise NotImplementedError

    def compute_losses(self, inputs, **kwargs):
        """
        
        Compute the loss objective KL(q||p) to train our generative process.

        Args:
            inputs: a dict that contains input types specific to different diffusion processes

        Returns:
            would possibly return a dict of relevant outputs, including loss used for training.
            
        """
        raise NotImplementedError

    def sample_step(self, decoder_out, denoising_fn, **kwargs):
        """
        Given a time step t, start from x_t and sample x_{t-k} from q(x_{t-k} | x_t).
        
        Args:
            decoder_out: a namedtuple that contains decoding info, including
                - x_t: token ids with shape [B, N]
                - t: scalar timesteps
                - max_steps: the maximum number of decoding steps
                - ...
            
            denoising_fn: a function that takes in x_t and t and returns model logits

            kwargs: other arguments that are used to control decoding.
        
        Returns:
            return a new decoder_out namedtuple.
        """
        raise NotImplementedError

    def _get_batched_decoding_strategy(self, t, max_step):
        """
            This function is used to compute the step size and the number of steps
            when t and max_step are both tensors with shape [B].
            It would be invoked in the case of adaptive decoding, where the sentences
            within the batch will be generated with different time steps.
        """
        _step_size = torch.div(self.num_timesteps, max_step, rounding_mode="floor")
        b = self.num_timesteps - _step_size * max_step
        step_size = torch.where(t < b, _step_size + 1, _step_size)
        step = torch.where(t < b, self.num_timesteps - t * step_size, self.num_timesteps - b - t * step_size)
        return step.long(), step_size.long()

    def _get_decoding_strategy(self, t, decoding_strategy, max_step):
        """
            This function is used to compute the step size and the number of steps
            when max_step is a scalar.
        """
        assert hasattr(self, "num_timesteps"), "num_timesteps is not set."
        if getattr(self, 'step_array', None) is None:
            if decoding_strategy == "spline":
                B = math.floor(max_step / 3)
                C = (self.num_timesteps - B) / (max_step - B)**2
                linear, quad = np.arange(start=0, stop=B + 1), np.arange(start=1, stop=max_step - B + 1)
                self.step_array = np.concatenate([linear, np.floor(B + C * quad** 2)])[::-1].astype(int)
            else:
                # t passed here ranges from 0 to max_step - 1.
                if max_step > self.num_timesteps:
                    raise NotImplementedError(
                        "will run {} empty steps. "
                        "Consider increase diffusion time-steps or decrease max_step."
                        .format(max_step - self.num_timesteps)
                        )
                elif self.num_timesteps % max_step == 0:
                    step_size = int(self.num_timesteps // max_step)
                    # terminates at max_step to ease computing the step_size.
                    self.step_array = np.array([(max_step - t) * step_size for t in range(max_step + 1)]).astype(int)
                else:
                    # why this is the case? we are given total diffusion steps and max_step budget.
                    # first we compute the approximate step size, which is also given for now.
                    # then we assume uneven strides: either steps with step-size or (step-size + 1)
                    # as a result, we can solve this system:
                    # step-size * a + (step-size + 1) * b = total_diffusion_steps
                    #             a +                   b = max_step.
                    # where a and b are number of steps of corresponding strides resp.
                    step_size = int(self.num_timesteps // max_step)
                    b = self.num_timesteps - step_size * max_step
                    self.step_array = np.concatenate(
                        [
                            [self.num_timesteps - t * (step_size + 1) for t in range(b)],
                            [step_size * (max_step - t) for t in range(b, max_step+1)]
                        ]
                        ).astype(int)
        return self.step_array[t], (self.step_array[t] - self.step_array[t+1])

    def _reparam_decoding(
        self, 
        output_tokens, 
        output_scores, 
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0, 
        non_special_sym_mask, 
        t,
        max_step,
        noise
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)
        
        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - (t + 1) / max_step
        elif schedule == "cosine":
            rate = np.cos((t + 1) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
            ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError
        
        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that 
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.
        
        # #### the `uncond` mode ####
        # In our reparameterized decoding, 
        # both v1_t and v2_t can be fully determined by the current token scores .
        
        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when 
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError
        
        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])

        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt_neq_x0