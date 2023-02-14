import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from discrete_diffusions.utils import mean_ds
from discrete_diffusions.discrete_diffusion_base import DiscreteDiffusion

# adapted from https://github.com/samb-t/unleashing-transformers/blob/master/models/absorbing_diffusion.py
class AbsorbingDiffusion(DiscreteDiffusion):
    def __init__(
            self, 
            num_timesteps,
            mask_id, 
            lambda_direct_xentropy, 
            not_diffusing_special_sym,
            pad_id, bos_id, eos_id
        ):
        super().__init__(num_timesteps)
        # mask the transition probability from normal tokens to special symbols.
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.mask_idx = mask_id
        self.lambda_direct_xentropy = lambda_direct_xentropy
        self.num_timesteps = num_timesteps
        self.not_diffusing_special_sym = not_diffusing_special_sym

    def q_sample_coupled(self, x_0, t1, t2, non_special_sym_mask):
        _t1 = torch.maximum(t1, t2).float().unsqueeze(-1) + 1
        _t2 = torch.minimum(t1, t2).float().unsqueeze(-1) + 1

        # first sample q(x_{t1} | x_0)
        # and then sample q(x_{t2} | x_{t1}, x_0)
        # if t1 == t2, then draw an indep. sample.

        select_mask = (_t1 == _t2).float()
        u1 = torch.rand_like(x_0.float())
        # TODO: investigate the effect of such antithetic pairs.
        u2 = torch.rand_like(x_0.float()) # 1. - u1
        mask_t1 = u1 < (_t1 / self.num_timesteps)
        # for skip steps, the prob. of being **decoded** is
        # p = (_t1 - _t2) / _t1. Therefore, u2 > p indicates
        # the prob. that each token still gets masked.
        _mask_t2_if_neq = u2 > ((_t1 - _t2) / _t1)
        mask_t2_if_neq = torch.bitwise_and(_mask_t2_if_neq, mask_t1)

        mask_t2_if_eq = u2 < (_t2 / self.num_timesteps)

        mask_t2 = mask_t2_if_neq * (1. - select_mask) + mask_t2_if_eq * select_mask
        mask_t2 = mask_t2.bool()

        # masked out special symbols
        if self.not_diffusing_special_sym:
            mask_t1 = torch.bitwise_and(mask_t1, non_special_sym_mask)
            mask_t2 = torch.bitwise_and(mask_t2, non_special_sym_mask)
            
        x_t1, x_0_ignore_t1 = x_0.clone(), x_0.clone()
        x_t2, x_0_ignore_t2 = x_0.clone(), x_0.clone()
        x_t1[mask_t1] = self.mask_idx
        x_0_ignore_t1[torch.bitwise_not(mask_t1)] = -1
        x_t2[mask_t2] = self.mask_idx
        x_0_ignore_t2[torch.bitwise_not(mask_t2)] = -1

        return (torch.cat([x_t1, x_t2], dim=0), 
                torch.cat([x_0_ignore_t1, x_0_ignore_t2], dim=0), 
                torch.cat([mask_t1, mask_t2], dim=0),
                torch.cat([_t1, _t2], dim=0).long().squeeze(dim=-1) - 1,
        )

    def q_sample(self, x_0, t, non_special_sym_mask):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_0.float()) < ((t.float().unsqueeze(-1) + 1) / self.num_timesteps)
        if self.not_diffusing_special_sym:
            mask = mask & non_special_sym_mask
        x_t[mask] = self.mask_idx
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def compute_loss(self, inputs, **kwargs):
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        x_0 = inputs["x_0"]
        x_0_ignore = inputs["x_0_ignore"]
        t = inputs["t"]
        weight_t = inputs["weight_t"]
        decoder_outputs = inputs["decoder_outputs"]
        assert t.dim() == 1
        if inputs["masks"] is None:
            masks = inputs["x_t"].eq(self.unk)
        else:
            masks = inputs["masks"]
        logits = decoder_outputs.transpose(-1, -2)
        # mean over all tokens, even though some unmasked tokens do not produce losses.
        cross_entropy_loss = F.cross_entropy(logits, x_0_ignore, ignore_index=-1, reduction='none').mean(1)
        # t + 1 here since the passed t ranges from 0 to T-1.
        vb_loss = (1/(t+1)) * cross_entropy_loss
        diffusion_nll_loss = mean_ds(weight_t * vb_loss)
        if label_smoothing > 0:
            logit_loss = mean_ds(
                weight_t * 
                F.log_softmax(decoder_outputs, dim=-1).mean(dim=-1).masked_fill(~masks, 0.).mean(1)
            )
            diffusion_loss = (
                diffusion_nll_loss * (1 - label_smoothing) - logit_loss * label_smoothing
            )
        else:
            diffusion_loss = diffusion_nll_loss
        if self.lambda_direct_xentropy > 0:
            ce_loss = mean_ds(weight_t * F.cross_entropy(logits, x_0, reduction='none').mean(1)) # [B]
            diffusion_loss = diffusion_loss + self.lambda_direct_xentropy * ce_loss
        else:
            ce_loss = None

        output_dict = {
            'diffusion_loss': diffusion_loss,
            'diffusion_nll_loss': diffusion_nll_loss,
            'ce_loss': ce_loss
        }
        logging_outputs = {
            'loss': vb_loss,
            "t": t,
            "weights": weight_t,
        }
        return output_dict, logging_outputs

    def sample_step(self, decoder_out, denoising_fn, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        t = decoder_out.step
        max_step = decoder_out.max_step
        
        temperature_annealing = kwargs.get('temperature_annealing', False)
        decoding_strategy = kwargs.get('decoding_strategy', "linear")
        argmax_decoding = kwargs.get('argmax_decoding', False)
        decoding_time_difference = kwargs.get('decoding_time_difference', 0.0)
        if temperature_annealing:
            temperature = -0.05 * (t / (max_step - 1)) + 0.5
        else:
            temperature = kwargs.get('temperature', 1.0)

        cur_step, cur_stepsize = self._get_decoding_strategy(t, decoding_strategy, max_step)
        # print("original t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(t, cur_stepsize, cur_step, max_step, self.num_timesteps))

        # denoising_fn(x_t, t, **kwargs)
        scores = denoising_fn(
            x_t=output_tokens,
            t=torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long),
        )
        # scores = denoising_fn(
        #     normalize=False,
        #     prev_output_tokens=output_tokens,
        #     encoder_out=encoder_out,
        #     t=torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
        # )

        # redistributing probs. to avoid generating unk explicitly.
        scores[..., self.mask_idx] = -math.inf  # apply unk penalty
        scores = torch.log_softmax(scores, dim=-1)

        # manually construct a non-special-sym mask.
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) & 
            output_tokens.ne(self.bos_id) & 
            output_tokens.ne(self.eos_id)
        )
        non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)

        if decoding_strategy.startswith("reparam"):

            if argmax_decoding:
                cur_scores, cur_tokens = scores.max(-1)
            else:
                cur_tokens = dists.Categorical(logits=scores / temperature).sample()
                cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

            # this function modifies output_tokens and output_scores in place.
            # see the function for more details.
            output_masks = self._reparam_decoding(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                decoding_strategy=decoding_strategy,
                xt_neq_x0=decoder_out.auxiliary_output["output_masks"],
                non_special_sym_mask=non_special_sym_mask,
                t=t,
                max_step=max_step,
                noise=self.mask_idx,
            )
        else:
            if decoding_time_difference > 0.0:
                if cur_step <= cur_stepsize:
                    cur_step = cur_stepsize
                else:
                    cur_step = max(cur_step - decoding_time_difference, int(1.5 * cur_stepsize))
            # get the mask
            # <bos>, <eos> are ignored in this case since
            # they are not equal to unk.
            output_masks = output_tokens.eq(self.mask_idx)
            unmask_prob = cur_stepsize / cur_step
            # where to unmask
            changes = torch.rand(output_tokens.shape, device=output_tokens.device) < unmask_prob
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_and(changes, output_masks)

            if argmax_decoding:
                output_scores, new_tokens = scores.max(-1)
            else:
                new_tokens = dists.Categorical(logits=scores / temperature).sample()
                output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            output_tokens[changes] = new_tokens[changes]
        # return output_tokens, output_scores
        history = decoder_out.history
        if history is not None:
            history.append(output_tokens.clone())

        auxiliary_output = decoder_out.auxiliary_output
        auxiliary_output["output_masks"] = output_masks

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            auxiliary_output=auxiliary_output,
            history=history,
        )