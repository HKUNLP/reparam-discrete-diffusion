import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from inspect import isfunction
from typing import NamedTuple
from discrete_diffusions.utils import mean_ds, log1mexp
from discrete_diffusions.discrete_diffusion_base import DiscreteDiffusion

noise_schedule =  NamedTuple("noise_schedule", [
    ("log_alpha", torch.Tensor), 
    ("log_1_min_alpha", torch.Tensor),
    ("log_cumprod_alpha", torch.Tensor),
    ("log_1_min_cumprod_alpha", torch.Tensor)
    ])

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_0, log_prob):
    return (log_x_0.exp() * log_prob).sum(dim=-1)


def index_to_log_onehot(x, vocab_size):
    assert x.max().item() < vocab_size, \
        f'Error: {x.max().item()} >= {vocab_size}'
    x_onehot = F.one_hot(x, vocab_size) # [b, n, c]
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(-1)

def multinomial_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def get_named_alpha_schedule(timesteps, name='cosine', alpha_min=0.001, alpha_max=1.):
    if name == 'linear':
        alphas_cumprod = np.linspace(1, 0., timesteps + 1)
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    elif name in ['cosine', 'cosine_old']:
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        s = 0.008
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    elif name in ['sqrt']:
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = 1 - np.sqrt((x / steps) + 0.0001)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=alpha_min, a_max=alpha_max)
    if name == 'cosine_old':
        # to reproduce multinomial_diffusion numbers.
        alphas = np.sqrt(alphas)
    return alphas

# adapted from https://github.com/ehoogeboom/multinomial_diffusion/blob/main/diffusion_utils/diffusion_multinomial.py
class MultinomialDiffusion(DiscreteDiffusion):
    def __init__(
            self, 
            num_timesteps,
            vocab_size, 
            lambda_direct_xentropy, 
            decoder_loss_type, 
            noise_scheduler_type,
            not_diffusing_special_sym,
            pad_id, bos_id, eos_id,
        ):
        super().__init__(num_timesteps)
        self.vocab_size = vocab_size
        # with <bos>, <eos>, <pad> removed.
        self.not_diffusing_special_sym = not_diffusing_special_sym
        assert self.not_diffusing_special_sym, "ignoring special symbols and diffusing them as well might lead to drastically worse performance."
        self.num_non_special_tokens = vocab_size - 3
        self.decoder_loss_type = decoder_loss_type
        self.lambda_direct_xentropy = lambda_direct_xentropy
        self.num_timesteps = num_timesteps
        alphas = torch.tensor(get_named_alpha_schedule(self.num_timesteps, name=noise_scheduler_type).astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)
        log_1_min_alpha = torch.log(1 - log_alpha.exp() + 1e-9)
        log_1_min_cumprod_alpha = torch.log(1 - log_cumprod_alpha.exp() + 1e-9)

        assert torch.logaddexp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert torch.logaddexp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # mask the transition probability from normal tokens to special symbols.
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # Convert to float32 and register buffers.
        # here log_alpha[t] denotes the noise schedule at t+1-th time step.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

    def q_sample_coupled(self, x_0, t1, t2, non_special_sym_mask):
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        _t1 = torch.maximum(t1, t2)
        _t2 = torch.minimum(t1, t2)
        log_x0 = index_to_log_onehot(x_0, self.vocab_size)
        
        log_x_t1_logits = self.q_xt_given_x0(log_x0, _t1, non_special_sym_mask)
        u1 = torch.rand_like(log_x_t1_logits)
        log_x_t1 = self.log_sample_categorical(log_x_t1_logits, u1) # [b, n, c]
        
        log_x_t2_logits_if_eq = self.q_xt_given_x0(log_x0, _t2, non_special_sym_mask)
        log_x_t2_logits_if_neq = self.q_tminusk_given_t_x0(
                    log_x_0=log_x0, 
                    log_x_t=log_x_t1, 
                    t=_t1,
                    non_special_sym_mask=non_special_sym_mask,
                    k=_t1 - _t2,
                    )
        select_mask = (_t1 == _t2).float().unsqueeze(-1).unsqueeze(-1)
        log_x_t2_logits = log_x_t2_logits_if_neq * (1. - select_mask) + log_x_t2_logits_if_eq * select_mask

        # TODO: investigate the effect of such antithetic pairs.
        # TODO: check when t1 == t2
        u2 = torch.rand_like(log_x_t2_logits) # 1. - u1
        log_x_t2 = self.log_sample_categorical(log_x_t2_logits, u2)
        # first sample q(x_{t1} | x_0)
        # and then sample q(x_{t2} | x_{t1}, x_0)
        # if t1 == t2, then draw an indep. sample.
        return (torch.cat([log_x_t1, log_x_t2], dim=0), torch.cat([_t1, _t2], dim=0))

    def q_sample(self, x_0, t, non_special_sym_mask):
        # here t ranges from 0 to T-1,
        # indexing time steps from 1 to T.
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        log_x0 = index_to_log_onehot(x_0, self.vocab_size)
        log_q_xt_x0 = self.q_xt_given_x0(log_x0, t, non_special_sym_mask)
        log_sample = self.log_sample_categorical(log_q_xt_x0) # [b, n, c]
        return log_sample
    
    def q_xt_given_x0(self, log_x0, t, non_special_sym_mask):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x0.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x0.shape)

        log_probs = torch.logaddexp(
            log_x0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_non_special_tokens)
        )
        # for non-symbol tokens, we mask their porbability getting into symbols;
        # for symbols, we force them to stay in the current state.
        log_probs[..., self.pad_id] = -30
        log_probs[..., self.bos_id] = -30
        log_probs[..., self.eos_id] = -30
        # log_probs = log_x0 * (1. - non_special_sym_mask_float) + (log_probs + self.tok_2_sym_mask) * non_special_sym_mask_float
        log_probs = torch.where(non_special_sym_mask.unsqueeze(-1), log_probs, log_x0)
        return log_probs
    
    def q_xt_given_xtminusk(self, log_xt, t, non_special_sym_mask, k=1):
        '''
            compute q(x_t | x_{t - k})
        '''
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_xt.shape)
        log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (t - k).clamp(min=0), log_xt.shape)
        mask = (t < k).float().unsqueeze(-1).unsqueeze(-1)
        zeros = torch.zeros_like(log_cumprod_alpha_t)
        log_cumprod_alpha_tminusk = mask * zeros + (1. - mask) * log_cumprod_alpha_tminusk

        log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk
        log_1_min_cumprod_alpha_from_tminusk_to_t = torch.log1p(-torch.exp(log_cumprod_alpha_from_tminusk_to_t))

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = torch.logaddexp(
            log_xt + log_cumprod_alpha_from_tminusk_to_t,
            log_1_min_cumprod_alpha_from_tminusk_to_t - np.log(self.num_non_special_tokens)
        )
        log_probs[..., self.pad_id] = -30
        log_probs[..., self.bos_id] = -30
        log_probs[..., self.eos_id] = -30
        # log_probs = log_x0 * (1. - non_special_sym_mask_float) + (log_probs + self.tok_2_sym_mask) * non_special_sym_mask_float
        log_probs = torch.where(non_special_sym_mask.unsqueeze(-1), log_probs, log_xt)
        return log_probs

    def q_tminusk_given_t_x0(
        self, 
        log_x_0, 
        log_x_t, 
        t,
        non_special_sym_mask,
        k=1, 
    ):
        '''
            compute q(x_{t - k} | x_{t}, x_0)
        '''
        t_minus_k = torch.clamp(t - k, min=0)
        theta_t_minus_k = self.q_xt_given_x0(log_x_0, t_minus_k, non_special_sym_mask)
        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        if isinstance(k, torch.Tensor):
            step_size = k.view(-1, *num_axes) * torch.ones_like(log_x_0)
        elif isinstance(k, int):
            step_size = k
        else:
            step_size = int(k)
        if self.decoder_loss_type == 'orig':
            theta_t_minus_k = torch.where(t_broadcast == step_size - 1, log_x_0, theta_t_minus_k)
            unnormed_logprobs = theta_t_minus_k + self.q_xt_given_xtminusk(log_x_t, t, non_special_sym_mask, k=k)
        elif self.decoder_loss_type == 'simple':
            # see https://github.com/ehoogeboom/multinomial_diffusion/issues/7#issue-1333344538
            # this usually improves performance for translation
            unnormed_logprobs = theta_t_minus_k + self.q_xt_given_xtminusk(log_x_t, t, non_special_sym_mask, k=k)
            unnormed_logprobs = torch.where(t_broadcast == step_size - 1, log_x_0, unnormed_logprobs)
        else:
            raise NotImplementedError("Decoder loss type {} is not implemented yet.".format(self.decoder_loss_type))
        theta_t_minus_k_given_t_x_0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return theta_t_minus_k_given_t_x_0

    def p_tminusk_given_t(
        self, 
        log_x0_recon,
        log_x_t, 
        t,
        non_special_sym_mask,
        k=1,
        posterior_parameterization="logit"
        ):
        if posterior_parameterization == "logit":
            log_x_0_tilde = log_x0_recon
        elif posterior_parameterization == "argmax":
            # gradients are blocked. need to use ST
            log_x_0_tilde = index_to_log_onehot(log_x0_recon.argmax(dim=-1), self.vocab_size)
        log_pred = self.q_tminusk_given_t_x0(
            log_x_0=log_x_0_tilde, 
            log_x_t=log_x_t, 
            t=t,
            non_special_sym_mask=non_special_sym_mask,
            k=k,
            )
        return log_pred

    def compute_kl_t_tminusk(
        self, 
        log_x0_recon,
        log_x_t,
        log_x_0, 
        t, 
        non_special_sym_mask,
        k=1,
        ):
        # t ranges from 0 to T-1
        # correspond to x_1 to x_T
        log_q_t = self.q_tminusk_given_t_x0(
            log_x_0=log_x_0, 
            log_x_t=log_x_t, 
            t=t,
            non_special_sym_mask=non_special_sym_mask,
            k=k,
            )
        log_p_t = self.p_tminusk_given_t(
            log_x0_recon=log_x0_recon,
            log_x_t=log_x_t, 
            t=t,
            non_special_sym_mask=non_special_sym_mask,
            k=k,
            )
        kl = multinomial_kl(log_q_t, log_p_t)
        decoder_nll = -log_categorical(log_x_0, log_p_t)

        mask = (t == torch.zeros_like(t)).float()
        kl = mask * decoder_nll.mean(dim=-1) + (1. - mask) * kl.mean(dim=-1)
        return kl
    
    def log_sample_categorical(self, logits, uniform_noise=None, eps=1e-10):
        if uniform_noise is None:
            uniform_noise = torch.rand_like(logits)
        else:
            assert uniform_noise.shape == logits.shape
        gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
        index_sample = (gumbel_noise + logits).argmax(dim=-1)
        log_sample = index_to_log_onehot(index_sample, self.vocab_size)

        # int_sample = dists.Categorical(logits=logits).sample()
        # onehot_sample = F.one_hot(int_sample, self.vocab_size)
        # log_sample = torch.log(onehot_sample.float().clamp(min=eps))
        return log_sample

    def kl_prior(self, log_x_0):
        b = log_x_0.size(0)
        device = log_x_0.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_xt_given_x0(
            log_x_0, 
            t=(self.num_timesteps - 1) * ones,
            )
        log_half_prob = -torch.log(self.vocab_size * torch.ones_like(log_qxT_prob))

        kl_prior = multinomial_kl(log_qxT_prob, log_half_prob).mean(dim=1)
        return kl_prior

    def compute_loss(self, inputs, **kwargs):
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        log_x_t = inputs["log_x_t"] # [b, n, c]
        x_0 = inputs["x_0"]
        t = inputs["t"]
        weight_t = inputs["weight_t"]
        non_special_sym_mask = inputs["non_special_sym_mask"]
        decoder_outputs = inputs["decoder_outputs"]
        log_x_0 = index_to_log_onehot(x_0, self.vocab_size)
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        log_x0_recon = F.log_softmax(decoder_outputs, dim=-1) # [b, n, c]
        # retain special symbols in input.
        log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_0)
        # t passed here are indexed in [0, T-1],
        # corresponding to x_1, x_2, \dots, x_T.
        kl_loss = self.compute_kl_t_tminusk(
            log_x0_recon,
            log_x_t,
            log_x_0, 
            t,
            non_special_sym_mask,
            k = 1,
            )
        loss = kl_loss * weight_t
        if self.lambda_direct_xentropy > 0:
            ce_loss = F.cross_entropy(log_x0_recon.transpose(-1, -2), x_0, reduction='none').mean(1) * weight_t
            loss = loss + self.lambda_direct_xentropy * ce_loss
        else:
            ce_loss = None

        # currently does not use label smoothing, as 
        # the target distribution q in loss KL(q||p) is already smooth enough.
        diffusion_nll_loss = mean_ds(loss)
        if label_smoothing > 0:
            logits = mean_ds(F.log_softmax(decoder_outputs[non_special_sym_mask], dim=-1))
            diffusion_loss = (
                diffusion_nll_loss * (1 - label_smoothing) - logits * label_smoothing
            )
        else:
            diffusion_loss = diffusion_nll_loss

        output_dict = {
            'diffusion_loss': diffusion_loss,
            'diffusion_nll_loss': diffusion_nll_loss,
            'ce_loss': ce_loss
        }
        logging_outputs = {
            'loss': kl_loss,
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
        decoding_time_difference = kwargs.get('decoding_time_difference', 0.0)
        argmax_decoding = kwargs.get('argmax_decoding', False)
        multinomial_decode_posterior_parameterization = kwargs.get('multinomial_decode_posterior_parameterization', "logit")
        decoding_strategy = kwargs.get('decoding_strategy', "linear")
        adaptive_decoding = kwargs.get('adaptive_decoding', False)
        if temperature_annealing:
            temperature = -0.05 * (t / (max_step - 1)) + 0.5
        else:
            temperature = kwargs.get('temperature', 1.0)

        # manually construct a non-special-sym mask.
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) & 
            output_tokens.ne(self.bos_id) & 
            output_tokens.ne(self.eos_id)
        )
        non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)

        if adaptive_decoding:
            effective_seq_len = non_special_sym_mask.float().sum(-1)
            _max_step = max_step + effective_seq_len - effective_seq_len 
            _max_step = torch.minimum(effective_seq_len, _max_step).clamp(min=1)
            # tensor, tensor
            cur_step_tensor, cur_stepsize = self._get_batched_decoding_strategy(t, _max_step)
            cur_step_tensor = cur_step_tensor - 1
            # print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_step_tensor, cur_stepsize, t, _max_step, self.num_timesteps), flush=True)
        else:
            # int, int
            cur_step, cur_stepsize = self._get_decoding_strategy(t, decoding_strategy, max_step)
            # minus 1 due to the offset.
            cur_step = cur_step - 1
            cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
            # print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_t[0], step_size, t, max_step, self.num_timesteps), flush=True)
        log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
        log_x0_recon = denoising_fn(
            x_t=output_tokens,
            t=cur_step_tensor,
        ) # [b, n, c]
        # log_x0_recon = torch.log_softmax(log_x0_recon / temperature, dim=-1)
        log_x0_recon = torch.log_softmax(log_x0_recon, dim=-1)
        log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_t)

        if decoding_strategy.startswith("reparam"):
            if argmax_decoding:
                cur_scores, cur_tokens = log_x0_recon.max(-1)
            else:
                cur_tokens = dists.Categorical(logits=log_x0_recon / temperature).sample()
                cur_scores = torch.gather(log_x0_recon, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
            log_cumprod_alpha_t = extract(self.log_cumprod_alpha, cur_step_tensor, log_x_t.shape)
            log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (cur_step_tensor - cur_stepsize).clamp(min=0), log_x_t.shape)
            mask = (cur_step_tensor < cur_stepsize).float().unsqueeze(-1).unsqueeze(-1)
            log_cumprod_alpha_tminusk = mask * torch.zeros_like(log_cumprod_alpha_t) + (1. - mask) * log_cumprod_alpha_tminusk
            log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk
            log_q_noise_xt = torch.logaddexp(
                log_x_t + log_cumprod_alpha_from_tminusk_to_t,
                log1mexp(log_cumprod_alpha_from_tminusk_to_t) - np.log(self.num_non_special_tokens)
            )
            _uniform_noise = torch.rand_like(log_q_noise_xt)
            gumbel_noise = -torch.log(-torch.log(_uniform_noise + 1e-10) + 1e-10)
            uniform_noise = torch.argmax(gumbel_noise + log_q_noise_xt, dim=-1)
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
                noise=uniform_noise,
            )
        else:
            output_masks = decoder_out.auxiliary_output["output_masks"]
            # instead of larger step sizes, we offset the current time instead.
            # we found this leads to better performance and less noisy translates.
            new_cur_step_tensor = cur_step_tensor
            # NOTE we only use shifted time steps in computing the posterior.
            if decoding_time_difference > 0:
                if adaptive_decoding:
                    new_cur_step_tensor = torch.maximum(cur_step_tensor - decoding_time_difference, (1.5 * cur_stepsize).long())
                    new_cur_step_tensor = torch.where(cur_step_tensor >= cur_stepsize, new_cur_step_tensor, cur_step_tensor)
                else:
                    if cur_step >= cur_stepsize:
                        new_cur_step_tensor = (cur_step_tensor - decoding_time_difference).clamp(min=math.floor(1.5 * cur_stepsize))
            log_prob = self.p_tminusk_given_t(
                log_x0_recon=log_x0_recon,
                log_x_t=log_x_t, 
                t=new_cur_step_tensor,
                non_special_sym_mask=non_special_sym_mask,
                k=cur_stepsize,
                posterior_parameterization=multinomial_decode_posterior_parameterization
            )
            if argmax_decoding:
                output_scores, output_tokens = log_prob.max(-1)
            else:
                output_tokens = dists.Categorical(logits=log_prob / temperature).sample()
                output_scores = torch.gather(log_prob, -1, output_tokens.unsqueeze(-1)).squeeze(-1)
        
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
            attn=None,
            history=history,
        )

# if __name__ == "__main__":
#     torch.set_printoptions(threshold=10_000)
#     import logging
#     logger = logging.getLogger(__name__)
#     num_diffusion_timesteps = 10
#     mask_id = 1000
#     vocab_size = 32
#     seq_len = 8
#     pad_id = 0
#     bos_id = 1
#     eos_id = 2
#     diffusion = MultinomialDiffusion(
#         num_diffusion_timesteps, 
#         vocab_size, 
#         -1, 
#         "simple", 
#         "cosine", 
#         True,
#         "elbo",
#         pad_id=0, bos_id=1, eos_id=2)
#     batch_size = num_diffusion_timesteps
#     length_tgt = torch.randint(low=4, high=seq_len, size=(1,)).repeat(batch_size)
#     idx_length = torch.arange(seq_len)
#     # TODO here
#     x_0 = torch.randint(
#         0,
#         vocab_size, 
#         size=(1, seq_len)).repeat(batch_size, 1)
#     x_0.masked_fill_(
#         idx_length[None, :] >= length_tgt[:, None], pad_id
#     )
#     x_0[:, 0] = bos_id
#     x_0.scatter_(1, length_tgt[:, None] - 1, eos_id)

#     non_special_sym_mask = (
#         x_0.ne(pad_id) & 
#         x_0.ne(bos_id) & 
#         x_0.ne(eos_id)
#     )
#     print("length : {}, x_0 : {}, non_special_sym_mask : {}".format(length_tgt, x_0, non_special_sym_mask))
#     # x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
#     t = torch.arange(start=0, end=num_diffusion_timesteps)
#     weight_t = 0
#     log_xt = diffusion.q_sample(x_0=x_0, t=t, non_special_sym_mask=non_special_sym_mask)
#     print("x_t : {}".format(log_xt.exp().max(dim=-1)[1]))
    
#     # batch_size = (seq_len + num_diffusion_timesteps - 1) ** 2
#     # x_0 = torch.randint(low=3, high=vocab_size, size=(batch_size, seq_len))
#     # _t1 = torch.arange(start=1, end=seq_len + num_diffusion_timesteps)
#     # _t2 = torch.arange(start=1, end=seq_len + num_diffusion_timesteps)
#     # abs_t = torch.cartesian_prod(_t1, _t2)
#     # abs_t = (abs_t[:, 0], abs_t[:, 1])
#     # weight_t = x_0
#     # x_t, x_0_ignore, mask, abs_t, rel_t, weight_t = diffusion.q_sample_coupled(x_0=x_0,abs_t=abs_t, weight_t=weight_t)
#     # print("x_t : {}, x_0_ignore : {}, mask : {}, abs_t : {}, rel_t : {}".format(x_t, x_0_ignore, mask, abs_t, rel_t))
#     decoding_steps = num_diffusion_timesteps // 3
#     output_tokens = x_0[0:1]
#     def decoder(normalize, prev_output_tokens, encoder_out, t):
#         bs, seq_len = prev_output_tokens.shape
#         return torch.ones(bs, seq_len, vocab_size)
#     for t in range(decoding_steps):
#         print("##############################################################################")
#         output_tokens, _ = diffusion.sample_step(output_tokens, t, None, decoding_steps, decoder)
#         print("current decoded ====> [t] : {}, output: {}".format(t, output_tokens))
#         print("##############################################################################")
        