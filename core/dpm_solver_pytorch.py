import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(self,
                 schedule='discrete',
                 betas=None,
                 alphas_cumprod=None,
                 continuous_beta_0=0.1,
                 continuous_beta_1=20.,
                 dtype=torch.float32,
        ):
        """
        NoiseScheduleVP 클래스는 VP type SDE의 noise schedule을 캡슐화합니다.
        
        - 'discrete' 모드:
            DDPM에서 사용하는 betas나 alphas_cumprod를 받아, log(alpha_t) 값을 계산합니다.
            t는 [1/N, 1] 범위의 연속 시간으로 변환됩니다.
        
        - 'linear' 모드:
            ScoreSDE에서 사용되는 선형 VPSDE로, beta_0, beta_1을 하이퍼파라미터로 사용합니다.
        """
        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. ...".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            # discrete-time인 경우: betas 또는 alphas_cumprod를 이용해서 log(alpha) 값을 계산
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            # 수치적 안정성을 위해 log_alphas를 clip
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            # t_array: discrete step에 해당하는 연속 시간 값 (ex. 1000개)
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        일부 beta schedule (예: cosine schedule)에서는 log-SNR 값이 불안정할 수 있으므로,
        t=T 근처에서 log-SNR 값을 clip합니다.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        주어진 t에 대해 log(alpha_t)를 계산합니다.
        - discrete schedule에서는 선형 보간(interpolate_fn)을 사용합니다.
        - linear schedule에서는 closed-form 해를 사용합니다.
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        alpha_t = exp(log(alpha_t))
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        sigma_t = sqrt(1 - alpha_t^2)
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        lambda_t = log(alpha_t) - log(sigma_t)
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        lambda_t가 주어졌을 때, 대응하는 t를 계산합니다.
        linear와 discrete mode 각각에 대해 다르게 구현됩니다.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """
    Diffusion 모델을 DPM-Solver에 사용할 수 있도록 래핑합니다.
    
    - model_type: "noise", "x_start", "v", "score" 중 하나로 모델의 파라미터화를 나타냄.
    - guidance_type: "uncond", "classifier", "classifier-free"로 샘플링 가이드 방식을 선택.
    - 내부에서 t_continuous를 모델이 요구하는 시간 t_input으로 변환합니다.
    """
    def get_model_input_time(t_continuous):
        # discrete-time DPMs의 경우, t_continuous를 [0, 1000*(N-1)/N] 범위로 변환
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        # 조건이 없으면 기본 모델 호출, 있으면 조건과 함께 처리 (여기서는 concat 사용)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(torch.cat((cond, x), dim=1), t_input, **model_kwargs)
        # model_type에 따라 반환 값 후처리
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        # classifier guidance를 위한 x에 대한 gradient를 계산
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        """
        DPM-Solver++ 초기화
        
        - model_fn: noise prediction 모델 함수 (연속 시간 t를 입력으로 사용)
        - algorithm_type: "dpmsolver" 또는 "dpmsolver++"
        - correcting_x0_fn: 데이터 예측 보정 함수 (ex, dynamic thresholding)
        - correcting_xt_fn: intermediate xt 보정 함수
        - dynamic_thresholding_ratio와 thresholding_max_val은 dynamic thresholding 설정값
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        Imagen에서 제안한 dynamic thresholding 기법: 
        x0의 절대값에 대해 정해진 quantile을 구한 후, 해당 범위 내로 clamp
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """모델을 통해 noise prediction 값을 반환"""
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        노이즈 예측값을 사용하여 x0 (denoised 데이터)를 계산
        필요시 correcting_x0_fn으로 보정
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        algorithm_type에 따라 noise prediction 또는 data prediction 값을 반환
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    # ---------------------------------------------------------------------------
    # 여러 solver 업데이트 함수 구현 (first, second, third update 등)
    # ---------------------------------------------------------------------------
    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        첫 번째 업데이트 (DPM-Solver-1, DDIM과 동일) - 1차 업데이트.
        dpmsolver++와 dpmsolver의 경우 식이 다르게 적용
        """
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 업데이트 식: x_t = (sigma_t/sigma_s)*x - alpha_t*phi_1*model_s
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    # (이외에도 singlestep_dpm_solver_second_update, third_update,
    #  multistep_dpm_solver_update, adaptive solver 등 여러 고차 업데이트 함수들이 구현되어 있음)
    # 각각 업데이트 함수는 모델 출력과 noise schedule을 조합하여 x의 다음 값을 계산
    
    # ---------------------------------------------------------------------------
    # 샘플링 인터페이스: add_noise, inverse, sample
    # ---------------------------------------------------------------------------
    def add_noise(self, x, t, noise=None):
        """
        주어진 x에 대해 t 시간의 노이즈를 추가하여 xt = alpha_t*x + sigma_t*noise를 계산
        """
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        # alpha_t와 sigma_t의 차원을 x에 맞게 확장
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        return xt.squeeze(0) if t.shape[0] == 1 else xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        주어진 x (노이즈 샘플)를 DPM-Solver를 사용하여 역 확산(inverse diffusion)
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        # t 범위 및 샘플링 방법에 따라 x를 점진적으로 업데이트합니다.
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        샘플링 인터페이스:
         - 초기 노이즈 x에서 시작하여, 주어진 시간 범위 [t_start, t_end]로부터 x0(클린 이미지)를 복원
         - method에 따라 singlestep, multistep, adaptive 등의 다양한 샘플링 방식을 지원
         - 또한, skip_type에 따라 시간 스텝 간 간격을 조절
        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        # ... (샘플링 루프 및 내부 업데이트 생략; 위에서 소개한 업데이트 함수들 호출)
        # 최종적으로 x0를 반환합니다.
        # (중간 값들을 함께 반환하는 옵션도 있음)

def interpolate_fn(x, xp, yp):
    """
    xp, yp를 keypoint로 사용하여 x에 대한 선형 보간 값을 계산
    autograd가 가능한 방식으로 구현되어 있어 역전파에도 사용
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # 선형 보간: start_y + (x - start_x) * slope
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

def expand_dims(v, dims):
    """
    텐서 v를 dims 차원까지 확장
    예를 들어, v가 (N,)이면 (N, 1, 1, ..., 1)로 확장
    """
    return v[(...,) + (None,)*(dims - 1)]
