"""
hymod.py - Modelo Hidrológico Conceitual HYMOD e Função Objetivo de Calibração.

Baseado na formulação clássica de Wagener et al. (2001), Boyle et al. (2000)
e adaptado para a bacia urbana do Córrego do Monjolinho (São Carlos, SP)
conforme Silva, Benso & Vasconcelos (2026).
"""

import numpy as np

# Área da porção urbanizada da bacia do Monjolinho (página 3 do artigo)
AREA_KM2 = 79.6
# Discretização temporal padrão: 5 minutos = 300 segundos
DT_SECONDS = 300.0
# Fator de conversão de mm no intervalo para vazão média em m³/s
# 1 mm sobre 1 km² = 1000 m³. Logo, A km² * 1000 m³ / dt segundos
MM_TO_M3S = (AREA_KM2 * 1000.0) / DT_SECONDS  # ≈ 265.333 m³/s por mm/intervalo
# Vazão de base padrão (usada apenas se nenhuma for informada). Na calibração usa-se
# a vazão observada no primeiro intervalo de cada evento (Q_base = Q_obs[0]).
BASEFLOW_M3S = 2.4

# Limites de caixa dos parâmetros conforme a página 5 do artigo
# (cmax (50; 500), beta (0.1; 2), alpha (0.01; 1), kq (0.1; 20), ks (0.1; 50))
# e intervalo de W2 (0; 60) definido na análise de sensibilidade do artigo.
# theta = [cmax, beta, alpha, kq, ks, w2]
PARAM_NAMES = ["cmax", "beta", "alpha", "kq", "ks", "w2"]
LOWER_BOUNDS = np.array([50.0, 0.1, 0.01, 0.1, 0.1, 0.0])
UPPER_BOUNDS = np.array([500.0, 2.0, 1.0, 20.0, 50.0, 60.0])


def linear_reservoir(x_prev, inflow, k_res):
    """
    Simula um reservatório linear discreto conservativo.
    k_res: tempo de residência (em passos de tempo).
    r = 1 - exp(-1 / k_res) é a fração liberada no passo.
    """
    # Garante taxa estritamente entre 0 e 1
    r = 1.0 - np.exp(-1.0 / max(k_res, 1e-4))
    total_water = x_prev + inflow
    outflow = r * total_water
    x_next = (1.0 - r) * total_water
    return x_next, outflow


def hymod_simulate(params, precip, pet=None, return_all=False, q_base=BASEFLOW_M3S):
    """
    Executa a simulação do HYMOD para uma série de precipitação.

    Parâmetros:
    -----------
    params : array-like de 5 ou 6 elementos
        [cmax, beta, alpha, kq, ks] ou [cmax, beta, alpha, kq, ks, w2]
    precip : array-like (N,)
        Precipitação em mm para cada intervalo de 5 min.
    pet : array-like (N,), opcional
        Evapotranspiração potencial em mm (default 0 para eventos de cheia).
    return_all : bool
        Se True, retorna também componentes internos (escoamento rápido, lento, etc.)

    Retorna:
    --------
    q_sim : np.ndarray (N,)
        Vazão simulada no exutório em m³/s.
    """
    cmax = float(params[0])
    beta = float(params[1])
    alpha = float(params[2])
    kq = float(params[3])
    ks = float(params[4])
    w2 = float(params[5]) if len(params) > 5 else 0.0

    n_steps = len(precip)
    if pet is None:
        pet = np.zeros(n_steps)

    # Capacidade máxima de retenção integrada da bacia: Smax = Cmax / (beta + 1)
    b_plus_1 = beta + 1.0
    s_max = cmax / b_plus_1

    # Estado inicial de umidade no solo: limitado a Smax
    x_loss = min(max(w2, 0.0), s_max)

    # Estados dos 3 tanques rápidos em série e 1 tanque lento em paralelo
    xq1 = 0.0
    xq2 = 0.0
    xq3 = 0.0
    xs = 0.0

    q_sim = np.zeros(n_steps)
    q_quick_series = np.zeros(n_steps)
    q_slow_series = np.zeros(n_steps)

    for t in range(n_steps):
        p_t = max(float(precip[t]), 0.0)
        pet_t = max(float(pet[t]), 0.0)

        # 1. Contabilidade de Umidade do Solo (Soil Moisture Accounting)
        # Altura de água correspondente à umidade atual x_loss
        ratio = max(0.0, min(1.0, (b_plus_1 * x_loss) / cmax))
        ct_prev = cmax * (1.0 - (1.0 - ratio) ** (1.0 / b_plus_1))

        # Excesso de precipitação direto por saturação
        er1 = max(0.0, p_t - cmax + ct_prev)
        infil_potential = p_t - er1

        # Atualização do conteúdo do solo
        dummy = min(1.0, (ct_prev + infil_potential) / cmax)
        xn = s_max * (1.0 - (1.0 - dummy) ** b_plus_1)

        # Excesso adicional que excede a capacidade de absorção
        er2 = max(0.0, infil_potential - (xn - x_loss))
        total_er = er1 + er2

        # Evapotranspiração no passo
        evap = (1.0 - (s_max - xn) / s_max) * pet_t
        x_loss = max(0.0, xn - evap)

        # 2. Roteamento: Partição entre Escoamento Rápido e Lento
        u_quick = alpha * total_er
        u_slow = (1.0 - alpha) * total_er

        # 3. Cascata de 3 reservatórios lineares rápidos
        xq1, out_q1 = linear_reservoir(xq1, u_quick, kq)
        xq2, out_q2 = linear_reservoir(xq2, out_q1, kq)
        xq3, out_q3 = linear_reservoir(xq3, out_q2, kq)

        # 4. Reservatório linear lento (subterrâneo)
        xs, out_s = linear_reservoir(xs, u_slow, ks)

        # Vazão total em mm/intervalo convertida para m³/s somada à vazão de base pré-evento
        q_total_mm = out_q3 + out_s
        q_sim[t] = q_base + q_total_mm * MM_TO_M3S
        q_quick_series[t] = out_q3 * MM_TO_M3S
        q_slow_series[t] = q_base + out_s * MM_TO_M3S

    if return_all:
        return q_sim, q_quick_series, q_slow_series
    return q_sim


def compute_nse(q_obs, q_sim):
    """
    Calcula o Coeficiente de Eficiência de Nash-Sutcliffe (NSE):
    NSE = 1 - sum((q_sim - q_obs)^2) / sum((q_obs - mean(q_obs))^2)
    """
    q_obs = np.asarray(q_obs, dtype=float)
    q_sim = np.asarray(q_sim, dtype=float)
    numerator = np.sum((q_sim - q_obs) ** 2)
    denom = np.sum((q_obs - np.mean(q_obs)) ** 2)
    if denom < 1e-12:
        return -1e6
    return 1.0 - (numerator / denom)


def objective_function(params, precip, q_obs, pet=None, q_base=None):
    """
    Função objetivo a ser MINIMIZADA:
    J(theta) = - NSE(theta)
    Se q_base não for informado, usa a vazão observada no primeiro intervalo.
    """
    if q_base is None:
        q_base = float(q_obs[0])
    q_sim = hymod_simulate(params, precip, pet=pet, q_base=q_base)
    nse = compute_nse(q_obs, q_sim)
    return -nse


def compute_gradient_fd(func, theta, h=1e-5, lower=None, upper=None):
    """
    Gradiente numérico por diferenças finitas com passo h_i = h * max(|theta_i|, 1):
      - centradas no interior da caixa: (f(x + h e_i) - f(x - h e_i)) / (2h);
      - unilaterais quando x +/- h sairia da caixa (progressiva no limite inferior,
        regressiva no limite superior), para nunca avaliar o modelo fora de Omega.
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    grad = np.zeros(n)
    f0 = None
    for i in range(n):
        step = max(abs(theta[i]) * h, h)
        can_up = upper is None or theta[i] + step <= upper[i]
        can_down = lower is None or theta[i] - step >= lower[i]
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += step
        theta_minus[i] -= step
        if can_up and can_down:
            grad[i] = (func(theta_plus) - func(theta_minus)) / (2.0 * step)
        else:
            if f0 is None:
                f0 = func(theta)
            if can_up:
                grad[i] = (func(theta_plus) - f0) / step
            else:
                grad[i] = (f0 - func(theta_minus)) / step
    return grad
