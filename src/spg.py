"""
spg.py - Método do Gradiente Espectral Projetado (SPG) para Minimização em Caixas.

Baseado em: Birgin, E. G., Martinez, J. M., & Raydan, M. (2000). "Nonmonotone spectral
projected gradient methods on convex sets", SIAM Journal on Optimization, 10(4), 1196-1211.
(Versão monótona: busca linear de Armijo com interpolação quadrática salvaguardada.)
"""

import numpy as np

# Tolerância do critério de estagnação ||x_{k+1} - x_k||_inf < STAG_TOL
STAG_TOL = 1e-10


def project_box(x, lower, upper):
    """Projeção ortogonal na caixa Omega: P_Omega(x)_i = min(upper_i, max(lower_i, x_i))."""
    return np.clip(x, lower, upper)


def compute_continuous_projected_gradient(x, grad, lower, upper):
    """Gradiente projetado contínuo: gP(x) = P_Omega(x - grad) - x."""
    return project_box(x - grad, lower, upper) - x


def initial_spectral_step(x0, grad0, grad_func, lower, upper,
                          eps_rel=1e-5, eps_abs=1e-5,
                          lambda_min=1e-10, lambda_max=1e10):
    """
    Passo espectral inicial lambda_0: dá-se um pequeno passo auxiliar
    x_bar = P_Omega(x0 - a * grad0) e usa-se s = x_bar - x0, y = grad(x_bar) - grad0.
    Retorna (lambda_0, número de avaliações de gradiente usadas).
    """
    a = max(eps_rel * np.max(np.abs(x0)), eps_abs)
    x_bar = project_box(x0 - a * grad0, lower, upper)
    s_bar = x_bar - x0
    if np.allclose(s_bar, 0.0):
        return 1.0, 0
    y_bar = grad_func(x_bar) - grad0
    sy = float(np.dot(s_bar, y_bar))
    if sy <= 0.0:
        return lambda_max, 1
    ss = float(np.dot(s_bar, s_bar))
    return min(lambda_max, max(lambda_min, ss / sy)), 1


def armijo_quadratic(func, x, fx, g_dot_p, pk, alpha0=1.0, c=1e-4,
                     rho1=0.1, rho2=0.9, max_ls_iter=30):
    """
    Busca linear de Armijo com interpolação quadrática salvaguardada ao longo de x + alpha * pk.
    Retorna (sucesso, alpha, f(x + alpha * pk), número de avaliações de f).
    Em caso de falha (Armijo não satisfeito em max_ls_iter tentativas), sucesso = False
    e o ponto atual NÃO deve ser alterado.
    """
    alpha = alpha0
    nfev = 0
    for _ in range(max_ls_iter):
        f_trial = func(x + alpha * pk)
        nfev += 1
        if f_trial <= fx + c * alpha * g_dot_p:
            return True, alpha, f_trial, nfev
        denom = 2.0 * (f_trial - fx - alpha * g_dot_p)
        alpha_tent = -(alpha ** 2 * g_dot_p) / denom if abs(denom) > 1e-14 else alpha / 2.0
        if rho1 * alpha <= alpha_tent <= rho2 * alpha:
            alpha = alpha_tent
        else:
            alpha = alpha / 2.0
    return False, 0.0, fx, nfev


def spectral_step(x, prev_x, grad, prev_grad, lambda_min, lambda_max):
    """Passo espectral de Barzilai-Borwein com salvaguardas."""
    sk = x - prev_x
    yk = grad - prev_grad
    sy = float(np.dot(sk, yk))
    if sy <= 0.0:
        return lambda_max
    return min(lambda_max, max(lambda_min, float(np.dot(sk, sk)) / sy))


def _result(x, fx, k, nfev, ngeval, norm_gp, history, converged, reason, message):
    return {'x': x, 'f': fx, 'iter': k, 'nfev': nfev, 'ngeval': ngeval,
            'norm_gp': norm_gp, 'history': history, 'converged': converged,
            'stop_reason': reason, 'message': message}


def spg_box(func, grad_func, x0, lower, upper,
            eps=1e-3, max_iter=60,
            lambda_min=1e-10, lambda_max=1e10,
            c=1e-4, rho1=0.1, rho2=0.9, verbose=False):
    """
    Método do Gradiente Espectral Projetado para caixas.

    Critérios de parada:
      1. ||gP(x_k)||_inf <= eps                      (convergência estacionária)
      2. k = max_iter                                 (limite de iterações)
      3. ||x_{k+1} - x_k||_inf < STAG_TOL ou falha da busca linear (estagnação)
    """
    x = project_box(np.asarray(x0, dtype=float), lower, upper)
    fx = func(x)
    grad = grad_func(x)
    nfev, ngeval = 1, 1

    history = []
    prev_x, prev_grad = None, None
    lambda_k = 1.0

    for k in range(max_iter):
        # Passo 1: teste de parada com o gradiente projetado contínuo
        gp = compute_continuous_projected_gradient(x, grad, lower, upper)
        norm_gp = float(np.linalg.norm(gp, ord=np.inf))
        history.append({'iter': k, 'f': fx, 'norm_gp': norm_gp, 'x': np.copy(x)})
        if verbose:
            print(f"Iter {k:3d}: f = {fx:12.6f}, ||gP||_inf = {norm_gp:10.4e}, lambda = {lambda_k:10.4e}")
        if norm_gp <= eps:
            return _result(x, fx, k, nfev, ngeval, norm_gp, history, True, 'gp',
                           f"Convergência: ||gP||_inf = {norm_gp:.2e} <= {eps:.2e}")

        # Passo 2: passo espectral lambda_k
        if k == 0:
            lambda_k, ng = initial_spectral_step(x, grad, grad_func, lower, upper,
                                                 lambda_min=lambda_min, lambda_max=lambda_max)
            ngeval += ng
        else:
            lambda_k = spectral_step(x, prev_x, grad, prev_grad, lambda_min, lambda_max)

        # Passo 3: direção projetada d_k = P_Omega(x_k - lambda_k grad_k) - x_k
        pk = project_box(x - lambda_k * grad, lower, upper) - x
        g_dot_p = float(np.dot(grad, pk))
        if g_dot_p >= 0.0:
            pk = gp
            g_dot_p = float(np.dot(grad, pk))

        # Passo 4: busca linear de Armijo com interpolação quadrática
        ok, alpha_k, f_new, nf = armijo_quadratic(func, x, fx, g_dot_p, pk, 1.0, c, rho1, rho2)
        nfev += nf
        step = alpha_k * pk
        if not ok or float(np.linalg.norm(step, ord=np.inf)) < STAG_TOL:
            return _result(x, fx, k, nfev, ngeval, norm_gp, history, False, 'stag',
                           "Estagnação: busca linear falhou ou passo < 1e-10.")

        prev_x, prev_grad = np.copy(x), np.copy(grad)
        x = x + step
        fx = f_new
        grad = grad_func(x)
        ngeval += 1

    gp = compute_continuous_projected_gradient(x, grad, lower, upper)
    norm_gp = float(np.linalg.norm(gp, ord=np.inf))
    history.append({'iter': max_iter, 'f': fx, 'norm_gp': norm_gp, 'x': np.copy(x)})
    return _result(x, fx, max_iter, nfev, ngeval, norm_gp, history, norm_gp <= eps,
                   'gp' if norm_gp <= eps else 'maxit',
                   f"Número máximo de iterações ({max_iter}) atingido.")
