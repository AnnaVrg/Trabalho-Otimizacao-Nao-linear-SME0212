"""
bfgs_box.py - Método de Restrições Ativas para Minimização em Caixas com direção
Quase-Newton BFGS nas variáveis livres e passo SPG para abandono de face ("BFGS-Box").

Implementação própria (sem pacotes externos de otimização). NÃO é o L-BFGS-B de
Byrd et al. (1995): aqui a aproximação da inversa da Hessiana é densa (sem memória
limitada) e a face ativa é tratada por uma estratégia de restrições ativas.
"""

import numpy as np
from spg import (project_box, compute_continuous_projected_gradient,
                 initial_spectral_step, armijo_quadratic, spectral_step, STAG_TOL)


def free_variables(x, lower, upper, tol=1e-9):
    """Máscara das variáveis livres (estritamente no interior da caixa)."""
    return (x > lower + tol) & (x < upper - tol)


def compute_internal_gradient(gp, free_mask):
    """Gradiente interno à face: componentes de g_P nas variáveis livres e zero nas ativas."""
    return np.where(free_mask, gp, 0.0)


def compute_alpha_max(x, p, lower, upper, free_mask, tol=1e-12):
    """Maior alpha em (0, 1] tal que x + alpha * p permanece na caixa."""
    alpha_max = 1.0
    for i in np.where(free_mask)[0]:
        if p[i] > tol:
            alpha_max = min(alpha_max, (upper[i] - x[i]) / p[i])
        elif p[i] < -tol:
            alpha_max = min(alpha_max, (lower[i] - x[i]) / p[i])
    return max(0.0, min(1.0, float(alpha_max)))


def update_bfgs_inverse_hessian(H, s, y):
    """
    Atualização BFGS da inversa da Hessiana, com rho = 1 / (y^T s):
        H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
    Só é aplicada se a condição de curvatura y^T s > 0 for satisfeita.
    """
    ys = float(np.dot(y, s))
    if ys <= 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
        return H, False
    rho = 1.0 / ys
    V = np.eye(len(s)) - rho * np.outer(s, y)
    return V @ H @ V.T + rho * np.outer(s, s), True


def _result(x, fx, k, nfev, ngeval, norm_gp, history, converged, reason, message):
    return {'x': x, 'f': fx, 'iter': k, 'nfev': nfev, 'ngeval': ngeval,
            'norm_gp': norm_gp, 'history': history, 'converged': converged,
            'stop_reason': reason, 'message': message}


def solve_bfgs_box(func, grad_func, x0, lower, upper,
                   eps=1e-3, max_iter=60, eta=0.1,
                   c=1e-4, rho1=0.1, rho2=0.9,
                   lambda_min=1e-10, lambda_max=1e10, verbose=False):
    """
    Passo a passo (a cada iteração k):
      1. Parada: ||g_P(x_k)||_inf <= eps.
      2. Teste de face: se ||g_I(x_k)||_2 < eta ||g_P(x_k)||_2, ABANDONA a face com
         um passo SPG e reinicia H = I.
      3. Caso contrário, PERMANECE na face:
         a. direção Quase-Newton nas variáveis livres L: d_L = -H_LL grad_L (d = 0 nas ativas);
         b. alpha_max = maior passo em (0, 1] sem sair da caixa;
         c. se alpha_max < 1 e f(x + alpha_max d) < f(x), aceita o ponto na fronteira;
         d. senão, busca de Armijo a partir de alpha = alpha_max.
      4. Se o novo ponto permaneceu na mesma face e s^T y > 0, atualiza H por BFGS
         (na primeira atualização após um reinício, H é escalada por (y^T s)/(y^T y)).
      Parada também por k = max_iter ou estagnação (falha da busca / passo < 1e-10).
    """
    x = project_box(np.asarray(x0, dtype=float), lower, upper)
    n = len(x)
    fx = func(x)
    grad = grad_func(x)
    nfev, ngeval = 1, 1

    H = np.eye(n)
    needs_scaling = True
    history = []
    prev_x, prev_grad = None, None

    for k in range(max_iter):
        # Passo 1: teste de parada
        gp = compute_continuous_projected_gradient(x, grad, lower, upper)
        norm_gp = float(np.linalg.norm(gp, ord=np.inf))
        history.append({'iter': k, 'f': fx, 'norm_gp': norm_gp, 'x': np.copy(x)})
        if verbose:
            print(f"Iter {k:3d}: f = {fx:12.6f}, ||gP||_inf = {norm_gp:10.4e}")
        if norm_gp <= eps:
            return _result(x, fx, k, nfev, ngeval, norm_gp, history, True, 'gp',
                           f"Convergência: ||gP||_inf = {norm_gp:.2e} <= {eps:.2e}")

        # Passo 2: teste de permanência na face (norma 2)
        free_mask = free_variables(x, lower, upper)
        gi = compute_internal_gradient(gp, free_mask)
        stay_in_face = np.any(free_mask) and \
            (np.linalg.norm(gi) >= eta * np.linalg.norm(gp))

        if stay_in_face:
            # Passo 3a: direção Quase-Newton nas variáveis livres
            L = np.where(free_mask)[0]
            pk = np.zeros(n)
            pk[L] = -(H[np.ix_(L, L)] @ grad[L])
            g_dot_p = float(np.dot(grad, pk))
            if g_dot_p >= 0.0:            # salvaguarda: H perdeu a positividade na face
                pk = np.zeros(n)
                pk[L] = -grad[L]
                g_dot_p = float(np.dot(grad, pk))
                H = np.eye(n)
                needs_scaling = True

            # Passo 3b: passo máximo dentro da caixa
            alpha_max = compute_alpha_max(x, pk, lower, upper, free_mask)

            # Passo 3c: tenta o ponto na fronteira
            ok = False
            if 0.0 < alpha_max < 1.0:
                x_b = project_box(x + alpha_max * pk, lower, upper)
                f_b = func(x_b)
                nfev += 1
                if f_b < fx:
                    ok, x_next, f_next = True, x_b, f_b

            # Passo 3d: busca linear de Armijo a partir de alpha_max
            if not ok:
                ok, alpha, f_next, nf = armijo_quadratic(func, x, fx, g_dot_p, pk,
                                                         alpha_max if alpha_max > 0 else 1.0,
                                                         c, rho1, rho2)
                nfev += nf
                x_next = project_box(x + alpha * pk, lower, upper)
        else:
            # Passo 2 (abandono da face): um passo SPG e reinício de H
            if prev_x is None:
                lam, ng = initial_spectral_step(x, grad, grad_func, lower, upper,
                                                lambda_min=lambda_min, lambda_max=lambda_max)
                ngeval += ng
            else:
                lam = spectral_step(x, prev_x, grad, prev_grad, lambda_min, lambda_max)
            pk = project_box(x - lam * grad, lower, upper) - x
            g_dot_p = float(np.dot(grad, pk))
            if g_dot_p >= 0.0:
                pk = gp
                g_dot_p = float(np.dot(grad, pk))
            ok, alpha, f_next, nf = armijo_quadratic(func, x, fx, g_dot_p, pk, 1.0, c, rho1, rho2)
            nfev += nf
            x_next = x + alpha * pk
            H = np.eye(n)
            needs_scaling = True

        # Critério de estagnação
        if not ok or float(np.linalg.norm(x_next - x, ord=np.inf)) < STAG_TOL:
            return _result(x, fx, k, nfev, ngeval, norm_gp, history, False, 'stag',
                           "Estagnação: busca linear falhou ou passo < 1e-10.")

        grad_next = grad_func(x_next)
        ngeval += 1

        # Passo 4: atualização BFGS apenas se permaneceu na mesma face
        if stay_in_face and np.array_equal(free_variables(x_next, lower, upper), free_mask):
            sk = x_next - x
            yk = grad_next - grad
            ys, yy = float(np.dot(yk, sk)), float(np.dot(yk, yk))
            if needs_scaling and ys > 0.0 and yy > 1e-14:
                H = (ys / yy) * np.eye(n)
                needs_scaling = False
            H, _ = update_bfgs_inverse_hessian(H, sk, yk)

        prev_x, prev_grad = np.copy(x), np.copy(grad)
        x, fx, grad = x_next, f_next, grad_next

    gp = compute_continuous_projected_gradient(x, grad, lower, upper)
    norm_gp = float(np.linalg.norm(gp, ord=np.inf))
    history.append({'iter': max_iter, 'f': fx, 'norm_gp': norm_gp, 'x': np.copy(x)})
    return _result(x, fx, max_iter, nfev, ngeval, norm_gp, history, norm_gp <= eps,
                   'gp' if norm_gp <= eps else 'maxit',
                   f"Número máximo de iterações ({max_iter}) atingido.")
