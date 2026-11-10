"""
de.py - Evolução Diferencial (DE) para minimização em caixas.

Implementação própria (sem pacotes externos de otimização). Por padrão reproduz a
configuração padrão do pacote DEoptim do R (Mullen et al., 2011), usado no artigo-base:
  - estratégia 2, "DE/local-to-best/1/bin":  v = x_i + F (x_best - x_i) + F (x_r2 - x_r3)
  - F = 0,8; CR = 0,5; população NP = 10 n; 200 gerações (itermax);
  - cruzamento como no código C do DEoptim: a partir de uma coordenada sorteada, as
    coordenadas seguintes (de forma circular) vêm do vetor mutante enquanto um sorteio
    uniforme ficar abaixo de CR (ao menos uma coordenada sempre vem do mutante);
  - x_best é o melhor indivíduo da geração anterior.
A variante clássica "DE/rand/1/bin" (Storn e Price, 1997) também está disponível.

Referências:
  Storn, R. & Price, K. (1997). Differential Evolution - A Simple and Efficient Heuristic for
  Global Optimization over Continuous Spaces. Journal of Global Optimization, 11, 341-359.
  Mullen, K. M. et al. (2011). DEoptim: An R Package for Global Optimization by Differential
  Evolution. Journal of Statistical Software, 40(6), 1-26.
"""

import numpy as np

# Configuração padrão do DEoptim.control()
DEOPTIM_F = 0.8
DEOPTIM_CR = 0.5
DEOPTIM_ITERMAX = 200


def differential_evolution(func, lower, upper, pop_size=None, F=DEOPTIM_F, CR=DEOPTIM_CR,
                           max_gen=DEOPTIM_ITERMAX, strategy='local-to-best', tol=None,
                           seed=0, x_init=None):
    """
    Minimiza func(x) com lower <= x <= upper.

    strategy: 'local-to-best' (padrão do DEoptim) ou 'rand1bin' (DE clássica).
    tol: se informado, para quando max f - min f na população < tol; se None (padrão do
         DEoptim), roda todas as max_gen gerações.
    x_init: pontos incluídos na população inicial (como o argumento initialpop do DEoptim).

    Em cada geração, para cada indivíduo x_i:
      1. mutação e cruzamento geram o candidato u;
      2. coordenadas de u fora da caixa são sorteadas de novo, uniformemente na caixa;
      3. seleção: x_i <- u se f(u) <= f(x_i).
    """
    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = len(lower)
    if pop_size is None:
        pop_size = 10 * n

    pop = lower + rng.random((pop_size, n)) * (upper - lower)
    if x_init is not None:
        for k, x0 in enumerate(x_init[:pop_size]):
            pop[k] = np.clip(x0, lower, upper)
    fit = np.array([func(x) for x in pop])
    nfev = pop_size

    history = [{'gen': 0, 'f': float(fit.min()), 'nfev': nfev}]
    stop_reason = 'maxgen'
    gen = 0
    for gen in range(1, max_gen + 1):
        best = pop[int(np.argmin(fit))].copy()     # melhor da geração anterior
        old = pop.copy()
        for i in range(pop_size):
            others = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(others, size=3, replace=False)

            # 1. Mutação
            if strategy == 'local-to-best':
                v = old[i] + F * (best - old[i]) + F * (old[r2] - old[r3])
            elif strategy == 'rand1bin':
                v = old[r1] + F * (old[r2] - old[r3])
            else:
                raise ValueError(f"estratégia desconhecida: {strategy}")

            # 1b. Cruzamento (como no DEoptim): bloco circular de coordenadas a partir de j0
            u = old[i].copy()
            j = int(rng.integers(n))
            k = 0
            while True:
                u[j] = v[j]
                j = (j + 1) % n
                k += 1
                if not (rng.random() < CR and k < n):
                    break

            # 2. Coordenadas fora da caixa: novo sorteio uniforme dentro dos limites
            out = (u < lower) | (u > upper)
            if out.any():
                u[out] = lower[out] + rng.random(int(out.sum())) * (upper[out] - lower[out])

            # 3. Seleção
            fu = func(u)
            nfev += 1
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu

        history.append({'gen': gen, 'f': float(fit.min()), 'nfev': nfev})
        if tol is not None and fit.max() - fit.min() < tol:
            stop_reason = 'tol'
            break

    b = int(np.argmin(fit))
    spread = float(fit.max() - fit.min())
    return {'x': pop[b].copy(), 'f': float(fit[b]), 'iter': gen, 'nfev': nfev,
            'ngeval': 0, 'norm_gp': float('nan'), 'history': history,
            'converged': stop_reason == 'tol', 'stop_reason': stop_reason,
            'spread': spread,
            'message': f"DE ({strategy}): {gen} gerações, parada por {stop_reason}."}
