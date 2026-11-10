"""
experimentos_extras.py - Análises complementares (rodar DEPOIS de main.py).

1. Perfil de W2 (Cenário 2): melhor NSE com W2 fixo entre 0 e 300 mm, demais
   parâmetros recalibrados. Mostra que a restrição W2 <= 60 está ativa.
   Figura: perfil_w2.png.
2. Identificabilidade (Cenário 2): NSE em função de Ks (0,1 a 50 passos) com os demais
   parâmetros no ótimo, e NSE no plano (Cmax, beta) no Evento 2. Mostram que Ks não
   altera o ajuste quando alpha = 1 e que Cmax e beta se compensam.
   Figura: identificabilidade.png.
3. Recalibração do Cenário 2 com a caixa de W2 ampliada para [0, 500] mm
   (BFGS-Box, mesmos pontos iniciais + a solução com a caixa original).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import hymod
import bfgs_box
import instances
import main as M

W2_MAX_AMPLIADO = 500.0


def run():
    with open(os.path.join(M.RESULTS_DIR, "resultados_benchmark.json"), encoding="utf-8") as f:
        bench = {r['evento_id']: r for r in json.load(f)}
    events = instances.get_all_events()

    w2_grid = np.arange(0.0, 320.0, 20.0)
    ks_grid = np.linspace(0.1, 50.0, 200)
    profiles = {}
    extras = []

    lower_amp = hymod.LOWER_BOUNDS.copy()
    upper_amp = hymod.UPPER_BOUNDS.copy()
    upper_amp[5] = W2_MAX_AMPLIADO
    rng = np.random.default_rng(M.SEED)

    for ev in events:
        ev_id = ev['id']
        P, Q = ev['precip'], ev['q_obs']
        qb = float(Q[0])
        f = lambda th: hymod.objective_function(th, P, Q, q_base=qb)
        th_opt = np.array(bench[ev_id]['bfgs_cen2']['theta'])

        # 1a. Perfil de W2: para cada W2 fixo, recalibram-se os outros 5 parâmetros
        #     (BFGS-Box, partindo do ponto padrão e da solução do W2 anterior).
        nse_w2 = []
        prev = th_opt[:5]
        for w in w2_grid:
            fw = lambda th5, w=w: f(np.r_[th5, w])
            r = M.multistart(bfgs_box.solve_bfgs_box, fw, hymod.LOWER_BOUNDS[:5],
                             hymod.UPPER_BOUNDS[:5], [M.THETA0_DEFAULT[:5], prev])
            prev = np.array(r['theta'])
            nse_w2.append(r['nse'])
        # 1b. Perfil de Ks: demais parâmetros fixos no ótimo do Cenário 2
        nse_ks = []
        for k in ks_grid:
            th = th_opt.copy()
            th[4] = k
            nse_ks.append(-f(th))
        s_max = th_opt[0] / (th_opt[1] + 1.0)
        profiles[ev_id] = (np.array(nse_w2), np.array(nse_ks), s_max)

        # 2. Recalibração com W2 em [0, 500]
        starts = [M.THETA0_DEFAULT] + M.random_starts(rng, lower_amp, upper_amp, M.N_RANDOM_STARTS)
        res = M.multistart(bfgs_box.solve_bfgs_box, f, lower_amp, upper_amp, starts + [th_opt])
        th_new = res['theta']
        extras.append({
            'evento_id': ev_id,
            'nse_caixa_original': bench[ev_id]['bfgs_cen2']['nse'],
            'nse_caixa_ampliada': res['nse'],
            'theta_caixa_ampliada': th_new,
            'smax_caixa_ampliada': th_new[0] / (th_new[1] + 1.0),
            'w2_satura_em_smax': bool(th_new[5] >= th_new[0] / (th_new[1] + 1.0) - 1e-6),
            'ks_perfil_min': float(min(nse_ks)), 'ks_perfil_max': float(max(nse_ks)),
            'smax_otimo_original': float(s_max),
            'perfil_w2': [float(v) for v in w2_grid],
            'perfil_nse': [float(v) for v in nse_w2],
            'nse_w2_max_no_perfil': float(max(nse_w2)),
            'w2_argmax_perfil': float(w2_grid[int(np.argmax(nse_w2))]),
            'n_conv': res['n_conv'], 'n_runs': res['n_runs'],
        })
        e = extras[-1]
        print(f"Evento {ev_id}: NSE caixa original {e['nse_caixa_original']:.4f} -> caixa ampliada "
              f"{e['nse_caixa_ampliada']:.4f} | theta = {np.round(th_new, 2)} | Smax = {e['smax_caixa_ampliada']:.1f} "
              f"| perfil Ks: [{e['ks_perfil_min']:.6f}, {e['ks_perfil_max']:.6f}]")

    fmt1 = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ','))
    fmt2 = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ','))
    shades = ['#08306b', '#2171b5', '#6baed6', '#9ecae1']

    # Figura 1: perfil de W2 (Seção do limite superior de W2)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (ev_id, (nw, nk, smax)), c in zip(sorted(profiles.items()), shades):
        ax.plot(w2_grid, nw, '-o', color=c, linewidth=1.8, markersize=3, label=f'Evento {ev_id}')
    ax.axvspan(60, w2_grid[-1], color='0.93', zorder=0)
    ax.axvline(60, color='0.4', linestyle='--', linewidth=1.0)
    ax.text(70, 0.33, 'fora da caixa original (W2 > 60 mm)', fontsize=9, color='0.3')
    ax.set_xlabel('W2 fixo (mm)')
    ax.set_ylabel('Melhor NSE\n(demais parâmetros recalibrados)')
    ax.set_title('Perfil de W2: o NSE continua subindo depois de 60 mm')
    ax.set_ylim(0.3, 0.95)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    ax.yaxis.set_major_formatter(fmt1)
    plt.tight_layout()
    M.save_fig("perfil_w2")

    # Figura 2: identificabilidade (Seção de parâmetros calibrados)
    #   (a) NSE em função de Ks, demais parâmetros no ótimo do Cenário 2 (alpha = 1);
    #   (b) NSE no plano (Cmax, beta) para o Evento 2, demais fixos no ótimo do BFGS-Box.
    ev2 = next(e for e in events if e['id'] == 2)
    f2 = lambda th: hymod.objective_function(th, ev2['precip'], ev2['q_obs'], q_base=float(ev2['q_obs'][0]))
    th2 = np.array(bench[2]['bfgs_cen2']['theta'])
    cm_grid = np.linspace(50, 500, 121)
    be_grid = np.linspace(0.1, 2.0, 191)
    Z = np.empty((len(be_grid), len(cm_grid)))
    for i, b in enumerate(be_grid):
        for j, c in enumerate(cm_grid):
            th = th2.copy()
            th[0], th[1] = c, b
            Z[i, j] = -f2(th)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for (ev_id, (nw, nk, smax)), c in zip(sorted(profiles.items()), shades):
        ax1.plot(ks_grid, nk, color=c, linewidth=1.8, label=f'Evento {ev_id}')
    ax1.set_xlabel('Ks (passos de 5 min)')
    ax1.set_ylabel('NSE')
    ax1.set_title('(a) NSE em função de Ks (α = 1): constante')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='center right', fontsize=9)
    ax1.yaxis.set_major_formatter(fmt2)

    levels = [0.5, 0.7, 0.8, 0.85, 0.88]
    cs = ax2.contour(cm_grid, be_grid, np.clip(Z, 0.3, None), levels=levels, cmap='Blues', linewidths=1.3)
    ax2.clabel(cs, fmt=lambda v: f'{v:.2f}'.replace('.', ','), fontsize=8)
    # Fundo do vale: para cada Cmax, o beta de maior NSE
    valley = be_grid[np.argmax(Z, axis=0)]
    good = Z.max(axis=0) > 0.8
    ax2.plot(cm_grid[good], valley[good], color='0.35', linestyle='--', linewidth=1.2, label='fundo do vale (melhor β para cada Cmax)')
    th_spg = bench[2]['spg_cen2']['theta']
    ax2.plot([th2[0]], [th2[1]], 's', color='tab:red', markersize=8, label='BFGS-Box e DE', clip_on=False, zorder=5)
    ax2.plot([th_spg[0]], [th_spg[1]], 'o', color='tab:blue', markersize=8, label='SPG', zorder=5)
    ax2.set_xlabel('Cmax (mm)')
    ax2.set_ylabel('β')
    ax2.yaxis.set_major_formatter(fmt2)
    ax2.set_title('(b) NSE no plano (Cmax, β), Evento 2: um vale estreito')
    ax2.legend(loc='upper left', fontsize=9)
    valley_nse = Z.max(axis=0)
    extras_vale = {'cmax': [float(v) for v in cm_grid], 'beta_vale': [float(v) for v in valley], 'nse_vale': [float(v) for v in valley_nse]}
    ax2.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    M.save_fig("identificabilidade")

    with open(os.path.join(M.RESULTS_DIR, "experimentos_extras.json"), "w", encoding="utf-8") as f:
        json.dump({'eventos': extras, 'vale_cmax_beta_evento2': extras_vale}, f, indent=2, ensure_ascii=False)
    print(f"Salvo em {M.RESULTS_DIR}: experimentos_extras.json, perfil_w2.png, identificabilidade.png")


if __name__ == "__main__":
    run()
