"""
main.py - Script principal de execução dos experimentos e geração de resultados.

Calibra o HYMOD nos 4 eventos do Córrego do Monjolinho (São Carlos, SP), nos cenários
W2 = 0 (Cenário 1) e W2 calibrado (Cenário 2), com três métodos implementados do zero:
  - SPG       (Gradiente Espectral Projetado, src/spg.py)
  - BFGS-Box  (restrições ativas + quase-Newton BFGS, src/bfgs_box.py)
  - DE        (Evolução Diferencial, src/de.py) - método global usado no artigo-base
Gera tabelas e gráficos para o relatório e para a apresentação.
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# Garante importação dos módulos locais
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import hymod
import spg
import bfgs_box
import de
import instances

PROJECT_DIR = os.path.dirname(script_dir)
RESULTS_DIR = os.path.join(PROJECT_DIR, "resultados")
REPORT_IMG_DIR = os.path.join(PROJECT_DIR, "relatorio", "Template", "imagens")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# =============================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS
# =============================================================================
EPS = 1e-3           # tolerância de ||gP||_inf (no espaço normalizado [0, 1]^n)
MAX_ITER = 200       # limite de iterações por execução de SPG / BFGS-Box
N_RANDOM_STARTS = 4  # pontos iniciais aleatórios adicionais (além do ponto padrão)
SEED = 2026          # semente fixa -> experimentos reprodutíveis

DE_SEEDS = (1, 2, 3)  # sementes das execuções independentes da Evolução Diferencial
# Evolução Diferencial com a configuração padrão do DEoptim (pacote usado no artigo):
# estratégia local-to-best, F = 0,8, CR = 0,5, população 10 n e 200 gerações (ver de.py).
DE_SPREAD_TOL = 1e-6  # população "concentrada" ao final: max f - min f < DE_SPREAD_TOL

# Ponto inicial padrão theta = [cmax, beta, alpha, kq, ks, w2]
THETA0_DEFAULT = np.array([250.0, 1.0, 0.5, 5.0, 15.0, 10.0])

METHODS = ('spg', 'bfgs', 'de')
METHOD_LABEL = {'spg': 'SPG', 'bfgs': 'BFGS-Box', 'de': 'Evolução Diferencial'}
METHOD_STYLE = {'spg': dict(color='tab:blue', linestyle='-', linewidth=1.8),
                'bfgs': dict(color='tab:red', linestyle='--', linewidth=1.8),
                'de': dict(color='tab:green', linestyle=':', linewidth=2.2)}


def to_theta(z, lower, upper):
    """Mudança de variáveis: z em [0,1]^n  ->  theta = l + z (u - l)."""
    return lower + z * (upper - lower)


def to_z(theta, lower, upper):
    """Mudança inversa: theta -> z = (theta - l) / (u - l)."""
    return (np.asarray(theta, dtype=float) - lower) / (upper - lower)


class CountedObjective:
    """
    F(z) = f(l + z(u - l)) no espaço normalizado, contando TODAS as simulações do HYMOD
    (inclusive as usadas no gradiente por diferenças finitas) e guardando o melhor valor
    encontrado após cada simulação (para comparar métodos por custo computacional).
    """

    def __init__(self, f_theta, lower, upper, track=False):
        self.f_theta, self.lower, self.upper = f_theta, lower, upper
        self.count = 0
        self.best = np.inf
        self.track = track
        self.trace = []

    def __call__(self, z):
        val = self.f_theta(to_theta(z, self.lower, self.upper))
        self.count += 1
        if val < self.best:
            self.best = val
        if self.track:
            self.trace.append(self.best)
        return val

    def grad(self, z):
        n = len(z)
        return hymod.compute_gradient_fd(self, z, lower=np.zeros(n), upper=np.ones(n))


def run_record(res, lower, upper, dt, nsim, extra=None):
    rec = {
        'nse': float(-res['f']), 'iter': int(res['iter']),
        'nsim': int(nsim), 'tempo': float(dt), 'stop_reason': res['stop_reason'],
        'norm_gp': float(res['norm_gp']),
        'theta': [float(v) for v in to_theta(res['x'], lower, upper)],
    }
    if extra:
        rec.update(extra)
    return rec


def summarize(runs):
    best = dict(max(runs, key=lambda r: r['nse']))
    best['tempo_total'] = float(sum(r['tempo'] for r in runs))
    best['nsim_total'] = int(sum(r['nsim'] for r in runs))
    best['n_runs'] = len(runs)
    # SPG/BFGS-Box: parada por ||gP|| <= eps; DE: população concentrada ao final
    best['n_conv'] = int(sum(r['stop_reason'] in ('gp', 'tol') or r.get('spread', 1.0) < DE_SPREAD_TOL
                             for r in runs))
    best['runs'] = runs
    return best


def multistart(solver, f_theta, lower, upper, starts_theta):
    """SPG ou BFGS-Box no problema normalizado, a partir de cada ponto inicial; fica o melhor."""
    n = len(lower)
    runs = []
    for j, th0 in enumerate(starts_theta):
        F = CountedObjective(f_theta, lower, upper)
        t0 = time.time()
        res = solver(F, F.grad, to_z(th0, lower, upper), np.zeros(n), np.ones(n),
                     eps=EPS, max_iter=MAX_ITER)
        runs.append(run_record(res, lower, upper, time.time() - t0, F.count,
                               {'start': j, 'theta0': [float(v) for v in th0]}))
    return summarize(runs)


def run_de(f_theta, lower, upper, init_theta=None):
    """Evolução Diferencial no problema normalizado, uma execução por semente; fica a melhor."""
    n = len(lower)
    runs = []
    x_init = None if init_theta is None else [to_z(init_theta, lower, upper)]
    for s in DE_SEEDS:
        F = CountedObjective(f_theta, lower, upper)
        t0 = time.time()
        res = de.differential_evolution(F, np.zeros(n), np.ones(n), seed=s, x_init=x_init)
        runs.append(run_record(res, lower, upper, time.time() - t0, F.count,
                               {'seed': s, 'spread': res['spread']}))
    return summarize(runs)


def random_starts(rng, lower, upper, k):
    """k pontos iniciais uniformes na caixa (gerados com semente fixa)."""
    return [lower + rng.random(len(lower)) * (upper - lower) for _ in range(k)]


def plot_hydrograph(ev, sims, title, fname):
    """Chuva (invertida) + vazão observada e simulada por cada método."""
    steps = np.arange(1, ev['time_steps'] + 1)
    fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 2.5]})
    ax_p.bar(steps, ev['precip'], width=0.8, color='royalblue', alpha=0.7,
             edgecolor='navy', label='Chuva (mm)')
    ax_p.set_ylabel('Precip. (mm)')
    ax_p.invert_yaxis()
    ax_p.grid(True, linestyle=':', alpha=0.6)
    ax_p.legend(loc='upper right')
    ax_p.set_title(title)
    ax_q.plot(steps, ev['q_obs'], 'k-', linewidth=2.0, label='Observado')
    for m, (q, nse, w2) in sims.items():
        lab = f'{METHOD_LABEL[m]} - NSE: {br(nse, 3)}'
        if w2 is not None:
            lab += f' (W2 = {br(w2, 1)} mm)'
        ax_q.plot(steps, q, label=lab, **METHOD_STYLE[m])
    ax_q.set_xlabel('Intervalo de tempo (5 min)')
    ax_q.set_ylabel('Vazão (m³/s)')
    ax_q.grid(True, linestyle=':', alpha=0.6)
    ax_q.legend(loc='upper left')
    plt.tight_layout()
    save_fig(fname)


def br(v, nd):
    """Número com vírgula decimal (padrão brasileiro) para rótulos de gráficos."""
    return f"{v:.{nd}f}".replace('.', ',')


def save_fig(fname, copy_to_report=True):
    png = os.path.join(RESULTS_DIR, f"{fname}.png")
    plt.savefig(png, dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, f"{fname}.pdf"))
    plt.close()
    if copy_to_report:
        shutil.copy2(png, os.path.join(REPORT_IMG_DIR, f"{fname}.png"))


def run_benchmark():
    print("=" * 80)
    print("TRABALHO DE OTIMIZAÇÃO NÃO-LINEAR - SME0212 / SME5720 (ICMC-USP)")
    print("Calibração do Modelo HYMOD na Bacia do Córrego do Monjolinho (São Carlos, SP)")
    print("Métodos: SPG, BFGS-Box e Evolução Diferencial")
    print(f"Variáveis normalizadas em [0,1] | eps = {EPS} | max_iter = {MAX_ITER} | "
          f"{1 + N_RANDOM_STARTS} pontos iniciais (semente {SEED}) | DE (config. DEoptim): sementes {DE_SEEDS}")
    print("=" * 80)

    events = instances.get_all_events()
    results_summary = []
    rng = np.random.default_rng(SEED)

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'figure.titlesize': 14})

    lower_6d, upper_6d = hymod.LOWER_BOUNDS, hymod.UPPER_BOUNDS
    lower_5d, upper_5d = hymod.LOWER_BOUNDS[:5], hymod.UPPER_BOUNDS[:5]
    best_sims = {}

    for ev in events:
        ev_id, name = ev['id'], ev['name']
        precip, q_obs = ev['precip'], ev['q_obs']
        q_base = float(q_obs[0])
        print(f"\n>>> {name} | Chuva = {ev['total_precip']} mm | Q_base = {q_base:.2f} m³/s")

        starts_6d = [THETA0_DEFAULT] + random_starts(rng, lower_6d, upper_6d, N_RANDOM_STARTS)
        starts_5d = [s[:5] for s in starts_6d]

        # ---------------- Cenário 1: W2 = 0 (5 parâmetros) ----------------
        def obj_cen1(th5):
            return hymod.objective_function(np.append(th5, 0.0), precip, q_obs, q_base=q_base)

        cen1 = {'spg': multistart(spg.spg_box, obj_cen1, lower_5d, upper_5d, starts_5d),
                'bfgs': multistart(bfgs_box.solve_bfgs_box, obj_cen1, lower_5d, upper_5d, starts_5d),
                'de': run_de(obj_cen1, lower_5d, upper_5d)}

        # ---------------- Cenário 2: W2 calibrado (6 parâmetros) ----------------
        # Cada método também parte da sua própria solução do Cenário 1 (com W2 = 0), que é
        # viável no Cenário 2: assim o NSE do Cenário 2 nunca fica abaixo do Cenário 1.
        def obj_cen2(th6):
            return hymod.objective_function(th6, precip, q_obs, q_base=q_base)

        warm = {m: np.append(cen1[m]['theta'], 0.0) for m in METHODS}
        cen2 = {'spg': multistart(spg.spg_box, obj_cen2, lower_6d, upper_6d, starts_6d + [warm['spg']]),
                'bfgs': multistart(bfgs_box.solve_bfgs_box, obj_cen2, lower_6d, upper_6d,
                                   starts_6d + [warm['bfgs']]),
                'de': run_de(obj_cen2, lower_6d, upper_6d, init_theta=warm['de'])}

        for tag, res in (("Cenário 1 (W2 = 0)", cen1), ("Cenário 2 (W2 calibrado)", cen2)):
            print(f"  [{tag}]")
            for m in METHODS:
                r = res[m]
                w2 = f" | W2* = {r['theta'][5]:.2f} mm" if len(r['theta']) > 5 else ""
                print(f"    {METHOD_LABEL[m]:22s} NSE = {r['nse']:.4f} | convergiram {r['n_conv']}/{r['n_runs']}"
                      f" | {r['nsim_total']:6d} simulações | {r['tempo_total']:6.2f}s{w2}")

        # ---------------- Hidrogramas por evento ----------------
        sims1 = {m: (hymod.hymod_simulate(np.append(cen1[m]['theta'], 0.0), precip, q_base=q_base),
                     cen1[m]['nse'], None) for m in METHODS}
        sims2 = {m: (hymod.hymod_simulate(cen2[m]['theta'], precip, q_base=q_base),
                     cen2[m]['nse'], cen2[m]['theta'][5]) for m in METHODS}
        plot_hydrograph(ev, sims1, f"{name} — Cenário 1 (W2 = 0)", f"evento_{ev_id}_w2_0")
        plot_hydrograph(ev, sims2, f"{name} — Cenário 2 (W2 calibrado)", f"evento_{ev_id}_w2_otimo")

        b1 = max(METHODS, key=lambda m: cen1[m]['nse'])
        b2 = max(METHODS, key=lambda m: cen2[m]['nse'])
        best_sims[ev_id] = (ev, sims1[b1][0], cen1[b1]['nse'], sims2[b2][0], cen2[b2]['nse'])

        results_summary.append({
            'evento_id': ev_id, 'evento_nome': name, 'data': ev['date'],
            'precip_total': ev['total_precip'], 'q_base': q_base,
            'paper_nse_w2_0': ev['nse_w2_0_paper'], 'paper_nse_w2_opt': ev['nse_w2_opt_paper'],
            'paper_w2': ev['w2_paper'],
            **{f'{m}_cen1': cen1[m] for m in METHODS},
            **{f'{m}_cen2': cen2[m] for m in METHODS},
        })

    # ---------------- Figura-resumo: melhor ajuste por evento, nos dois cenários ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, ev_id in zip(axes.flat, sorted(best_sims)):
        ev, q1, n1, q2, n2 = best_sims[ev_id]
        t_h = np.arange(1, ev['time_steps'] + 1) * 5 / 60
        ax.plot(t_h, ev['q_obs'], 'k-', linewidth=1.8, label='Observado')
        ax.plot(t_h, q1, color='0.55', linewidth=1.6, label=f'W2 = 0 (NSE {br(n1, 2)})')
        ax.plot(t_h, q2, color='tab:blue', linewidth=1.6, label=f'W2 calibrado (NSE {br(n2, 2)})')
        ax.set_title(f"Evento {ev_id} ({ev['date']}, {br(ev['total_precip'], 1)} mm)", fontsize=11)
        ax.set_xlabel('Tempo (h)')
        ax.set_ylabel('Vazão (m³/s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    save_fig("hidrogramas_resumo")

    # ---------------- Comparação de convergência (Evento 1, Cenário 2) ----------------
    # SPG e BFGS-Box a partir do ponto padrão; DE com a primeira semente. Eixo x = número de
    # simulações do HYMOD (inclui as do gradiente por diferenças finitas): custo comparável.
    ev1 = events[0]
    qb1 = float(ev1['q_obs'][0])
    f_ev1 = lambda th: hymod.objective_function(th, ev1['precip'], ev1['q_obs'], q_base=qb1)
    lo, up = hymod.LOWER_BOUNDS, hymod.UPPER_BOUNDS
    z0 = to_z(THETA0_DEFAULT, lo, up)
    traces, gp_hist = {}, {}
    for m, solver in (('spg', spg.spg_box), ('bfgs', bfgs_box.solve_bfgs_box)):
        F = CountedObjective(f_ev1, lo, up, track=True)
        res = solver(F, F.grad, z0, np.zeros(6), np.ones(6), eps=EPS, max_iter=MAX_ITER)
        traces[m] = F.trace
        gp_hist[m] = [h['norm_gp'] for h in res['history']]
    F = CountedObjective(f_ev1, lo, up, track=True)
    de.differential_evolution(F, np.zeros(6), np.ones(6), seed=DE_SEEDS[0])
    traces['de'] = F.trace

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in METHODS:
        nse = -np.array(traces[m])
        ax1.plot(np.arange(1, len(nse) + 1), np.maximum(nse, 0.0), label=METHOD_LABEL[m],
                 **METHOD_STYLE[m])
    ax1.set_xscale('log')
    ax1.set_xlabel('Número de simulações do HYMOD (escala log)')
    ax1.set_ylabel('Melhor NSE encontrado')
    ax1.set_ylim(0, 0.9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax1.set_title('Melhor NSE vs. custo (Evento 1, Cenário 2)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    for m in ('spg', 'bfgs'):
        ax2.semilogy(gp_hist[m], label=f'{METHOD_LABEL[m]}', **METHOD_STYLE[m])
    ax2.axhline(EPS, color='gray', linestyle='--', linewidth=1.0, label='Tolerância (10⁻³)')
    ax2.set_xlabel('Iteração (k)')
    ax2.set_ylabel(r'$\|g_P(z_k)\|_\infty$ (escala log)')
    ax2.set_title('Estacionariedade na caixa (variáveis normalizadas)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    save_fig("convergencia_metodos")

    # ---------------- Salva resultados ----------------
    with open(os.path.join(RESULTS_DIR, "resultados_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    hdr = ("| Evento | Chuva (mm) | Cenário | NSE artigo | NSE SPG | NSE BFGS-Box | NSE DE "
           "| W2* SPG | W2* BFGS-Box | W2* DE |")
    lines = [hdr, "|" + ":---:|" * 10]
    for r in results_summary:
        for c, lab, paper in (('cen1', 'W2 = 0', r['paper_nse_w2_0']),
                              ('cen2', 'W2 calibrado', r['paper_nse_w2_opt'])):
            w2 = (lambda m: f"{r[f'{m}_{c}']['theta'][5]:.1f}") if c == 'cen2' else (lambda m: "0")
            lines.append(f"| {r['evento_id']} | {r['precip_total']:.1f} | {lab} | {paper:.3f} | "
                         f"{r[f'spg_{c}']['nse']:.3f} | {r[f'bfgs_{c}']['nse']:.3f} | {r[f'de_{c}']['nse']:.3f} | "
                         f"{w2('spg')} | {w2('bfgs')} | {w2('de')} |")
    md = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "tabela_resumo.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print("\n" + "=" * 80)
    print(md)
    print("=" * 80)
    print(f"Resultados e gráficos salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    run_benchmark()
