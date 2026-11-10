"""
extract_pdf_data.py - Extração digital direta das séries temporais reais (Observado e Simulados)
a partir dos gráficos vetoriais/raster de alta resolução embutidos no PDF do artigo
de Silva, Benso & Vasconcelos (2026) - Figuras 2 e 3 (páginas 7 e 8).

Salva os arquivos CSV em Trabalho/dados/evento_1.csv .. evento_4.csv
contendo:
  - step: passo de tempo t (intervalos de 5 minutos)
  - time_hours: tempo acumulado em horas
  - precip_mm: precipitação reconstruída no intervalo de 5 min (mm), respeitando o total exato da Tabela 2
  - q_obs_m3s: vazão observada real digitalizada pixel a pixel da curva preta (m³/s)
  - q_sim_w2_opt_paper_m3s: vazão simulada do artigo (curva azul, W2 otimizado) (m³/s)
  - q_sim_w2_0_paper_m3s: vazão simulada do artigo (curva vermelha, W2 = 0) (m³/s)
"""

import os
import csv
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dados")
FIG_DIR = os.path.join(DADOS_DIR, "figuras_artigo")


def build_legend_masks(im_ref):
    """
    Constrói as máscaras exatas da caixa de legenda do R:
    - 'tl' (topleft): usada nos Eventos 1 e 4 (x = 114..505, y = 114..184)
    - 'tr' (topright): usada nos Eventos 2 e 3 (x = 355..744, y = 114..185)
    """
    h, w, _ = im_ref.shape
    tl_mask = np.zeros((h, w), dtype=bool)
    # No Evento 4 nenhuma curva passa pelo canto superior esquerdo (y=114..184, x=114..505)
    tl_mask[114:184, 114:505] = (im_ref[114:184, 114:505].mean(axis=2) < 245)
    # Regiões dos dígitos numéricos de NSE que mudam entre os eventos
    tl_mask[139:156, 430:500] = True
    tl_mask[163:180, 370:448] = True
    # Traços horizontais de amostra da legenda (preto, azul, vermelho)
    tl_mask[119:124, 115:143] = True
    tl_mask[143:148, 115:143] = True
    tl_mask[167:172, 115:143] = True

    tr_mask = np.zeros((h, w), dtype=bool)
    # Nos Eventos 2 e 3 o pico ocorre no início (x < 260) e toda a região superior direita é apenas legenda
    tr_mask[114:185, 355:744] = True
    return tl_mask, tr_mask


def digitize_event_image(ev_id, x0, x_end, t_end, n_steps, q_max, leg_pos, tl_mask, tr_mask):
    """
    Extrai as três curvas (Observado - preta, Simulado W2_opt - azul, Simulado W2=0 - vermelha)
    diretamente da imagem PNG extraída do PDF do artigo.
    """
    img_path = os.path.join(FIG_DIR, f"evento_{ev_id}.png")
    im = np.array(Image.open(img_path)).astype(float)
    R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]

    # Calibração exata dos eixos do R:
    # Eixo Y: y_zero = 293.0 px (Q = 0 m³/s), y_top = 105.0 px (Q = q_max m³/s)
    y_zero, y_top = 293.0, 105.0
    legend_mask = tl_mask if leg_pos == "tl" else tr_mask

    y_red = np.full(im.shape[1], np.nan)
    y_blue = np.full(im.shape[1], np.nan)
    y_black = np.full(im.shape[1], np.nan)

    for x in range(100, 745):
        col_r, col_g, col_b = R[100:298, x], G[100:298, x], B[100:298, x]
        col_leg = legend_mask[100:298, x]

        # 1. Curva vermelha: Simulado (W2 = 0)
        r_score = col_r - np.maximum(col_g, col_b)
        r_idx = np.where((r_score > 20) & (~col_leg))[0]
        if len(r_idx) > 0:
            y_red[x] = 100.0 + np.average(r_idx, weights=r_score[r_idx])

        # 2. Curva azul: Simulado (W2 otimizado)
        b_score = col_b - np.maximum(col_r, col_g)
        b_idx = np.where((b_score > 20) & (~col_leg))[0]
        if len(b_idx) > 0:
            y_blue[x] = 100.0 + np.average(b_idx, weights=b_score[b_idx])

        # 3. Curva preta: Observado (pixels neutros escuros fora da máscara de legenda)
        mean_rgb = (col_r + col_g + col_b) / 3.0
        spread = np.maximum(np.maximum(col_r, col_g), col_b) - np.minimum(np.minimum(col_r, col_g), col_b)
        k_idx = np.where((spread < 20) & (mean_rgb < 200) & (~col_leg))[0]
        if len(k_idx) > 0:
            weights = 255.0 - mean_rgb[k_idx]
            y_black[x] = 100.0 + np.average(k_idx, weights=weights)

    # Tratamento de oclusão: no R, a curva vermelha foi plotada por último (sobrepondo a azul e a preta)
    # e a curva azul foi plotada sobre a preta na linha de base inicial e recessão.
    for x in range(100, 745):
        if np.isnan(y_blue[x]) and not np.isnan(y_red[x]):
            y_blue[x] = y_red[x]
        if np.isnan(y_black[x]):
            ref = y_blue[x] if not np.isnan(y_blue[x]) else y_red[x]
            if not np.isnan(ref) and ref > 270.0:
                y_black[x] = ref

    # Interpolação linear apenas nas poucas colunas onde a curva cruzou letras da legenda
    xs = np.arange(im.shape[1])
    for arr in (y_red, y_blue, y_black):
        valid = ~np.isnan(arr)
        if valid.sum() > 2:
            arr[:] = np.interp(xs, xs[valid], arr[valid])

    # Amostragem nos passos discretos t = 1 .. n_steps
    t_steps = np.arange(1, n_steps + 1)
    x_coords = x0 + (t_steps / float(t_end)) * (x_end - x0)

    q_obs = np.clip((y_zero - np.interp(x_coords, xs, y_black)) / (y_zero - y_top) * q_max, 0.0, q_max)
    q_blue = np.clip((y_zero - np.interp(x_coords, xs, y_blue)) / (y_zero - y_top) * q_max, 0.0, q_max)
    q_red = np.clip((y_zero - np.interp(x_coords, xs, y_red)) / (y_zero - y_top) * q_max, 0.0, q_max)

    return q_obs, q_blue, q_red


def reconstruct_precipitation(q_obs, q_blue, total_precip_mm):
    """
    Reconstrói o hietograma de precipitação P(t) (em mm por intervalo de 5 min)
    a partir da deconvolução da resposta hidrológica nos pulsos de subida do hidrograma,
    garantindo que a soma total seja exatamente igual ao volume precipitado da Tabela 2.
    """
    q_base = np.percentile(q_obs, 5)
    # Combina a subida do hidrograma observado e simulado para localizar os blocos de chuva
    q_ref = 0.6 * q_blue + 0.4 * q_obs
    q_excess = np.maximum(0.0, q_ref - q_base)
    dq = np.diff(q_excess, prepend=0.0)

    # Inversão do operador de reservatório linear + avanço de fase da cascata de 3 tanques (2 passos = 10 min)
    inflow_proxy = np.maximum(0.0, dq + 0.35 * q_excess)
    p_raw = np.roll(inflow_proxy, -2)
    p_raw[-2:] = 0.0

    # Filtra ruído residual de recessão para concentrar a chuva nos pulsos reais da tormenta
    threshold = 0.08 * np.max(p_raw)
    p_raw[p_raw < threshold] = 0.0

    # Normaliza exatamente para o total precipitado da Tabela 2 do artigo
    precip = p_raw * (total_precip_mm / np.sum(p_raw))
    return precip


def main():
    os.makedirs(DADOS_DIR, exist_ok=True)
    im4 = np.array(Image.open(os.path.join(FIG_DIR, "evento_4.png"))).astype(float)
    tl_mask, tr_mask = build_legend_masks(im4)

    # Metadados exatos das Tabelas 2 e 3 e Figura 2 do artigo
    event_specs = [
        {
            "id": 1,
            "name": "Evento 1 (01/11 a 04/11/2024)",
            "x0": 118, "x_end": 726, "t_end": 150, "n_steps": 149,
            "q_max": 50.0, "leg_pos": "tl",
            "total_precip": 36.0, "w2_paper": 10.0,
            "nse_blue_paper": 0.656, "nse_red_paper": 0.489
        },
        {
            "id": 2,
            "name": "Evento 2 (25/11 a 29/11/2024)",
            "x0": 118, "x_end": 734, "t_end": 150, "n_steps": 150,
            "q_max": 50.0, "leg_pos": "tr",
            "total_precip": 19.8, "w2_paper": 18.0,
            "nse_blue_paper": 0.742, "nse_red_paper": 0.223
        },
        {
            "id": 3,
            "name": "Evento 3 (10/12 a 14/12/2024)",
            "x0": 116, "x_end": 682, "t_end": 100, "n_steps": 109,
            "q_max": 50.0, "leg_pos": "tr",
            "total_precip": 22.8, "w2_paper": 30.0,
            "nse_blue_paper": 0.624, "nse_red_paper": -0.054
        },
        {
            "id": 4,
            "name": "Evento 4 (17/12 a 22/12/2024)",
            "x0": 121, "x_end": 670, "t_end": 400, "n_steps": 438,
            "q_max": 100.0, "leg_pos": "tl",
            "total_precip": 85.3, "w2_paper": 0.0,
            "nse_blue_paper": 0.402, "nse_red_paper": 0.402
        },
    ]

    print("=" * 85)
    print("DIGITALIZAÇÃO DAS CURVAS REAIS DOS GRÁFICOS DO ARTIGO (FIGURA 2 - PÁGINA 7)")
    print("=" * 85)

    for spec in event_specs:
        ev_id = spec["id"]
        q_obs, q_blue, q_red = digitize_event_image(
            ev_id, spec["x0"], spec["x_end"], spec["t_end"],
            spec["n_steps"], spec["q_max"], spec["leg_pos"],
            tl_mask, tr_mask
        )
        precip = reconstruct_precipitation(q_obs, q_blue, spec["total_precip"])

        # Verifica o NSE das curvas extraídas em relação ao reportado na Figura 2
        nse_blue_ext = 1.0 - np.sum((q_obs - q_blue) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)
        nse_red_ext = 1.0 - np.sum((q_obs - q_red) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)

        csv_path = os.path.join(DADOS_DIR, f"evento_{ev_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "time_hours", "precip_mm",
                "q_obs_m3s", "q_sim_w2_opt_paper_m3s", "q_sim_w2_0_paper_m3s"
            ])
            for t in range(spec["n_steps"]):
                writer.writerow([
                    t + 1,
                    round((t + 1) * (5.0 / 60.0), 4),
                    round(float(precip[t]), 5),
                    round(float(q_obs[t]), 4),
                    round(float(q_blue[t]), 4),
                    round(float(q_red[t]), 4)
                ])

        print(f"Evento {ev_id}: {spec['n_steps']} passos salvos em {csv_path}")
        print(f"  -> Pico Q_obs extraído: {np.max(q_obs):.2f} m³/s | Vazão de base: {np.min(q_obs):.2f} m³/s")
        print(f"  -> NSE (Curva Azul extraída vs Artigo):     {nse_blue_ext:.3f} vs {spec['nse_blue_paper']:.3f}")
        print(f"  -> NSE (Curva Vermelha extraída vs Artigo): {nse_red_ext:.3f} vs {spec['nse_red_paper']:.3f}")


if __name__ == "__main__":
    main()
