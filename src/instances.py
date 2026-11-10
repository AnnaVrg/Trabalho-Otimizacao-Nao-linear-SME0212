"""
instances.py - Carrega as séries temporais reais digitalizadas dos eventos de chuva e vazão
da bacia do Córrego do Monjolinho (São Carlos, SP).

Os dados observados (q_obs) e simulados de referência do artigo são lidos diretamente dos
arquivos CSV em Trabalho/dados/evento_1.csv .. evento_4.csv, extraídos pixel a pixel
das Figuras 2 e 3 (páginas 7 e 8) e Tabelas 2 e 3 do artigo de Silva, Benso & Vasconcelos (2026).
"""

import os
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dados")

EVENT_METADATA = {
    1: {
        "id": 1,
        "name": "Evento 1 (01/11 a 04/11/2024)",
        "date": "01/11 - 04/11/2024",
        "total_precip": 36.0,
        "precip_5days_prior": 0.0,
        "w2_paper": 10.0,
        "nse_w2_0_paper": 0.489,
        "nse_w2_opt_paper": 0.656,
    },
    2: {
        "id": 2,
        "name": "Evento 2 (25/11 a 29/11/2024)",
        "date": "25/11 - 29/11/2024",
        "total_precip": 19.8,
        "precip_5days_prior": 8.8,
        "w2_paper": 18.0,
        "nse_w2_0_paper": 0.223,
        "nse_w2_opt_paper": 0.742,
    },
    3: {
        "id": 3,
        "name": "Evento 3 (10/12 a 14/12/2024)",
        "date": "10/12 - 14/12/2024",
        "total_precip": 22.8,
        "precip_5days_prior": 2.3,
        "w2_paper": 30.0,
        "nse_w2_0_paper": -0.054,
        "nse_w2_opt_paper": 0.624,
    },
    4: {
        "id": 4,
        "name": "Evento 4 (17/12 a 22/12/2024)",
        "date": "17/12 - 22/12/2024",
        "total_precip": 85.3,
        "precip_5days_prior": 18.5,
        "w2_paper": 0.0,
        "nse_w2_0_paper": 0.402,
        "nse_w2_opt_paper": 0.402,
    },
}


def load_event_from_csv(ev_id):
    """
    Carrega os dados reais digitalizados do arquivo Trabalho/dados/evento_<ev_id>.csv.
    Caso o CSV ainda não exista, executa extract_pdf_data.main() automaticamente.
    """
    csv_path = os.path.join(DADOS_DIR, f"evento_{ev_id}.csv")
    if not os.path.exists(csv_path):
        import extract_pdf_data
        extract_pdf_data.main()

    precip = []
    q_obs = []
    q_sim_w2_opt_paper = []
    q_sim_w2_0_paper = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            precip.append(float(row["precip_mm"]))
            q_obs.append(float(row["q_obs_m3s"]))
            q_sim_w2_opt_paper.append(float(row["q_sim_w2_opt_paper_m3s"]))
            q_sim_w2_0_paper.append(float(row["q_sim_w2_0_paper_m3s"]))

    meta = dict(EVENT_METADATA[ev_id])
    meta["time_steps"] = len(q_obs)
    meta["precip"] = np.array(precip, dtype=float)
    meta["q_obs"] = np.array(q_obs, dtype=float)
    meta["q_sim_w2_opt_paper"] = np.array(q_sim_w2_opt_paper, dtype=float)
    meta["q_sim_w2_0_paper"] = np.array(q_sim_w2_0_paper, dtype=float)
    meta["csv_path"] = csv_path
    return meta


def generate_event_1():
    return load_event_from_csv(1)


def generate_event_2():
    return load_event_from_csv(2)


def generate_event_3():
    return load_event_from_csv(3)


def generate_event_4():
    return load_event_from_csv(4)


def get_all_events():
    """Retorna a lista dos quatro eventos reais digitalizados do artigo."""
    return [load_event_from_csv(i) for i in (1, 2, 3, 4)]
