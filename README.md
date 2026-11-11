# Calibração do HYMOD na bacia do Córrego do Monjolinho (São Carlos, SP)

Trabalho de SME0212/SME5720 – Otimização Não-Linear (ICMC-USP, 2º semestre de 2026).

O código calibra os parâmetros do modelo hidrológico HYMOD minimizando −NSE com restrições
de caixa. São comparados três métodos implementados do zero em Python:

- **SPG**: Gradiente Espectral Projetado (`src/spg.py`)
- **BFGS-Box**: restrições ativas com direção quase-Newton BFGS (`src/bfgs_box.py`)
- **Evolução Diferencial**: método global sem derivadas, usado como referência (`src/de.py`),
  com a configuração padrão do pacote DEoptim do R usado no artigo-base
  (estratégia local-to-best, F = 0,8, CR = 0,5, população 10n, 200 gerações)

## Requisitos

- Python 3.10 ou superior
- Pacotes listados em `requirements.txt` (NumPy, Matplotlib e Pillow)

```bash
python3 -m venv venv
source venv/bin/activate          # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

A partir da pasta `Trabalho/`:

```bash
python3 src/main.py                 # experimentos principais (~5 a 7 min)
python3 src/experimentos_extras.py  # análises complementares (~30 s; rodar depois do main.py)
```

O `main.py`:

1. lê os quatro eventos em `dados/evento_1.csv` … `dados/evento_4.csv`;
2. calibra, em cada evento, o **Cenário 1** (W2 = 0, 5 parâmetros) e o **Cenário 2**
   (W2 calibrado, 6 parâmetros), com variáveis normalizadas em [0, 1]:
   - SPG e BFGS-Box a partir de 5 pontos iniciais (o padrão e mais 4 sorteados com semente fixa 2026);
   - Evolução Diferencial com 3 sementes (1, 2, 3);
   - no Cenário 2, cada método também parte da própria solução do Cenário 1;
3. salva em `resultados/`:
   - `tabela_resumo.md`: tabela de NSE e W2 ótimo por evento e método;
   - `resultados_benchmark.json`: parâmetros, iterações, número de simulações, tempos e
     motivo de parada de **cada** ponto inicial ou semente;
   - `evento_<i>_w2_0.png/.pdf` e `evento_<i>_w2_otimo.png/.pdf`: hidrogramas por método;
   - `hidrogramas_resumo.png/.pdf`: melhor ajuste de cada cenário nos 4 eventos;
   - `convergencia_metodos.png/.pdf`: melhor NSE vs. número de simulações e ‖gP‖∞ por iteração
     (Evento 1, Cenário 2);
4. copia as figuras PNG para `relatorio/Template/imagens/` (pasta do relatório, criada se não existir).

O `experimentos_extras.py`:
- calcula o perfil do NSE em função de W2, com os demais parâmetros recalibrados
  (`resultados/perfil_w2.png/.pdf`);
- calcula o NSE em função de Ks e no plano (Cmax, β) do Evento 2, para mostrar os parâmetros
  que os dados não determinam (`resultados/identificabilidade.png/.pdf`);
- recalibra o Cenário 2 com a caixa de W2 ampliada para [0, 500] mm;
- salva os números em `resultados/experimentos_extras.json`.

Os parâmetros dos experimentos (tolerância, número máximo de iterações, número de pontos
iniciais e semente) ficam no início de `src/main.py`.

## Dados

Os CSVs em `dados/` (`evento_1.csv` a `evento_4.csv`) já acompanham o repositório e são tudo o
que o código precisa para rodar. Eles foram obtidos a partir das figuras do artigo de
Silva, Benso & Vasconcelos (2026): as vazões foram digitalizadas dos gráficos e a chuva de 5 min
foi reconstruída preservando o total de cada evento.

Os recortes das figuras do artigo usados nessa digitalização **não estão no repositório**, por
serem material do artigo. Por isso, o script `src/extract_pdf_data.py`, que gera os CSVs a partir
desses recortes, só funciona com eles em `dados/figuras_artigo/`; ele é mantido apenas para
documentar como os dados foram obtidos.

## Estrutura

```
src/hymod.py             modelo HYMOD, NSE, função objetivo e gradiente por diferenças finitas
src/spg.py               Gradiente Espectral Projetado para caixas
src/bfgs_box.py          restrições ativas + BFGS nas variáveis livres
src/de.py                Evolução Diferencial (configuração padrão do DEoptim)
src/experimentos_extras.py  perfis de W2 e Ks, caixa de W2 ampliada
src/instances.py         leitura dos eventos
src/extract_pdf_data.py  digitalização dos dados (precisa dos recortes das figuras, fora do repositório)
src/main.py              experimentos principais, tabelas e gráficos
dados/                   eventos 1 a 4 em CSV (chuva, vazão observada e simulações do artigo)
resultados/              tabelas, JSONs e figuras gerados pelos scripts
```
