"""
Leitura dos dados brutos, aplicação dos critérios de descarte e testes
estatísticos do plano de análise (seção 5 do protocolo).

Nada aqui depende do Streamlit: dá para rodar e conferir em um notebook.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ITENS_C = ["C1", "C2", "C3", "C4"]
ITENS_R = ["R1", "R2", "R3"]
ITENS_D = ["D1", "D2"]
ITENS_LIKERT = ["C1", "C2", "C3", "C4"]                 # concordância
ITENS_FREQ = ["D1", "D2", "R1", "R2", "R3"]              # frequência (R3 foi para a grade de frequência no formulário)
ITENS_ESCALA = ITENS_C + ITENS_D + ITENS_R               # straight-lining (sem A1)
CONSTRUTOS = {
    "M1": ("Clareza dos requisitos", ITENS_C),
    "M2": ("Retrabalho percebido", ITENS_R),
    "M3": ("Instabilidade dos requisitos", ITENS_D),
}

MOTIVOS = {
    "a": "Não aceitou o termo de consentimento",
    "b": "Fora da população-alvo",
    "c": "Errou a pergunta de atenção",
    "d": "Tempo de preenchimento abaixo de 1/3 da mediana do pré-teste",
    "e": "Resposta idêntica em todos os itens de escala (straight-lining)",
    "f": "Duplicata evidente",
}


# ---------------------------------------------------------------------------
# Configuração e leitura
# ---------------------------------------------------------------------------
def carregar_config(caminho: str | Path = "config.yaml") -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ler_bruto(origem, nome: str | None = None) -> pd.DataFrame:
    """Lê CSV ou XLSX (caminho ou arquivo enviado)."""
    nome = (nome or str(origem)).lower()
    if nome.endswith((".xlsx", ".xls")):
        return pd.read_excel(origem)
    try:
        return pd.read_csv(origem, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(origem, "seek"):
            origem.seek(0)
        return pd.read_csv(origem, encoding="latin-1")


def _num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else np.nan


def _escala(serie: pd.Series, mapa: dict) -> pd.Series:
    mapa_norm = {str(k).strip().lower(): v for k, v in mapa.items()}

    def conv(x):
        if pd.isna(x) or str(x).strip() == "":
            return np.nan
        s = str(x).strip().lower()
        if s in mapa_norm:
            return float(mapa_norm[s])
        v = _num(x)
        return v if 1 <= v <= 5 else np.nan

    return serie.map(conv)


def _datahora(serie: pd.Series) -> pd.Series:
    """Aceita o formato do Google Forms ("2026/11/23 2:41:07 PM GMT-3"),
    do Microsoft Forms e datas dd/mm/aaaa."""
    s = serie.astype("string").str.replace(r"\s*GMT[+-]?\d*\s*$", "", regex=True).str.strip()
    for fmt in ("%Y/%m/%d %I:%M:%S %p", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().mean() > 0.9:
            return out
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _norm(nome) -> str:
    return re.sub(r"\s+", " ", str(nome)).strip()


def padronizar(bruto: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Renomeia colunas para os códigos do questionário e converte escalas.
    Remove colunas identificadoras antes de qualquer outra coisa."""
    bruto = bruto.rename(columns=_norm)
    remover = {_norm(c) for c in cfg.get("colunas_remover", [])}
    df = bruto.drop(columns=[c for c in bruto.columns if c in remover])
    cfg = {**cfg, "colunas": {k: (_norm(v) if v else v) for k, v in cfg["colunas"].items()}}
    col = cfg["colunas"]
    val = cfg["valores"]
    out = pd.DataFrame(index=df.index)

    def pega(codigo):
        nome = col.get(codigo)
        return df[nome] if nome and nome in df.columns else pd.Series(np.nan, index=df.index)

    out["id"] = pega("id") if col.get("id") in df.columns else pd.Series(range(1, len(df) + 1), index=df.index)
    out["carimbo"] = _datahora(pega("carimbo"))
    if col.get("duracao_seg") and col["duracao_seg"] in df.columns:
        out["duracao_seg"] = pega("duracao_seg").map(_num)
    elif col.get("inicio") and col["inicio"] in df.columns:
        ini = _datahora(pega("inicio"))
        out["duracao_seg"] = (out["carimbo"] - ini).dt.total_seconds()
    else:
        out["duracao_seg"] = np.nan
    for c in ["canal", "consentimento", "maior_idade", "atua_dev", "colega_turma", "P1", "P3", "P4", "E1", "E2"]:
        out[c] = pega(c).astype("string").str.strip()
    out["P2"] = pega("P2").map(_num)
    # Os dois mapas são aceitos em todos os itens: no formulário oficial,
    # R3 foi apresentado na grade de frequência (Nunca…Sempre), não em concordância.
    ambos = {**val["likert"], **val["frequencia"]}
    for c in ITENS_LIKERT + ITENS_FREQ + ["A1"]:
        out[c] = _escala(pega(c), ambos)

    # Faixas de experiência (M4)
    def faixa(v):
        if pd.isna(v):
            return pd.NA
        for f in cfg["analise"]["faixas_experiencia"]:
            sup = v <= f["max"] if f["inclui_max"] else v < f["max"]
            if v >= f["min"] and sup:
                return f["rotulo"]
        return pd.NA

    rotulos = [f["rotulo"] for f in cfg["analise"]["faixas_experiencia"]]
    mapa_p2 = cfg["valores"].get("p2_categorias")
    if mapa_p2:   # formulário coletou P2 já em faixas (texto)
        texto = pega("P2").astype("string").str.strip()
        out["P2_texto"] = texto
        out["P2"] = np.nan
        out["faixa_exp"] = pd.Categorical(texto.map(mapa_p2), categories=rotulos, ordered=True)
    else:
        out["faixa_exp"] = pd.Categorical(out["P2"].map(faixa), categories=rotulos, ordered=True)
    # Recodificação opcional de categorias (ex.: porte com 4 opções → 3 do protocolo)
    for c, mapa in (cfg["valores"].get("recodificar") or {}).items():
        out[c] = out[c].map(lambda x: mapa.get(x, x) if pd.notna(x) else x).astype("string")
    out["estagiario"] = np.where(
        out["P1"].isna(), pd.NA,
        np.where(out["P1"].isin(val["funcao_estagiario"]), "Estagiário(a)", "Demais funções"))
    out["estagiario"] = pd.Categorical(out["estagiario"], categories=["Estagiário(a)", "Demais funções"], ordered=True)
    for c, ordem in [("P1", "ordem_funcao"), ("P3", "ordem_porte"), ("P4", "ordem_forma")]:
        cats = val.get(ordem, [])
        extras = [x for x in out[c].dropna().unique() if x not in cats]
        out[c] = pd.Categorical(out[c], categories=cats + sorted(extras), ordered=True)

    # Práticas E1 (múltipla escolha) -> colunas booleanas
    for p in val["praticas_e1"]:
        out[f"E1::{p}"] = out["E1"].fillna("").str.contains(re.escape(p), case=False, regex=True)
    return out


# ---------------------------------------------------------------------------
# Descartes (seção 2 do protocolo)
# ---------------------------------------------------------------------------
def aplicar_descartes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Avalia cada critério de forma independente e atribui a cada resposta o
    PRIMEIRO critério violado na ordem (a)→(f) do protocolo, para que a
    contagem não tenha dupla contagem. Retorna (válidas, log, info)."""
    val = cfg["valores"]
    flags = pd.DataFrame(index=df.index)
    info = {}

    flags["a"] = ~df["consentimento"].isin(val["consentimento_aceito"])

    eleg = pd.Series(True, index=df.index)
    for campo, aceitos in val["elegibilidade"].items():
        if df[campo].notna().any():
            eleg &= df[campo].isin(aceitos)
    flags["b"] = ~eleg

    info["atencao_disponivel"] = df["A1"].notna().any()
    flags["c"] = (df["A1"] != float(val["atencao_correta"])) if info["atencao_disponivel"] else False

    limite = cfg["pre_teste"]["mediana_min"] * 60 / 3
    info["limite_tempo_seg"] = limite
    info["tempo_disponivel"] = df["duracao_seg"].notna().any()
    flags["d"] = (df["duracao_seg"] < limite) if info["tempo_disponivel"] else False

    itens = df[ITENS_ESCALA]
    flags["e"] = (itens.notna().all(axis=1)) & (itens.nunique(axis=1) == 1)

    conteudo = [c for c in df.columns if c not in ("id", "carimbo", "duracao_seg", "canal", "P2_texto")]
    chave = df[conteudo].astype(object).where(df[conteudo].notna(), "").map(str).agg("|".join, axis=1)
    flags["f"] = chave.duplicated(keep="first")

    # A pergunta (c) e seguintes não se aplicam a quem não passou de (a)
    flags.loc[flags["a"], ["b", "c", "d", "e", "f"]] = False

    primeiro = flags.apply(lambda r: next((k for k in "abcdef" if r[k]), None), axis=1)
    log = pd.DataFrame({
        "id": df["id"],
        "carimbo": df["carimbo"],
        "criterio": primeiro,
        "motivo": primeiro.map(MOTIVOS),
    })
    info["sobreposicao"] = {k: int(flags[k].sum()) for k in "abcdef"}
    validas = df[primeiro.isna()].copy()
    return validas, log, info


# ---------------------------------------------------------------------------
# Índices e confiabilidade
# ---------------------------------------------------------------------------
def calcular_indices(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    minimos = cfg["analise"]["min_itens_indice"]
    for m, (_, itens) in CONSTRUTOS.items():
        n_ok = df[itens].notna().sum(axis=1)
        df[m] = df[itens].mean(axis=1).where(n_ok >= minimos[m])
    return df


def cronbach(itens: pd.DataFrame) -> float:
    x = itens.dropna()
    k = x.shape[1]
    if len(x) < 3 or k < 2:
        return np.nan
    var_tot = x.sum(axis=1).var(ddof=1)
    if var_tot == 0:
        return np.nan
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / var_tot)


def confiabilidade(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg["analise"]["semente"])
    reps = cfg["analise"]["bootstrap_reps"]
    lim = cfg["analise"]["alfa_cronbach_minimo"]
    linhas, itens_lin = [], []
    for m, (nome, itens) in CONSTRUTOS.items():
        x = df[itens].dropna()
        a = cronbach(x)
        boots = []
        arr = x.to_numpy()
        for _ in range(reps):
            amostra = pd.DataFrame(arr[rng.integers(0, len(arr), len(arr))])
            boots.append(cronbach(amostra))
        boots = np.array([b for b in boots if not np.isnan(b)])
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
        linhas.append({
            "Índice": m, "Construto": nome, "Itens": ", ".join(itens), "k": len(itens),
            "n (casos completos)": len(x), "Alfa de Cronbach": a, "IC95% inf": lo, "IC95% sup": hi,
            "Situação": "Aceitável (≥ %.2f)" % lim if a >= lim else "Abaixo de %.2f → itens isolados" % lim,
        })
        total = x.sum(axis=1)
        for it in itens:
            resto = total - x[it]
            r_it = stats.pearsonr(x[it], resto)[0] if x[it].std() > 0 and resto.std() > 0 else np.nan
            itens_lin.append({
                "Índice": m, "Item": it,
                "Correlação item-total corrigida": r_it,
                "Alfa se o item for removido": cronbach(x.drop(columns=it)) if len(itens) > 2 else np.nan,
            })
    return pd.DataFrame(linhas), pd.DataFrame(itens_lin)


# ---------------------------------------------------------------------------
# Utilitários estatísticos
# ---------------------------------------------------------------------------
def margem_erro(n: int, z: float = 1.96, p: float = 0.5) -> float:
    """Margem de erro em pontos percentuais (pior caso p = 0,5)."""
    return 100 * z * math.sqrt(p * (1 - p) / n) if n > 0 else np.nan


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    m = len(pv)
    ordem = np.argsort(pv)
    aj = np.empty(m)
    acc = 0.0
    for i, idx in enumerate(ordem):
        acc = max(acc, (m - i) * pv[idx])
        aj[idx] = min(1.0, acc)
    out[ok] = aj
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centro - meia, centro + meia


def rotulo_efeito(valor: float, limites=(0.1, 0.3, 0.5)) -> str:
    if valor is None or np.isnan(valor):
        return "—"
    v = abs(valor)
    if v < limites[0]:
        return "desprezível"
    if v < limites[1]:
        return "pequeno"
    if v < limites[2]:
        return "médio"
    return "grande"


# ---------------------------------------------------------------------------
# QP1 — Spearman com IC bootstrap
# ---------------------------------------------------------------------------
def spearman_boot(x: pd.Series, y: pd.Series, reps: int, semente: int) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 4:
        return {"n": n, "rho": np.nan, "p": np.nan, "ic_inf": np.nan, "ic_sup": np.nan}
    a, b = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    rho, p = stats.spearmanr(a, b)
    rng = np.random.default_rng(semente)
    boots = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if np.ptp(a[i]) == 0 or np.ptp(b[i]) == 0:
            continue
        boots.append(stats.spearmanr(a[i], b[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "rho": rho, "p": p, "ic_inf": lo, "ic_sup": hi}


def qp1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    reps, sem = cfg["analise"]["bootstrap_reps"], cfg["analise"]["semente"]
    testes = [("H1: Clareza (M1) × Retrabalho (M2)", "M1", "M2"),
              ("Complementar: Instabilidade (M3) × Retrabalho (M2)", "M3", "M2")]
    linhas = []
    for nome, x, y in testes:
        r = spearman_boot(df[x], df[y], reps, sem)
        linhas.append({"Teste": nome, "x": x, "y": y, **r})
    out = pd.DataFrame(linhas)
    out["p_holm"] = holm(out["p"])
    out["efeito"] = out["rho"].map(rotulo_efeito)
    out["margem_erro_pp"] = out["n"].map(margem_erro)
    return out


def qp1_itens(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Análise item a item (usada quando o alfa fica abaixo do mínimo)."""
    linhas = []
    for c in ITENS_C:
        for r in ITENS_R:
            d = df[[c, r]].dropna()
            rho, p = stats.spearmanr(d[c], d[r]) if len(d) > 3 else (np.nan, np.nan)
            linhas.append({"Clareza": c, "Retrabalho": r, "n": len(d), "rho": rho, "p": p})
    out = pd.DataFrame(linhas)
    out["p_holm"] = holm(out["p"])
    out["efeito"] = out["rho"].map(rotulo_efeito)
    return out


# ---------------------------------------------------------------------------
# QP2 — Kruskal-Wallis + Dunn/Holm; Mann-Whitney + bisserial por postos
# ---------------------------------------------------------------------------
def kruskal(df: pd.DataFrame, var: str, grupo: str) -> dict:
    d = df[[var, grupo]].dropna()
    grupos = [g[var].to_numpy() for _, g in d.groupby(grupo, observed=True) if len(g) > 0]
    n = len(d)
    if len(grupos) < 2:
        return {"n": n, "k": len(grupos), "H": np.nan, "p": np.nan, "eps2": np.nan}
    H, p = stats.kruskal(*grupos)
    eps2 = H / (n - 1)       # epsilon² = H / ((n² − 1)/(n + 1))
    return {"n": n, "k": len(grupos), "H": H, "p": p, "eps2": eps2}


def dunn(df: pd.DataFrame, var: str, grupo: str) -> pd.DataFrame:
    d = df[[var, grupo]].dropna().copy()
    N = len(d)
    if N < 3:
        return pd.DataFrame()
    d["rank"] = stats.rankdata(d[var])
    _, t = np.unique(d[var], return_counts=True)
    corr_emp = (t ** 3 - t).sum() / (12 * (N - 1))
    resumo = d.groupby(grupo, observed=True)["rank"].agg(["mean", "count"])
    nomes = list(resumo.index)
    if len(nomes) < 2:
        return pd.DataFrame(columns=["Grupo A", "Grupo B", "z", "p", "p_holm"])
    linhas = []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            gi, gj = nomes[i], nomes[j]
            ni, nj = resumo.loc[gi, "count"], resumo.loc[gj, "count"]
            se = math.sqrt((N * (N + 1) / 12 - corr_emp) * (1 / ni + 1 / nj))
            z = (resumo.loc[gi, "mean"] - resumo.loc[gj, "mean"]) / se
            linhas.append({"Grupo A": gi, "Grupo B": gj, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    out = pd.DataFrame(linhas)
    out["p_holm"] = holm(out["p"])
    return out


def mann_whitney(df: pd.DataFrame, var: str, grupo: str, alvo: str) -> dict:
    d = df[[var, grupo]].dropna()
    a = d.loc[d[grupo] == alvo, var].to_numpy()
    b = d.loc[d[grupo] != alvo, var].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return {"n1": len(a), "n2": len(b), "U": np.nan, "p": np.nan, "r_rb": np.nan}
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r_rb = 2 * U / (len(a) * len(b)) - 1     # > 0: grupo-alvo tende a valores maiores
    return {"n1": len(a), "n2": len(b), "U": U, "p": p, "r_rb": r_rb,
            "med1": float(np.median(a)), "med2": float(np.median(b))}


def qp2(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    linhas = []
    for m in ["M1", "M2"]:
        r = kruskal(df, m, "faixa_exp")
        linhas.append({"Teste": f"Kruskal-Wallis: {m} entre faixas de experiência", "Índice": m,
                       "Estatística": r["H"], "p": r["p"], "Efeito": "ε²", "Valor do efeito": r["eps2"],
                       "Magnitude": rotulo_efeito(r["eps2"], (0.01, 0.08, 0.26)), "n": r["n"]})
    for m in ["M1", "M2"]:
        r = mann_whitney(df, m, "estagiario", "Estagiário(a)")
        linhas.append({"Teste": f"Mann-Whitney: {m}, estagiários × demais", "Índice": m,
                       "Estatística": r["U"], "p": r["p"], "Efeito": "r (bisserial por postos)",
                       "Valor do efeito": r["r_rb"], "Magnitude": rotulo_efeito(r["r_rb"]),
                       "n": r["n1"] + r["n2"],
                       "Medianas": f"medianas: estagiários {r.get('med1', np.nan):.2f} · demais {r.get('med2', np.nan):.2f}".replace(".", ",")})
    out = pd.DataFrame(linhas)
    out["p_holm"] = holm(out["p"])
    out["margem_erro_pp"] = out["n"].map(margem_erro)
    return out


def contingencia(df: pd.DataFrame, linha: str, coluna: str, semente: int) -> dict:
    tab = pd.crosstab(df[linha], df[coluna])
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    n = int(tab.to_numpy().sum())
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"tabela": tab, "n": n, "teste": "—", "p": np.nan, "V": np.nan, "nota": "Tabela com menos de 2 categorias."}
    chi2, p_chi, dof, esperado = stats.chi2_contingency(tab, correction=False)
    k = min(tab.shape) - 1
    V = math.sqrt(chi2 / (n * k)) if n > 0 else np.nan
    pct_baixo = (esperado < 5).mean()
    if pct_baixo > 0.2 or (esperado < 1).any():
        rng = np.random.default_rng(semente)
        if tab.shape == (2, 2):
            p = stats.fisher_exact(tab.to_numpy())[1]
            teste = "Exato de Fisher (2×2)"
        else:
            try:
                p = stats.fisher_exact(tab.to_numpy(), method=stats.MonteCarloMethod(n_resamples=10000, rng=rng)).pvalue
                teste = "Fisher-Freeman-Halton (Monte Carlo, 10.000)"
            except Exception:  # SciPy antigo
                p = stats.chi2_contingency(tab, method=stats.PermutationMethod(n_resamples=10000, rng=rng)).pvalue
                teste = "Qui-quadrado por permutação (10.000)"
        nota = f"{pct_baixo:.0%} das células com frequência esperada < 5 → teste exato."
    else:
        p, teste, nota = p_chi, "Qui-quadrado de Pearson", "Frequências esperadas adequadas."
    limites = tuple(x / math.sqrt(k) for x in (0.1, 0.3, 0.5))
    return {"tabela": tab, "n": n, "teste": teste, "chi2": chi2, "gl": dof, "p": p, "V": V,
            "magnitude": rotulo_efeito(V, limites), "nota": nota}


# ---------------------------------------------------------------------------
# QP3 — proporções com Wilson e kappa
# ---------------------------------------------------------------------------
def qp3(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    respondeu = df["E1"].notna() & (df["E1"].str.strip() != "")
    n = int(respondeu.sum())
    linhas = []
    for p in cfg["valores"]["praticas_e1"]:
        k = int(df.loc[respondeu, f"E1::{p}"].sum())
        lo, hi = wilson(k, n)
        linhas.append({"Prática": p, "k": k, "n": n, "Proporção": k / n if n else np.nan,
                       "IC95% inf": lo, "IC95% sup": hi})
    return pd.DataFrame(linhas).sort_values("Proporção", ascending=False)


def kappa_cohen(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    if d.empty:
        return np.nan
    tab = pd.crosstab(d.iloc[:, 0], d.iloc[:, 1])
    cats = sorted(set(tab.index) | set(tab.columns))
    tab = tab.reindex(index=cats, columns=cats, fill_value=0).to_numpy()
    n = tab.sum()
    po = np.trace(tab) / n
    pe = (tab.sum(0) * tab.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


# ---------------------------------------------------------------------------
# Distribuições Likert
# ---------------------------------------------------------------------------
def distribuicao(df: pd.DataFrame, itens: list[str]) -> pd.DataFrame:
    linhas = []
    for it in itens:
        s = df[it].dropna()
        n = len(s)
        cont = s.value_counts().reindex([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0)
        q1, med, q3 = (s.quantile([0.25, 0.5, 0.75]).tolist() if n else [np.nan] * 3)
        linhas.append({"Item": it, "n": n, "faltantes": int(df[it].isna().sum()),
                       **{f"p{int(k)}": (v / n if n else np.nan) for k, v in cont.items()},
                       "mediana": med, "Q1": q1, "Q3": q3})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
@dataclass
class Resultado:
    cfg: dict
    bruto_n: int
    padronizado: pd.DataFrame
    validas: pd.DataFrame
    log: pd.DataFrame
    info: dict
    desvios_auto: list = field(default_factory=list)


def pipeline(bruto: pd.DataFrame, cfg: dict) -> Resultado:
    pad = padronizar(bruto, cfg)
    validas, log, info = aplicar_descartes(pad, cfg)
    validas = calcular_indices(validas, cfg)
    return Resultado(cfg, len(bruto), pad, validas, log, info)
