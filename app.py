"""
Dashboard do survey "Clareza dos requisitos e retrabalho percebido".
Rodar: streamlit run app.py
Todos os números são calculados a partir da exportação bruta do formulário.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analise as A

st.set_page_config(page_title="Survey · Clareza e retrabalho", page_icon="📊", layout="wide")

BASE = Path(__file__).parent
CFG = A.carregar_config(BASE / "config.yaml")

# Paleta (divergente azul↔vermelho com cinza neutro; sequencial azul)
DIV = {1: "#c62f30", 2: "#f2a3a3", 3: "#d4d4d4", 4: "#84b9f3", 5: "#1b64be"}
SEQ = ["#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]
AZUL, LARANJA = "#257de0", "#f47230"
ROT_LIK = {1: "Discordo totalmente", 2: "Discordo parcialmente", 3: "Neutro",
           4: "Concordo parcialmente", 5: "Concordo totalmente"}
ROT_FRQ = {1: "Nunca", 2: "Raramente", 3: "Às vezes", 4: "Frequentemente", 5: "Sempre"}
ENUNCIADOS = {
    "C1": "C1 · Critérios de aceitação claros",
    "C2": "C2 · Informações completas antes da tarefa",
    "C3": "C3 · Entendo sem muitas explicações",
    "C4": "C4 · Requisitos sem ambiguidades",
    "R3": "R3 · Retrabalho consome parte significativa do tempo",
    "D1": "D1 · Dúvidas sobre o que implementar",
    "D2": "D2 · Mudança de requisito com tarefa em andamento",
    "R1": "R1 · Precisou refazer parte de tarefa já feita",
    "R2": "R2 · Retrabalho por interpretações diferentes",
}
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, system-ui, sans-serif", size=13),
              margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(bgcolor="white"))


# ---------------------------------------------------------------------------
# Carga dos dados brutos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lendo a exportação bruta…")
def carregar():
    caminho = BASE / CFG["arquivo_dados"]
    res = A.pipeline(A.ler_bruto(caminho), CFG)
    return res, caminho.name


with st.sidebar:
    st.markdown(f"### {CFG['estudo']['titulo']}")

res, nome_arquivo = carregar()
simulado = "simulad" in nome_arquivo.lower()


PAGINAS = ["Visão geral e amostra", "Itens Likert", "QP1 · Clareza × retrabalho",
           "QP2 · Experiência e função", "QP3 · Práticas", "Confiabilidade (alfa)",
           "Metodologia e desvios", "Evidências da coleta"]

with st.sidebar:
    pagina = st.radio("Página", PAGINAS, label_visibility="collapsed")
    st.divider()
    st.markdown("**Filtros por segmento**")
    v = res.validas

    def ms(rotulo, coluna):
        opcoes = [x for x in (v[coluna].cat.categories if hasattr(v[coluna], "cat") else sorted(v[coluna].dropna().unique()))
                  if (v[coluna] == x).any()]
        return st.multiselect(rotulo, opcoes, placeholder="Todos")

    f_exp = ms("Tempo de experiência", "faixa_exp")
    f_fun = ms("Função", "P1")
    f_por = ms("Porte da organização", "P3")
    f_for = ms("Forma de trabalho", "P4")

filtros = {"faixa_exp": f_exp, "P1": f_fun, "P3": f_por, "P4": f_for}
df = res.validas.copy()
for col, sel in filtros.items():
    if sel:
        df = df[df[col].isin(sel)]
filtro_ativo = any(filtros.values())
n = len(df)

# ---------------------------------------------------------------------------
# Avisos globais
# ---------------------------------------------------------------------------
if simulado:
    st.error("**DADOS SIMULADOS** — o arquivo carregado é o conjunto fictício de demonstração "
             "(`gerar_dados_simulados.py`). Troque `arquivo_dados` em `config.yaml` pela exportação real.",
             icon="⚠️")
if filtro_ativo:
    desc = " · ".join(f"{', '.join(map(str, s))}" for s in filtros.values() if s)
    st.info(f"**Filtro ativo** ({desc}) — n = {n}. Recortes por segmento não estavam no plano de análise; "
            f"leia como exploratório. Margem de erro com este n: ±{A.margem_erro(n):.1f} p.p.", icon="🔎")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def pouco(nmin=10):
    if n < nmin:
        st.warning(f"Com o filtro atual restam apenas {n} respostas válidas — insuficiente para este teste.")
        return True
    return False


def cartao_teste(titulo, teste, estat, p, p_aj, efeito_nome, efeito, magnitude, n_, ic=None, extra=None):
    with st.container(border=True):
        st.markdown(f"**{titulo}** &nbsp;·&nbsp; <span style='color:#666'>{teste}</span>", unsafe_allow_html=True)
        c = st.columns(5)
        c[0].metric("p-valor bruto", fmt_p(p))
        c[1].metric("p-valor ajustado (Holm)", fmt_p(p_aj))
        c[2].metric(f"Efeito {efeito_nome.split(' (')[0]} · {magnitude}", fmt(efeito))
        c[3].metric("n", n_)
        c[4].metric("Margem de erro (95%)", f"±{fmt(A.margem_erro(n_), 1)} p.p.")
        linha = []
        if estat is not None:
            linha.append(f"Estatística: {estat}")
        if ic is not None:
            linha.append(f"IC 95% bootstrap ({CFG['analise']['bootstrap_reps']} reamostragens): [{fmt(ic[0])}; {fmt(ic[1])}]")
        if extra:
            linha.append(extra)
        if linha:
            st.caption(" · ".join(linha))


# ---------------------------------------------------------------------------
# Página: visão geral e perfil da amostra
# ---------------------------------------------------------------------------
def pg_visao():
    st.title("Perfil da amostra")
    log = res.log
    desc = log[log["criterio"].notna()]
    e = CFG["estudo"]
    c = st.columns(5)
    c[0].metric("Respostas recebidas", res.bruto_n)
    c[1].metric("Descartadas", len(desc))
    c[2].metric(f"Válidas (meta {e['meta_n']} · mín. {e['minimo_n']})", len(res.validas))
    c[3].metric("Margem de erro (n válido)", f"±{fmt(A.margem_erro(len(res.validas)), 1)} p.p.")
    c[4].metric("Válidas no filtro", n)

    st.subheader("Descartes e motivo")
    linhas = []
    for k, motivo in A.MOTIVOS.items():
        linhas.append({"Critério": f"({k})", "Motivo": motivo,
                       "Descartadas": int((log["criterio"] == k).sum()),
                       "Violaram (com sobreposição)": res.info["sobreposicao"][k]})
    tab = pd.DataFrame(linhas)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.table(tab.set_index("Critério"))
        lim = res.info["limite_tempo_seg"]
        aviso_pre = "" if CFG["pre_teste"]["preenchido"] else " ⚠️ valor provisório — preencher com a mediana real do pré-teste em config.yaml."
        st.caption(
            "Cada resposta é contada no primeiro critério violado, na ordem (a)→(f) do protocolo, para não haver dupla contagem."
            + (f" Limite do critério (d): {fmt(lim / 60, 1)} min (1/3 de {fmt(CFG['pre_teste']['mediana_min'], 1)} min).{aviso_pre}"
               if res.info["tempo_disponivel"] else ""))
        nao_aplic = []
        if not res.info.get("atencao_disponivel", True):
            nao_aplic.append("(c) — o formulário não teve pergunta de atenção")
        if not res.info["tempo_disponivel"]:
            nao_aplic.append("(d) — o Google Forms não registra hora de início")
        if nao_aplic:
            st.warning("Critérios **não aplicáveis** a esta coleta: " + "; ".join(nao_aplic) + ". Ver página de desvios.")
    with col2:
        rot = ["Recebidas"] + [f"Descarte ({k})" for k in A.MOTIVOS] + ["Válidas"]
        val = [res.bruto_n] + [int((log["criterio"] == k).sum()) for k in A.MOTIVOS] + [len(res.validas)]
        fig = go.Figure(go.Bar(x=val, y=rot, orientation="h", text=val, textposition="outside", cliponaxis=False,
                               marker_color=[AZUL] + ["#bdbdbd"] * 6 + [SEQ[2]], marker_cornerradius=4,
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_layout(**LAYOUT, height=330, title="Do bruto às respostas válidas",
                          yaxis=dict(autorange="reversed"), xaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Registro de descartes (ID sequencial do formulário e motivo — sem dados pessoais)"):
        st.dataframe(desc[["id", "carimbo", "criterio", "motivo"]], hide_index=True, use_container_width=True)
        st.download_button("Baixar registro de descartes (CSV)", desc[["id", "criterio", "motivo"]].to_csv(index=False).encode("utf-8-sig"),
                           "registro_descartes.csv", "text/csv")

    st.subheader("Quem respondeu (respostas válidas no filtro)")
    c = st.columns(4)
    for col, (var, titulo) in zip(c, [("faixa_exp", "Tempo de experiência"), ("P1", "Função"),
                                       ("P3", "Porte da organização"), ("P4", "Forma de trabalho")]):
        cont = df[var].value_counts(sort=False)
        cont = cont[cont.index.isin(df[var].dropna().unique())]
        fig = go.Figure(go.Bar(x=cont.values, y=[str(i) for i in cont.index], orientation="h",
                               marker_color=AZUL, marker_cornerradius=4, text=cont.values, textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x} (%{customdata:.0%})<extra></extra>",
                               customdata=cont.values / max(1, cont.sum())))
        fig.update_layout(**LAYOUT, height=260, title=titulo, yaxis=dict(autorange="reversed"),
                          xaxis=dict(showgrid=False, visible=False, range=[0, cont.max() * 1.25 if len(cont) else 1]))
        col.plotly_chart(fig, use_container_width=True)
    anos = df["P2"].dropna()
    if len(anos):
        st.caption(f"Anos de experiência: mediana {fmt(anos.median(), 1)} · intervalo {fmt(anos.min(), 1)}–{fmt(anos.max(), 1)} · "
                   f"IIQ {fmt(anos.quantile(.25), 1)}–{fmt(anos.quantile(.75), 1)} (n = {len(anos)}).")

    st.subheader("Índices (mediana e intervalo interquartil)")
    linhas = []
    for m, (nome, itens) in A.CONSTRUTOS.items():
        s = df[m].dropna()
        linhas.append({"Índice": m, "Construto": nome, "Itens": ", ".join(itens), "n": len(s),
                       "Mediana": s.median(), "Q1": s.quantile(.25), "Q3": s.quantile(.75)})
    st.dataframe(pd.DataFrame(linhas).round(2), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Página: Likert divergente
# ---------------------------------------------------------------------------
def barras_divergentes(itens, rotulos, titulo):
    dist = A.distribuicao(df, itens).set_index("Item")
    fig = go.Figure()
    ys = [ENUNCIADOS[i] for i in itens]
    neutro = dist["p3"] / 2
    ordem = [(3, -neutro, True), (2, -dist["p2"], True), (1, -dist["p1"], True),
             (3, neutro, False), (4, dist["p4"], True), (5, dist["p5"], True)]
    for cat, valores, legenda in ordem:
        real = dist[f"p{cat}"]
        fig.add_bar(y=ys, x=valores * 100, orientation="h", name=rotulos[cat], legendgroup=str(cat), legendrank=cat,
                    showlegend=legenda and not (cat == 3 and valores.sum() > 0), marker_color=DIV[cat],
                    marker_line=dict(color="white", width=2),
                    customdata=np.c_[real * 100, dist["n"]],
                    hovertemplate=f"<b>%{{y}}</b><br>{rotulos[cat]}: %{{customdata[0]:.0f}}%<br>n = %{{customdata[1]}}<extra></extra>")
    neg = (dist["p1"] + dist["p2"]) * 100
    pos = (dist["p4"] + dist["p5"]) * 100
    for y, a, b, lo, hi in zip(ys, neg, pos, -(neg + neutro * 100), pos + neutro * 100):
        fig.add_annotation(x=lo, y=y, text=f"{a:.0f}%", showarrow=False, xanchor="right", xshift=-4, font=dict(color="#666"))
        fig.add_annotation(x=hi, y=y, text=f"{b:.0f}%", showarrow=False, xanchor="left", xshift=4, font=dict(color="#666"))
    fig.update_layout(**LAYOUT, barmode="relative", height=110 + 55 * len(itens), title=titulo,
                      legend=dict(orientation="h", y=-0.15, traceorder="normal"),
                      xaxis=dict(range=[-105, 105], ticksuffix="%", tickvals=[-100, -50, 0, 50, 100],
                                 ticktext=["100%", "50%", "0", "50%", "100%"], zeroline=True, zerolinecolor="#999"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    tab = dist.reset_index()[["Item", "n", "faltantes", "mediana", "Q1", "Q3"] + [f"p{k}" for k in range(1, 6)]]
    tab = tab.rename(columns={f"p{k}": rotulos[k] for k in range(1, 6)})
    with st.expander("Tabela (percentuais, mediana, IIQ e faltantes)"):
        st.dataframe(tab.style.format({rotulos[k]: "{:.0%}" for k in range(1, 6)} | {"mediana": "{:.1f}", "Q1": "{:.1f}", "Q3": "{:.1f}"}),
                     hide_index=True, use_container_width=True)


def pg_likert():
    st.title("Itens em escala Likert")
    st.caption(f"n = {n} respostas válidas · margem de erro ±{fmt(A.margem_erro(n), 1)} p.p. · "
               "a categoria neutra é dividida ao meio em torno do zero; os números nas pontas somam as categorias de cada lado.")
    barras_divergentes(A.ITENS_LIKERT, ROT_LIK, "Concordância (1 = discordo totalmente … 5 = concordo totalmente)")
    barras_divergentes(A.ITENS_FREQ, ROT_FRQ, "Frequência nos últimos 3 meses (1 = nunca … 5 = sempre)")
    st.caption("R3 aparece na escala de frequência porque foi assim que o formulário oficial o apresentou (desvio declarado).")


# ---------------------------------------------------------------------------
# Página: QP1
# ---------------------------------------------------------------------------
def dispersao(x, y, tx, ty):
    d = df[[x, y]].dropna()
    rng = np.random.default_rng(1)
    fig = go.Figure(go.Scatter(x=d[x] + rng.uniform(-.06, .06, len(d)), y=d[y] + rng.uniform(-.06, .06, len(d)),
                               mode="markers", marker=dict(size=9, color=AZUL, opacity=.65, line=dict(color="white", width=1.5)),
                               hovertemplate=f"{tx}: %{{customdata[0]:.2f}}<br>{ty}: %{{customdata[1]:.2f}}<extra></extra>",
                               customdata=d[[x, y]].to_numpy()))
    fig.update_layout(**LAYOUT, height=380, xaxis=dict(title=tx, range=[.8, 5.2]), yaxis=dict(title=ty, range=[.8, 5.2]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Pontos com leve deslocamento aleatório para não se sobreporem (os valores reais aparecem no hover).")


def pg_qp1():
    st.title("QP1 · A clareza percebida se associa ao retrabalho percebido?")
    st.markdown("**H1:** associação negativa entre clareza (M1) e retrabalho (M2), ρ < 0. "
                "Spearman com IC 95% bootstrap; Holm dentro da família QP1. **Associação não é causa.**")
    if pouco():
        return
    r = A.qp1(df, CFG)
    rel, _ = A.confiabilidade(df, CFG)
    alfa = dict(zip(rel["Índice"], rel["Alfa de Cronbach"]))
    lim = CFG["analise"]["alfa_cronbach_minimo"]
    for _, t in r.iterrows():
        cartao_teste(t["Teste"], "Correlação de Spearman", f"ρ = {fmt(t['rho'])}", t["p"], t["p_holm"], "|ρ|", abs(t["rho"]),
                     t["efeito"], int(t["n"]), ic=(t["ic_inf"], t["ic_sup"]),
                     extra=f"direção {'negativa' if t['rho'] < 0 else 'positiva'}")
        if t["x"] == "M1":
            sup = "apoia" if (t["rho"] < 0 and t["p_holm"] < CFG["analise"]["alfa"]) else "não apoia"
            st.success(f"**Resposta à QP1:** nesta amostra, o resultado **{sup} H1** — ρ = {fmt(t['rho'])}, "
                       f"IC 95% [{fmt(t['ic_inf'])}; {fmt(t['ic_sup'])}], p ajustado = {fmt_p(t['p_holm'])}, efeito {t['efeito']}. "
                       "Vale apenas para esta amostra por conveniência; não generaliza para a população.")
    c1, c2 = st.columns(2)
    with c1:
        dispersao("M1", "M2", "Clareza (M1)", "Retrabalho (M2)")
    with c2:
        dispersao("M3", "M2", "Instabilidade (M3)", "Retrabalho (M2)")

    baixos = [m for m, a in alfa.items() if a < lim]
    if baixos:
        st.warning(f"Alfa abaixo de {fmt(lim)} em {', '.join(baixos)} → conforme o protocolo, os itens também são analisados isoladamente (desvio declarado).")
        if "M1" in baixos or "M2" in baixos:
            st.dataframe(A.qp1_itens(df, CFG).round(3), hide_index=True, use_container_width=True)
        if "M3" in baixos:
            linhas = []
            for it in A.ITENS_D:
                s = A.spearman_boot(df[it], df["M2"], CFG["analise"]["bootstrap_reps"], CFG["analise"]["semente"])
                linhas.append({"Item": it, "× M2 (retrabalho)": "", **s})
            t = pd.DataFrame(linhas)
            t["p_holm"] = A.holm(t["p"])
            t["efeito"] = t["rho"].map(A.rotulo_efeito)
            st.markdown("**Itens de instabilidade (D1, D2) isolados × retrabalho (M2)**")
            st.dataframe(t.drop(columns="× M2 (retrabalho)").round(3), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Página: QP2
# ---------------------------------------------------------------------------
def boxplot(var, grupo, titulo, cores):
    d = df[[var, grupo]].dropna()
    fig = go.Figure()
    cats = [c for c in (d[grupo].cat.categories if hasattr(d[grupo], "cat") else sorted(d[grupo].unique())) if (d[grupo] == c).any()]
    for i, c in enumerate(cats):
        s = d.loc[d[grupo] == c, var]
        fig.add_box(y=s, name=f"{c} (n={len(s)})", marker_color=cores[i % len(cores)], boxpoints="all",
                    jitter=.35, pointpos=0, marker=dict(size=6, opacity=.5), line=dict(width=2))
    fig.update_layout(**LAYOUT, height=340, title=titulo, showlegend=False, yaxis=dict(range=[.8, 5.2]))
    st.plotly_chart(fig, use_container_width=True)


def pg_qp2():
    st.title("QP2 · Clareza e retrabalho variam com a experiência?")
    st.markdown("**H2 (exploratória):** há diferença entre faixas de experiência, sem direção pré-definida. "
                "Kruskal-Wallis (ε²) + Dunn/Holm; Mann-Whitney estagiários × demais (r bisserial por postos). "
                "Holm sobre os 4 testes da família QP2.")
    if pouco():
        return
    r = A.qp2(df, CFG)
    for _, t in r.iterrows():
        est = f"H = {fmt(t['Estatística'])}" if "Kruskal" in t["Teste"] else f"U = {fmt(t['Estatística'], 1)}"
        cartao_teste(t["Teste"], t["Teste"].split(":")[0], est, t["p"], t["p_holm"], t["Efeito"], t["Valor do efeito"],
                     t["Magnitude"], int(t["n"]), extra=t.get("Medianas") if isinstance(t.get("Medianas"), str) else None)
    alfa = CFG["analise"]["alfa"]
    sig = r[r["p_holm"] < alfa]
    st.success("**Resposta à QP2:** " + (
        "nenhuma diferença significativa após Holm; os tamanhos de efeito estão nos cartões acima. "
        if sig.empty else f"diferença significativa após Holm em: {'; '.join(sig['Teste'])}. ") +
        "Resultado exploratório, restrito a esta amostra.")

    c1, c2 = st.columns(2)
    with c1:
        boxplot("M1", "faixa_exp", "Clareza (M1) por faixa de experiência", SEQ)
        boxplot("M1", "estagiario", "Clareza (M1): estagiários × demais", [LARANJA, AZUL])
    with c2:
        boxplot("M2", "faixa_exp", "Retrabalho (M2) por faixa de experiência", SEQ)
        boxplot("M2", "estagiario", "Retrabalho (M2): estagiários × demais", [LARANJA, AZUL])

    st.subheader("Post-hoc de Dunn (correção de Holm)")
    c1, c2 = st.columns(2)
    for col, m in zip((c1, c2), ("M1", "M2")):
        col.markdown(f"**{m}**")
        col.dataframe(A.dunn(df, m, "faixa_exp").round(3), hide_index=True, use_container_width=True)

    st.subheader("Associação entre variáveis nominais (qui-quadrado / Fisher, V de Cramér)")
    opcoes = {"Porte da organização": "P3", "Forma de trabalho": "P4", "Função": "P1", "Faixa de experiência": "faixa_exp",
              "Estagiário × demais": "estagiario"}
    opcoes |= {f"Usa: {p}": f"E1::{p}" for p in CFG["valores"]["praticas_e1"] if p not in ("Nenhuma", "Outra")}
    c1, c2 = st.columns(2)
    a = c1.selectbox("Linhas", list(opcoes), index=0)
    b = c2.selectbox("Colunas", list(opcoes), index=5)
    if a == b:
        st.info("Escolha duas variáveis diferentes.")
        return
    ct = A.contingencia(df, opcoes[a], opcoes[b], CFG["analise"]["semente"])
    c = st.columns(4)
    c[0].metric("Teste", ct["teste"].split(" (")[0], help=ct["teste"])
    c[1].metric("p-valor", fmt_p(ct["p"]))
    c[2].metric(f"V de Cramér · {ct.get('magnitude', '—')}", fmt(ct["V"]))
    c[3].metric("Margem de erro (n = %d)" % ct["n"], f"±{fmt(A.margem_erro(ct['n']), 1)} p.p.")
    st.caption(ct["nota"] + " Análise complementar, fora da família de Holm da QP2.")
    st.dataframe(ct["tabela"], use_container_width=True)


# ---------------------------------------------------------------------------
# Página: QP3
# ---------------------------------------------------------------------------
def pg_qp3():
    st.title("QP3 · Quais práticas de esclarecimento são mais usadas?")
    st.markdown("Descritiva, sem hipótese. Proporção de respondentes que usam cada prática, com **intervalo de Wilson de 95%** (múltipla escolha: as proporções não somam 100%).")
    if pouco(5):
        return
    t = A.qp3(df, CFG)
    nn = int(t["n"].iloc[0])
    t = t.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["Proporção"] * 100, y=t["Prática"], orientation="h", marker_color=AZUL, marker_cornerradius=4,
        error_x=dict(type="data", symmetric=False, array=(t["IC95% sup"] - t["Proporção"]) * 100,
                     arrayminus=(t["Proporção"] - t["IC95% inf"]) * 100, color="#555", thickness=1.5, width=5),
        text=[f"{p:.0%}" for p in t["Proporção"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"),
        customdata=np.c_[t["k"], t["IC95% inf"] * 100, t["IC95% sup"] * 100],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]} de " + str(nn) + " (%{x:.0f}%)<br>IC 95% Wilson: %{customdata[1]:.0f}%–%{customdata[2]:.0f}%<extra></extra>"))
    fig.update_layout(**LAYOUT, height=420, xaxis=dict(range=[0, 105], ticksuffix="%", title="% dos respondentes"))
    st.plotly_chart(fig, use_container_width=True)
    top = t.iloc[-1]
    st.success(f"**Resposta à QP3:** a prática mais citada é **{top['Prática']}** ({top['Proporção']:.0%}, "
               f"IC 95% {top['IC95% inf']:.0%}–{top['IC95% sup']:.0%}; n = {nn}). Intervalos que se sobrepõem não permitem ordenar as práticas com segurança.")
    with st.expander("Tabela"):
        st.dataframe(t.iloc[::-1].style.format({"Proporção": "{:.1%}", "IC95% inf": "{:.1%}", "IC95% sup": "{:.1%}"}),
                     hide_index=True, use_container_width=True)

    st.subheader("E2 · Prática mais útil (pergunta aberta, codificada)")
    arq = CFG.get("arquivo_codificacao_e2")
    caminho = BASE / arq if arq else None
    if not caminho or not caminho.exists():
        st.info("Ainda não há arquivo de codificação. Gere `codificacao_E2.csv` com as colunas id, codificador_1, codificador_2, codigo_final.")
        return
    cod = pd.read_csv(caminho)
    cod = cod[cod["id"].isin(df["id"])]
    if cod[["codificador_1", "codificador_2"]].isna().all().any():
        n_txt = int((df["E2"].fillna("").str.strip() != "").sum())
        st.info(f"{n_txt} respostas abertas aguardam codificação. Dois integrantes devem preencher, de forma independente, "
                f"as colunas `codificador_1` e `codificador_2` de `{arq}` e, depois do consenso, `codigo_final`. "
                "O kappa de Cohen e o gráfico de códigos aparecem aqui automaticamente.")
        return
    kappa = A.kappa_cohen(cod["codificador_1"], cod["codificador_2"])
    conc = (cod["codificador_1"] == cod["codificador_2"]).mean()
    c = st.columns(3)
    c[0].metric("Respostas codificadas", len(cod))
    c[1].metric("Kappa de Cohen", fmt(kappa))
    c[2].metric("Concordância bruta", f"{conc:.0%}")
    cont = cod["codigo_final"].value_counts().iloc[::-1]
    fig = go.Figure(go.Bar(x=cont.values, y=cont.index, orientation="h", marker_color=SEQ[2], marker_cornerradius=4,
                           text=cont.values, textposition="outside", cliponaxis=False))
    fig.update_layout(**LAYOUT, height=300, title="Códigos finais (após consenso)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("O texto livre das respostas não é exibido: só os códigos, depois da revisão que remove identificadores (seção 6 do protocolo).")


# ---------------------------------------------------------------------------
# Página: confiabilidade
# ---------------------------------------------------------------------------
def pg_alfa():
    st.title("Confiabilidade dos construtos (alfa de Cronbach)")
    if pouco():
        return
    rel, itens = A.confiabilidade(df, CFG)
    lim = CFG["analise"]["alfa_cronbach_minimo"]
    cols = st.columns(3)
    for col, (_, r) in zip(cols, rel.iterrows()):
        ok = r["Alfa de Cronbach"] >= lim
        col.metric(f"{r['Índice']} · {r['Construto']}", fmt(r["Alfa de Cronbach"]))
        col.markdown(f":{'green' if ok else 'orange'}-badge[{'✓ aceitável' if ok else '✗ abaixo de ' + fmt(lim) + ' → itens isolados'}]")
        col.caption(f"Itens {r['Itens']} · n = {r['n (casos completos)']} · IC 95% [{fmt(r['IC95% inf'])}; {fmt(r['IC95% sup'])}]")
    fig = go.Figure()
    fig.add_bar(x=rel["Índice"] + " · " + rel["Construto"], y=rel["Alfa de Cronbach"], marker_cornerradius=4,
                marker_color=[AZUL if a >= lim else LARANJA for a in rel["Alfa de Cronbach"]],
                error_y=dict(type="data", symmetric=False, array=rel["IC95% sup"] - rel["Alfa de Cronbach"],
                             arrayminus=rel["Alfa de Cronbach"] - rel["IC95% inf"], color="#555"),
                hovertemplate="%{x}<br>α = %{y:.2f}<extra></extra>")
    fig.add_hline(y=lim, line_dash="dash", line_color="#666", annotation_text=f"mínimo {fmt(lim)}")
    fig.update_layout(**LAYOUT, height=340, yaxis=dict(range=[0, 1], title="alfa"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Diagnóstico por item")
    st.dataframe(itens.round(3), hide_index=True, use_container_width=True)
    st.caption("Com 2 itens (M3), o alfa equivale ao coeficiente de Spearman-Brown sobre a covariância dos itens e tende a ser baixo; "
               "'alfa se o item for removido' não se aplica. IC por bootstrap percentil.")


# ---------------------------------------------------------------------------
# Página: metodologia e desvios
# ---------------------------------------------------------------------------
def desvios_automaticos():
    out = []
    rel, _ = A.confiabilidade(res.validas, CFG)
    lim = CFG["analise"]["alfa_cronbach_minimo"]
    for _, r in rel.iterrows():
        if r["Alfa de Cronbach"] < lim:
            out.append((f"Alfa de {r['Índice']} ({r['Construto']}) = {fmt(r['Alfa de Cronbach'])} < {fmt(lim)}",
                        "Conforme a regra pré-registrada, os itens desse construto também são analisados isoladamente (QP1)."))
    nv = len(res.validas)
    if nv < CFG["estudo"]["minimo_n"]:
        out.append((f"n válido = {nv}, abaixo do mínimo de {CFG['estudo']['minimo_n']}",
                    f"Margem de erro sobe para ±{fmt(A.margem_erro(nv), 1)} p.p.; poder estatístico menor."))
    elif nv < CFG["estudo"]["meta_n"]:
        out.append((f"n válido = {nv}, abaixo da meta de {CFG['estudo']['meta_n']} (acima do mínimo)",
                    f"Margem de erro ±{fmt(A.margem_erro(nv), 1)} p.p. em vez de ±12,7 p.p."))
    if not res.info.get("atencao_disponivel", True):
        out.append(("Critério (c) não aplicado", "O formulário não incluiu a pergunta de atenção A1."))
    if not res.info["tempo_disponivel"]:
        out.append(("Critério (d) não aplicado", "A exportação não registra hora de início/duração."))
    p = res.padronizado
    if p["carimbo"].notna().any():
        ini_real, ini_plan = p["carimbo"].min().normalize(), pd.Timestamp(CFG["estudo"]["coleta_inicio"])
        if ini_real != ini_plan:
            out.append((f"Primeira resposta em {ini_real:%d/%m/%Y}; o protocolo previa início em {ini_plan:%d/%m/%Y}",
                        "Período de coleta efetivo menor que os 14 dias planejados."))
    return out


def pg_metodo():
    st.title("Metodologia e desvios do protocolo")
    e = CFG["estudo"]
    st.markdown(f"Protocolo pré-registrado (Parte 1): [commit fbe39a1]({e['commit_protocolo']}) · "
                f"Coleta planejada: {pd.Timestamp(e['coleta_inicio']):%d/%m/%Y} a {pd.Timestamp(e['coleta_fim']):%d/%m/%Y}.")

    st.subheader("Desvios em relação ao protocolo")
    md = BASE / "desvios.md"
    if md.exists():
        import re
        st.markdown(re.sub(r"<!--.*?-->", "", md.read_text(encoding="utf-8"), flags=re.S))
    auto = desvios_automaticos()
    if auto:
        st.markdown("**Detectados automaticamente a partir dos dados:**")
        st.dataframe(pd.DataFrame(auto, columns=["O que mudou", "Consequência / tratamento"]), hide_index=True, use_container_width=True)

    st.subheader("Plano de análise executado")
    plano = pd.DataFrame([
        ["QP1", "M1 × M2 (e M3 × M2)", "Spearman, IC 95% bootstrap (2.000)", "|ρ|: 0,1 / 0,3 / 0,5", "Holm na família QP1"],
        ["QP2", "M1, M2 entre faixas de experiência", "Kruskal-Wallis + Dunn (Holm)", "ε²: 0,01 / 0,08 / 0,26", "Holm na família QP2"],
        ["QP2", "M1, M2 estagiários × demais", "Mann-Whitney", "r bisserial por postos: 0,1 / 0,3 / 0,5", "Holm na família QP2"],
        ["QP2", "Porte / forma × categorias nominais", "Qui-quadrado ou exato de Fisher", "V de Cramér (ajustado por gl)", "Complementar"],
        ["QP3", "Práticas usadas (E1)", "Proporção + IC de Wilson 95%", "—", "Descritiva"],
        ["QP3", "Prática mais útil (E2)", "Análise temática, 2 codificadores", "Kappa de Cohen", "Descritiva"],
        ["Todos", "Construtos C, R, D", "Alfa de Cronbach (IC bootstrap)", "mínimo 0,70", "—"],
    ], columns=["QP", "Variáveis", "Teste", "Efeito (pequeno/médio/grande)", "Multiplicidade"])
    st.dataframe(plano, hide_index=True, use_container_width=True)

    st.subheader("Decisões operacionais (o protocolo não detalhava)")
    st.markdown("""
- **Ordem dos descartes:** cada resposta conta no primeiro critério violado, de (a) a (f); a tabela também mostra quantas violaram cada critério com sobreposição.
- **Straight-lining (e):** mesma resposta em todos os itens C1–C4, D1–D2 e R1–R3 (A1 fica de fora porque tem resposta imposta). Como os blocos têm sentidos opostos, isso indica preenchimento descuidado.
- **Duplicata evidente (f):** todas as respostas iguais, ignorando data/hora, ID e canal. Fica a primeira.
- **Faixas de experiência:** o formulário coletou P2 em faixas fechadas (menos de 1, 1–2, 3–6, 7–9, mais de 10 anos), e não como número. “7–9” e “mais de 10” foram unidas em “7 anos ou mais” para manter as 4 faixas do protocolo; por isso não há mediana de anos.
- **Porte:** o formulário teve 4 opções; “51–100” e “101–500” foram unidas em “51–500 pessoas”, como no protocolo.
- **Índices:** média dos itens, só com todos os itens respondidos (análise por caso disponível, sem imputação).
- **Tabelas maiores que 2×2 com esperado < 5:** exato de Fisher-Freeman-Halton por Monte Carlo (10.000).
- **Margem de erro:** e = 1,96·√(0,25/n), pior caso. Como a amostra é por conveniência, ela serve como referência de precisão e não permite inferir para a população.
- **Filtros por segmento:** recortes pós-hoc e exploratórios; o dashboard recalcula tudo e mostra o n e a margem de erro do recorte.
""")
    st.subheader("Ameaças à validade (resumo)")
    st.markdown("Amostra por conveniência/bola de neve (validade externa) · percepção ≠ retrabalho real e possível causalidade "
                "invertida (constructo/interna) · desejabilidade social · viés de memória (janela de 3 meses) · "
                "muitos testes com amostra pequena (Holm, efeitos e IC).")


# ---------------------------------------------------------------------------
# Página: evidências da coleta
# ---------------------------------------------------------------------------
def pg_evid():
    st.title("Evidências da coleta")
    st.caption("Tudo nesta página vem da exportação bruta, depois de removidas as colunas identificadoras listadas em config.yaml.")
    p = res.padronizado.copy()
    p = p.merge(res.log[["id", "criterio"]], on="id", how="left")
    p["status"] = np.where(p["criterio"].isna(), "Válida", "Descartada")
    e = CFG["estudo"]
    st.markdown(
        f"| Arquivo lido | Primeira resposta | Última resposta | Colunas identificadoras removidas na leitura |\n|---|---|---|---|\n"
        f"| `{nome_arquivo}` | {p['carimbo'].min():%d/%m/%Y %H:%M} | {p['carimbo'].max():%d/%m/%Y %H:%M} | "
        f"{', '.join(c for c in CFG['colunas_remover']) or '—'} |")

    dia = p.assign(dia=p["carimbo"].dt.floor("D")).groupby(["dia", "status"]).size().unstack(fill_value=0)
    dias = pd.date_range(min(dia.index.min(), pd.Timestamp(e["coleta_inicio"])), max(dia.index.max(), pd.Timestamp(e["coleta_fim"])))
    dia = dia.reindex(dias, fill_value=0)
    for col_ in ("Válida", "Descartada"):
        if col_ not in dia:
            dia[col_] = 0
    fig = go.Figure()
    fig.add_bar(x=dia.index, y=dia["Válida"], name="Válidas", marker_color=AZUL, marker_cornerradius=3)
    fig.add_bar(x=dia.index, y=dia["Descartada"], name="Descartadas", marker_color="#bdbdbd", marker_cornerradius=3)
    for l in e.get("lembretes", []):
        fig.add_vline(x=pd.Timestamp(l), line_dash="dot", line_color=LARANJA)
        fig.add_annotation(x=pd.Timestamp(l), y=1, yref="paper", text="lembrete", showarrow=False, font=dict(color=LARANJA), yshift=8)
    fig.update_layout(**LAYOUT, barmode="stack", height=320, title="Respostas por dia", legend=dict(orientation="h", x=1, xanchor="right", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    acum = dia["Válida"].cumsum()
    fig = go.Figure(go.Scatter(x=acum.index, y=acum.values, mode="lines+markers", line=dict(color=AZUL, width=2),
                               marker=dict(size=8), name="Válidas acumuladas",
                               hovertemplate="%{x|%d/%m}: %{y} válidas<extra></extra>"))
    fig.add_hline(y=e["meta_n"], line_dash="dash", line_color="#666", annotation_text=f"meta {e['meta_n']}")
    fig.add_hline(y=e["minimo_n"], line_dash="dot", line_color="#999", annotation_text=f"mínimo {e['minimo_n']}")
    fig.add_vrect(x0=pd.Timestamp(e["coleta_inicio"]), x1=pd.Timestamp(e["coleta_fim"]) + pd.Timedelta(days=1),
                  fillcolor="#e8f1fc", opacity=.5, line_width=0, layer="below")
    fig.update_layout(**LAYOUT, height=320, title="Respostas válidas acumuladas (faixa azul = período planejado)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if p["canal"].notna().any():
        cc = p.groupby(["canal", "status"]).size().unstack(fill_value=0)
        cc["Total"] = cc.sum(axis=1)
        st.markdown("**Respostas por canal de divulgação (identificador genérico do link)**")
        st.dataframe(cc.sort_values("Total", ascending=False), use_container_width=True)

    st.markdown("**Exportação anonimizada com data e hora de cada resposta**")
    exp = p[["id", "carimbo", "duracao_seg", "canal", "status", "criterio"]].rename(
        columns={"carimbo": "data_hora_envio", "duracao_seg": "duracao_segundos"}).dropna(axis=1, how="all")
    st.dataframe(exp, hide_index=True, use_container_width=True, height=260)
    st.download_button("Baixar (CSV)", exp.to_csv(index=False).encode("utf-8-sig"), "evidencia_data_hora.csv", "text/csv")

{"Visão geral e amostra": pg_visao, "Itens Likert": pg_likert, "QP1 · Clareza × retrabalho": pg_qp1,
 "QP2 · Experiência e função": pg_qp2, "QP3 · Práticas": pg_qp3, "Confiabilidade (alfa)": pg_alfa,
 "Metodologia e desvios": pg_metodo, "Evidências da coleta": pg_evid}[pagina]()
