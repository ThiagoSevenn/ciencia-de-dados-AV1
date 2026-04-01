# =============================================================================
# DASHBOARD — RECLAMEAQUI  |  Streamlit
# =============================================================================
# Como executar localmente:
#   pip install -r library/requirements.txt
#   streamlit run dashboard/dashboard_reclameaqui.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import io
import base64

# ── Setup NLTK ────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
PT_STOPS = set(stopwords.words("portuguese")).union({
    "nao", "ser", "empresa", "produto", "atendimento",
    "cliente", "reclamacao", "problema", "ja", "pois",
    "mais", "sem_descricao", "dia", "vez", "que", "com",
    # Termos da marca
    "pao", "pão", "acucar", "açucar", "açúcar", "acucar",
    "paodeacucar", "pãodeaçúcar", "pão de açúcar",
    "extra", "grupo", "gpa",
})

# ── Configuração geral da página ──────────────────────────────────────────────
st.set_page_config(
    page_title="ReclameAqui — Dashboard BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1f4e79; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO E CACHE DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carrega e pré-processa o dataset do ReclameAqui."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Padroniza colunas para letras minúsculas
    df.columns = df.columns.str.strip().str.lower()

    # Drop registros duplicados ignorando a url
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=df.columns.drop('url')).copy()
    else:
        df = df.drop_duplicates().copy()

    # Remover espaços extras em colunas de texto
    str_cols = [c for c in ['tema', 'local', 'tempo', 'categoria', 'status', 'descricao'] if c in df.columns]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Padronizar categoria para title case
    if 'categoria' in df.columns:
        df['categoria'] = df['categoria'].str.title()
        df['categoria'] = df['categoria'].str.replace('<->', ',')

    # Converter TEMPO para datetime
    if 'tempo' in df.columns:
        df['tempo'] = pd.to_datetime(df['tempo'], errors='coerce')

    # Extrair cidade e estado de LOCAL
    if 'local' in df.columns:
        extracted = df['local'].str.extract(r'^(.*) - ([A-Z]{2})$')
        df['cidade'] = extracted[0]
        df['estado'] = extracted[1]
        # Correção específica
        mask = (df['cidade'] == 'Fortaleza') & (df['estado'] == 'RJ')
        df.loc[mask, 'estado'] = 'CE'

    # Mapeamento dos dias da semana (se não existir já)
    dias_da_semana = {
        0: 'Domingo', 1: 'Segunda', 2: 'Terça',
        3: 'Quarta', 4: 'Quinta', 5: 'Sexta', 6: 'Sábado'
    }
    if 'dia_da_semana' in df.columns and 'dia_nome' not in df.columns:
        df['dia_nome'] = df['dia_da_semana'].map(dias_da_semana)
    elif 'dia_nome' in df.columns:
        # Padroniza capitalização
        df['dia_nome'] = df['dia_nome'].str.capitalize()

    # Remover nulos
    df = df.dropna(subset=['status', 'casos'] if all(c in df.columns for c in ['status', 'casos']) else [])

    # Colunas em maiúsculo
    df.columns = df.columns.str.upper()

    return df


def wordcloud_image(text: str) -> str:
    """Gera WordCloud e retorna como base64 PNG."""
    # Remove variações de "pão de açúcar" e derivados diretamente no texto
    text = re.sub(
        r'\b(p[aã]o\s*de\s*a[cç][uú]car|p[aã]odeac[uú]car|p[aã]o|a[cç][uú]car|acucar|paodeacucar|gpa|extra)\b',
        '', text, flags=re.IGNORECASE
    )
    wc = WordCloud(
        width=900, height=400, background_color="white",
        stopwords=PT_STOPS, max_words=100, collocations=False,
        colormap="RdBu_r",
    ).generate(text.strip() or "sem dados")
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tokenize(text: str) -> list[str]:
    """Tokeniza texto removendo stopwords e termos da marca."""
    MARCA_REGEX = re.compile(
        r'\b(p[aã]o\s*de\s*a[cç][uú]car|p[aã]odeac[uú]car|p[aã]o|a[cç][uú]car'
        r'|acucar|paodeacucar|gpa|extra)\b',
        re.IGNORECASE,
    )
    text = MARCA_REGEX.sub('', text)
    tokens = re.findall(r'\b[a-záéíóúàâêôãõüç]{3,}\b', text.lower())
    return [t for t in tokens if t not in PT_STOPS]


@st.cache_data
def compute_ngrams(texts: pd.Series, n: int, top_k: int = 50) -> pd.DataFrame:
    """Extrai os top_k n-gramas mais frequentes de uma série de textos."""
    counter: Counter = Counter()
    for doc in texts.dropna().astype(str):
        tokens = tokenize(doc)
        ngrams = zip(*[tokens[i:] for i in range(n)])
        counter.update(" ".join(gram) for gram in ngrams)
    rows = [{"ngrama": k, "frequencia": v} for k, v in counter.most_common(top_k)]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Filtros globais
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 ReclameAqui BI")
    st.markdown("---")
    st.subheader("🔎 Filtros globais")

DATA_PATH = 'dataset/RECLAMEAQUI_PAODEACUCAR.csv'

try:
    df_raw = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

df_raw["LEN_DESCRICAO"] = df_raw["DESCRICAO"].str.len() if "DESCRICAO" in df_raw.columns else 0
df_raw["FAIXA_TEXTO"] = pd.cut(
    df_raw["LEN_DESCRICAO"],
    bins=[0, 100, 300, 700, 99999],
    labels=["Curto (0-100)", "Médio (101-300)", "Longo (301-700)", "Muito longo (700+)"]
).astype(str)

with st.sidebar:
    # Filtro: Estado
    ufs_disp = sorted(df_raw["ESTADO"].dropna().unique().tolist()) if "ESTADO" in df_raw.columns else []
    uf_sel = st.multiselect("Estado", options=ufs_disp, default=ufs_disp)

    # Filtro: Status
    status_disp = sorted(df_raw["STATUS"].dropna().unique().tolist()) if "STATUS" in df_raw.columns else []
    status_sel = st.multiselect("Status", options=status_disp, default=status_disp)

    # Filtro: Faixa de tamanho do texto
    faixas_disp = ["Curto (0-100)", "Médio (101-300)", "Longo (301-700)", "Muito longo (700+)"]
    faixa_sel = st.multiselect("Faixa de tamanho do texto", options=faixas_disp, default=faixas_disp)

    st.markdown("---")
    st.caption("Projeto 1 — Ciência de Dados")


# ─────────────────────────────────────────────────────────────────────────────
# APLICAÇÃO DOS FILTROS
# ─────────────────────────────────────────────────────────────────────────────

df = df_raw.copy()
if uf_sel and "ESTADO" in df.columns:
    df = df[df["ESTADO"].isin(uf_sel)]
if status_sel and "STATUS" in df.columns:
    df = df[df["STATUS"].isin(status_sel)]
if faixa_sel and "FAIXA_TEXTO" in df.columns:
    df = df[df["FAIXA_TEXTO"].isin(faixa_sel)]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER E KPIs
# ─────────────────────────────────────────────────────────────────────────────

st.title("📋 Dashboard — Análise de Reclamações ReclameAqui")
st.markdown("Painel interativo de Business Intelligence · Filtros globais no menu lateral")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

total = int(df["CASOS"].sum()) if "CASOS" in df.columns else len(df)

if "STATUS" in df.columns and "CASOS" in df.columns:
    total_casos = df["CASOS"].sum()
    resolvidos = df[df["STATUS"].str.lower() == "resolvido"]["CASOS"].sum()
    resolvido_pct = resolvidos / max(total_casos, 1) * 100
else:
    resolvido_pct = 0

media_texto = df["LEN_DESCRICAO"].mean() if "LEN_DESCRICAO" in df.columns else 0
estados = df["ESTADO"].nunique() if "ESTADO" in df.columns else 0

with col1:
    st.metric("Total de reclamações", f"{total:,}")
with col2:
    st.metric("Taxa de resolução", f"{resolvido_pct:.1f}%")
with col3:
    st.metric("Tamanho médio do texto", f"{media_texto:.0f} chars")
with col4:
    st.metric("Estados representados", estados)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 1 — Série Temporal com Média Móvel
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("1 · Evolução temporal das reclamações")

if "TEMPO" in df.columns:
    df["_DATA"] = pd.to_datetime(df["TEMPO"], errors="coerce").dt.date
    serie = (
        df.groupby("_DATA")["CASOS"].sum()
        .reset_index()
        .rename(columns={"_DATA": "DATA"})
        .sort_values("DATA")
    )

    janela = st.slider("Janela da média móvel (dias)", 7, 90, 30, step=7, key="mm_slider")
    serie["MM"] = serie["CASOS"].rolling(janela, min_periods=1).mean()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=serie["DATA"], y=serie["CASOS"],
        name="Reclamações diárias",
        mode="lines",
        line=dict(color="#aec6e8", width=1),
        opacity=0.6,
    ))
    fig_ts.add_trace(go.Scatter(
        x=serie["DATA"], y=serie["MM"],
        name=f"Média móvel {janela}d",
        mode="lines",
        line=dict(color="#d62728", width=2.5),
    ))
    fig_ts.update_layout(
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Data", yaxis_title="Reclamações",
        template="plotly_white",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.warning("Coluna TEMPO não disponível para série temporal.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 2 — Mapa Coroplético do Brasil
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("2 · Distribuição geográfica por estado")

if "ESTADO" in df.columns and "ANO" in df.columns and "CASOS" in df.columns:
    anos_disp = sorted(df["ANO"].dropna().unique().tolist())

    if len(anos_disp) == 0:
        st.warning("Nenhum ano disponível nos dados filtrados.")
    else:
        if len(anos_disp) == 1:
            ano_sel = anos_disp[0]
            st.info(f"Apenas o ano **{ano_sel}** disponível nos dados filtrados.")
        else:
            ano_sel = st.select_slider("Ano", options=anos_disp, value=anos_disp[-1])

        df_mapa = df[df["ANO"] == ano_sel].groupby("ESTADO")["CASOS"].sum().reset_index()

        fig_map = px.choropleth(
            df_mapa,
            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/"
                    "master/public/data/brazil-states.geojson",
            locations="ESTADO",
            featureidkey="properties.sigla",
            color="CASOS",
            color_continuous_scale="Blues",
            title=f"Reclamações por estado — {ano_sel}",
            labels={"CASOS": "Reclamações"},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            height=480,
            margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title="Reclamações"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Colunas ESTADO, ANO ou CASOS não disponíveis para o mapa.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 3 — Pareto por Estado
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("3 · Pareto — estados mais críticos")

if "ESTADO" in df.columns and "CASOS" in df.columns:
    por_uf = (
        df.groupby("ESTADO")["CASOS"].sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    por_uf["ACUM_PCT"] = por_uf["CASOS"].cumsum() / por_uf["CASOS"].sum() * 100

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(x=por_uf["ESTADO"], y=por_uf["CASOS"],
               name="Reclamações", marker_color="#1f77b4"),
        secondary_y=False,
    )
    fig_pareto.add_trace(
        go.Scatter(x=por_uf["ESTADO"], y=por_uf["ACUM_PCT"],
                   name="% acumulado", mode="lines+markers",
                   marker_color="#d62728", line_width=2),
        secondary_y=True,
    )
    fig_pareto.add_hline(
        y=80, secondary_y=True, line_dash="dash",
        line_color="#d62728", opacity=0.5,
        annotation_text="80%",
    )
    fig_pareto.update_layout(
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(t=10, b=10),
    )
    fig_pareto.update_yaxes(title_text="Reclamações", secondary_y=False)
    fig_pareto.update_yaxes(title_text="% acumulado", ticksuffix="%", secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)
else:
    st.warning("Colunas ESTADO ou CASOS não disponíveis.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 4 — Proporção de Resoluções
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("4 · Proporção por tipo de status")

if "STATUS" in df.columns and "CASOS" in df.columns:
    status_cnt = df.groupby("STATUS")["CASOS"].sum().reset_index()
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = px.pie(
            status_cnt, names="STATUS", values="CASOS",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=340, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar_status = px.bar(
            status_cnt.sort_values("CASOS", ascending=True),
            x="CASOS", y="STATUS", orientation="h",
            color="STATUS",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True,
        )
        fig_bar_status.update_layout(
            height=340, showlegend=False,
            xaxis_title="Total de reclamações", yaxis_title="",
            template="plotly_white",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_bar_status, use_container_width=True)
else:
    st.warning("Colunas STATUS ou CASOS não disponíveis.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 5 — Análise Estatística de Textos
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("5 · Distribuição do tamanho da descrição por status")

if {"LEN_DESCRICAO", "STATUS"}.issubset(df.columns):
    top_status = df["STATUS"].value_counts().head(5).index.tolist()
    df_box = df[df["STATUS"].isin(top_status)]

    col_e, col_f = st.columns(2)

    with col_e:
        fig_box = px.box(
            df_box, x="STATUS", y="LEN_DESCRICAO",
            color="STATUS",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points=False,
        )
        fig_box.update_layout(
            height=380, showlegend=False,
            xaxis_title="Status", yaxis_title="Comprimento (chars)",
            template="plotly_white",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_f:
        fig_hist = px.histogram(
            df_box, x="LEN_DESCRICAO", color="STATUS",
            nbins=50, barmode="overlay", opacity=0.65,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_hist.update_layout(
            height=380,
            xaxis_title="Comprimento (chars)", yaxis_title="Frequência",
            template="plotly_white", legend_title="Status",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("**Estatísticas resumidas:**")
    resumo = (
        df_box.groupby("STATUS")["LEN_DESCRICAO"]
        .describe()
        .round(1)
        .rename(columns={
            "count": "N", "mean": "Média", "std": "Desvio padrão",
            "min": "Mín", "25%": "Q1", "50%": "Mediana",
            "75%": "Q3", "max": "Máx"
        })
    )
    st.dataframe(resumo, use_container_width=True)
else:
    st.warning("Colunas LEN_DESCRICAO ou STATUS não disponíveis.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 6 — Distribuição Temporal (Semana / Mês / Dia da Semana)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("6 · Distribuição temporal de reclamações")

ORDEM_DIAS = ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"]

NOMES_MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

colunas_ok = {
    "semana": "SEMANA_DO_ANO" in df.columns and "CASOS" in df.columns,
    "mes":    "MES" in df.columns and "CASOS" in df.columns,
    "dia":    "DIA_NOME" in df.columns and "CASOS" in df.columns,
}

if any(colunas_ok.values()):
    # ── Filtro de meses (usado por Semana e Dia da Semana) ────────────────────
    if "MES" in df.columns:
        meses_disp = sorted(df["MES"].dropna().unique().tolist())
        meses_labels = {m: NOMES_MESES.get(int(m), str(m)) for m in meses_disp}
        meses_sel = st.multiselect(
            "Filtrar por mês",
            options=meses_disp,
            default=meses_disp,
            format_func=lambda m: meses_labels.get(m, str(m)),
            key="comp6_meses",
        )
        df_c6 = df[df["MES"].isin(meses_sel)] if meses_sel else df
    else:
        df_c6 = df

    col_ctrl1, col_ctrl2 = st.columns([1, 3])

    with col_ctrl1:
        opcoes_view = []
        if colunas_ok["semana"]:
            opcoes_view.append("Semana do ano")
        if colunas_ok["mes"]:
            opcoes_view.append("Mês do ano")
        if colunas_ok["dia"]:
            opcoes_view.append("Dia da semana")

        view_tipo = st.radio(
            "Visualizar por:",
            options=opcoes_view,
            key="temporal_radio",
            horizontal=False,
        )

    # ── Semana do ano ─────────────────────────────────────────────────────────
    if view_tipo == "Semana do ano":
        with col_ctrl2:
            if "ANO" in df_c6.columns:
                anos_temp = sorted(df_c6["ANO"].dropna().unique().tolist())
                anos_temp_sel = st.multiselect(
                    "Filtrar por ano",
                    options=anos_temp,
                    default=anos_temp,
                    key="semana_anos",
                )
                df_sem = df_c6[df_c6["ANO"].isin(anos_temp_sel)] if anos_temp_sel else df_c6
            else:
                df_sem = df_c6
                st.empty()

        agg_semana = (
            df_sem.groupby(
                (["SEMANA_DO_ANO", "ANO"] if "ANO" in df_sem.columns else ["SEMANA_DO_ANO"])
            )["CASOS"].sum()
            .reset_index()
            .sort_values("SEMANA_DO_ANO")
        )

        if "ANO" in agg_semana.columns and agg_semana["ANO"].nunique() > 1:
            fig_sem = px.line(
                agg_semana, x="SEMANA_DO_ANO", y="CASOS",
                color="ANO", markers=True,
                labels={"SEMANA_DO_ANO": "Semana do ano", "CASOS": "Reclamações", "ANO": "Ano"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
        else:
            fig_sem = px.bar(
                agg_semana, x="SEMANA_DO_ANO", y="CASOS",
                labels={"SEMANA_DO_ANO": "Semana do ano", "CASOS": "Reclamações"},
                color_discrete_sequence=["#1f77b4"],
            )

        fig_sem.update_layout(
            height=380,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(dtick=4, title="Semana do ano"),
            yaxis_title="Reclamações",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_sem, use_container_width=True)

    # ── Mês do ano ────────────────────────────────────────────────────────────
    elif view_tipo == "Mês do ano":
        with col_ctrl2:
            if "ANO" in df.columns:
                anos_mes = sorted(df["ANO"].dropna().unique().tolist())
                anos_mes_sel = st.multiselect(
                    "Filtrar por ano",
                    options=anos_mes,
                    default=anos_mes,
                    key="mes_anos",
                )
                df_mes = df[df["ANO"].isin(anos_mes_sel)] if anos_mes_sel else df
            else:
                df_mes = df
                st.empty()

        group_cols = ["MES", "ANO"] if "ANO" in df_mes.columns else ["MES"]
        agg_mes = (
            df_mes.groupby(group_cols)["CASOS"].sum()
            .reset_index()
            .sort_values("MES")
        )
        # Adiciona rótulo de mês em português
        agg_mes["MES_NOME"] = agg_mes["MES"].apply(
            lambda m: NOMES_MESES.get(int(m), str(m))
        )
        # Categoria ordenada pelos meses (1-12)
        agg_mes["MES_NOME"] = pd.Categorical(
            agg_mes["MES_NOME"],
            categories=[NOMES_MESES[i] for i in range(1, 13)],
            ordered=True,
        )

        if "ANO" in agg_mes.columns and agg_mes["ANO"].nunique() > 1:
            fig_mes = px.line(
                agg_mes, x="MES_NOME", y="CASOS",
                color="ANO", markers=True,
                labels={"MES_NOME": "Mês", "CASOS": "Reclamações", "ANO": "Ano"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
        else:
            fig_mes = px.bar(
                agg_mes, x="MES_NOME", y="CASOS",
                text_auto=".0f",
                labels={"MES_NOME": "Mês", "CASOS": "Reclamações"},
                color="MES_NOME",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )

        fig_mes.update_layout(
            height=380,
            showlegend=agg_mes["ANO"].nunique() > 1 if "ANO" in agg_mes.columns else False,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Mês",
            yaxis_title="Reclamações",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_mes, use_container_width=True)

    # ── Dia da semana ─────────────────────────────────────────────────────────
    elif view_tipo == "Dia da semana":
        with col_ctrl2:
            agg_tipo = st.radio(
                "Agregação:",
                options=["Total", "Média por semana"],
                horizontal=True,
                key="dia_agg",
            )

        agg_dia = df_c6.groupby("DIA_NOME")["CASOS"]

        if agg_tipo == "Total":
            agg_dia = agg_dia.sum().reset_index()
            y_label = "Total de reclamações"
        else:
            n_semanas = df_c6["SEMANA_DO_ANO"].nunique() if "SEMANA_DO_ANO" in df_c6.columns else 1
            agg_dia = (agg_dia.sum() / max(n_semanas, 1)).reset_index()
            y_label = "Média de reclamações por semana"

        agg_dia["DIA_NOME"] = pd.Categorical(
            agg_dia["DIA_NOME"],
            categories=ORDEM_DIAS,
            ordered=True,
        )
        agg_dia = agg_dia.sort_values("DIA_NOME")

        fig_dia = px.bar(
            agg_dia, x="DIA_NOME", y="CASOS",
            text_auto=".0f",
            labels={"DIA_NOME": "Dia da semana", "CASOS": y_label},
            color="DIA_NOME",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_dia.update_traces(textposition="outside")
        fig_dia.update_layout(
            height=380,
            showlegend=False,
            template="plotly_white",
            xaxis_title="Dia da semana",
            yaxis_title=y_label,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_dia, use_container_width=True)

else:
    st.warning("Colunas SEMANA_DO_ANO, MES ou DIA_NOME não disponíveis.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 7 — WordCloud (NLP básica)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("7 · WordCloud — palavras mais frequentes nas reclamações")

if "DESCRICAO" in df.columns:
    wc_status_filter = st.selectbox(
        "Filtrar WordCloud por status",
        options=["Todos"] + sorted(df["STATUS"].unique().tolist()),
        key="wc_filter",
    )

    df_wc = df if wc_status_filter == "Todos" else df[df["STATUS"] == wc_status_filter]
    texto = " ".join(df_wc["DESCRICAO"].dropna().astype(str).str.lower())

    if texto.strip():
        b64 = wordcloud_image(texto)
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:8px">',
            unsafe_allow_html=True,
        )
    else:
        st.info("Sem texto disponível para o filtro selecionado.")
else:
    st.warning("Coluna DESCRICAO não disponível.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 8 — Ranking de N-gramas (Bigramas / Trigramas)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("8 · Ranking de expressões mais frequentes (N-gramas)")
st.markdown(
    "Identifica **padrões de co-ocorrência** de palavras nas descrições das reclamações, "
    "revelando temas recorrentes e problemas estruturais reportados pelos consumidores."
)

if "DESCRICAO" in df.columns:
    c8_col1, c8_col2, c8_col3 = st.columns([1.2, 1, 1.2])

    with c8_col1:
        ng_tipo = st.radio(
            "Tipo de N-grama",
            options=["Bigrama (2 palavras)", "Trigrama (3 palavras)"],
            horizontal=False,
            key="ng_tipo",
        )
        n_val = 2 if ng_tipo.startswith("Bigrama") else 3

    with c8_col2:
        top_k = st.slider(
            "Quantidade de expressões",
            min_value=5, max_value=40, value=20, step=5,
            key="ng_topk",
        )

    with c8_col3:
        ng_status = st.selectbox(
            "Filtrar por status",
            options=["Todos"] + sorted(df["STATUS"].dropna().unique().tolist()),
            key="ng_status",
        )

    df_ng = df if ng_status == "Todos" else df[df["STATUS"] == ng_status]

    if df_ng.empty:
        st.info("Nenhum dado disponível para o filtro selecionado.")
    else:
        with st.spinner("Calculando n-gramas…"):
            # cache_key baseado nos parâmetros para aproveitar o cache do Streamlit
            df_ngrams = compute_ngrams(df_ng["DESCRICAO"], n=n_val, top_k=top_k)

        if df_ngrams.empty:
            st.info("Sem n-gramas suficientes para os filtros aplicados.")
        else:
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig_ng = px.bar(
                    df_ngrams.sort_values("frequencia"),
                    x="frequencia",
                    y="ngrama",
                    orientation="h",
                    text="frequencia",
                    color="frequencia",
                    color_continuous_scale="Blues",
                    labels={"frequencia": "Frequência", "ngrama": "Expressão"},
                )
                fig_ng.update_traces(textposition="outside")
                fig_ng.update_layout(
                    height=max(380, top_k * 22),
                    template="plotly_white",
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis_title="Frequência (nº de ocorrências)",
                    yaxis_title="",
                    margin=dict(t=10, b=10, l=10, r=60),
                )
                st.plotly_chart(fig_ng, use_container_width=True)

            with col_table:
                st.markdown("**Tabela completa**")
                st.dataframe(
                    df_ngrams.rename(columns={"ngrama": "Expressão", "frequencia": "Freq."})
                    .reset_index(drop=True),
                    use_container_width=True,
                    height=max(380, top_k * 22),
                )

            # Insight automático
            top1 = df_ngrams.iloc[0]
            top2 = df_ngrams.iloc[1] if len(df_ngrams) > 1 else None
            insight = (
                f"A expressão mais recorrente é **\"{top1['ngrama']}\"** "
                f"({int(top1['frequencia']):,} ocorrências)"
            )
            if top2 is not None:
                insight += (
                    f", seguida de **\"{top2['ngrama']}\"** "
                    f"({int(top2['frequencia']):,} ocorrências)."
                )
            st.info(f"💡 {insight}")
else:
    st.warning("Coluna DESCRICAO não disponível para análise de n-gramas.")


# ─────────────────────────────────────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Dashboard desenvolvido para o Projeto 1 — Análise Exploratória ReclameAqui · "
    "Ciência de Dados"
)