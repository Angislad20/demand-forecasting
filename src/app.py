import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prévision des ventes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette de couleurs (thème sombre)
COLOR_BG = "#0b1220"        # fond de page
COLOR_CARD = "#131c31"      # fond des cartes
COLOR_BORDER = "#1e2a44"    # bordures des cartes
COLOR_GRID = "#1e2a44"      # lignes de grille des graphiques
COLOR_PRIMARY = "#38bdf8"   # bleu cyan : prévisions et chiffres clés
COLOR_SECONDARY = "#818cf8" # violet : graphique mensuel
COLOR_TEXT = "#e2e8f0"      # texte principal
COLOR_MUTED = "#94a3b8"     # texte secondaire
COLOR_HIST = "#64748b"      # gris : historique
COLOR_ALERT = "#fb7185"     # rose/rouge : seuil d'alerte
COLOR_SUCCESS = "#34d399"   # vert : économies

# Taux d'erreur mesurés dans le notebook 03 (WAPE sur la validation T4 2017)
WAPE_BASELINE = 0.1528      # méthode classique : ventes de l'année précédente
WAPE_MODELE = 0.1077        # modèle LightGBM

JOURS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MOIS_FR = {1: "Janvier", 2: "Février", 3: "Mars"}

# ---------------------------------------------------------------------------
# Style CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {COLOR_BG}; }}
    [data-testid="stSidebar"] {{ background: #070c17; border-right: 1px solid {COLOR_BORDER}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* Bandeau de titre */
    .hero {{
        background: linear-gradient(135deg, {COLOR_CARD} 0%, #0f2747 100%);
        border: 1px solid {COLOR_BORDER};
        border-left: 4px solid {COLOR_PRIMARY};
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 26px;
    }}
    .hero h1 {{ color: {COLOR_TEXT}; font-size: 1.8rem; font-weight: 700; margin: 0 0 6px 0; padding: 0; }}
    .hero p  {{ color: {COLOR_MUTED}; font-size: 1rem; margin: 0; }}

    /* Cartes indicateurs */
    .kpi {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
    }}
    .kpi-label {{ color: {COLOR_MUTED}; font-size: 0.8rem; font-weight: 500;
                  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }}
    .kpi-value {{ color: {COLOR_PRIMARY}; font-size: 2rem; font-weight: 700; line-height: 1.1; }}
    .kpi-unit  {{ color: {COLOR_MUTED}; font-size: 0.95rem; font-weight: 500; margin-left: 6px; }}
    .kpi-sub   {{ color: {COLOR_MUTED}; font-size: 0.85rem; margin-top: 10px; }}
    .kpi.accent {{ border: 1px solid {COLOR_ALERT}; }}
    .kpi.accent .kpi-value {{ color: {COLOR_ALERT}; }}

    /* Titres de section */
    .section-title {{ color: {COLOR_TEXT}; font-size: 1.15rem; font-weight: 600; margin: 34px 0 4px 0; }}
    .section-sub   {{ color: {COLOR_MUTED}; font-size: 0.9rem; margin-bottom: 14px; }}

    /* Encadrés graphiques */
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 14px;
    }}

    /* Encadrés texte */
    .panel {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
    }}
    .panel h4 {{ color: {COLOR_TEXT}; font-size: 1rem; font-weight: 600; margin: 0 0 12px 0; }}
    .panel p, .panel li {{ color: #cbd5e1; font-size: 0.93rem; line-height: 1.6; }}
    .panel b {{ color: {COLOR_PRIMARY}; }}
    .row {{ display: flex; justify-content: space-between; padding: 9px 0;
            border-bottom: 1px solid {COLOR_BORDER}; font-size: 0.93rem; color: #cbd5e1; }}
    .row:last-child {{ border-bottom: none; }}
    .row b {{ color: {COLOR_TEXT}; }}
    .gain {{ color: {COLOR_SUCCESS}; font-weight: 700; font-size: 1.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt(n: float) -> str:
    """Formate un nombre avec un espace comme séparateur de milliers."""
    return f"{n:,.0f}".replace(",", " ")


def kpi_card(label: str, value: str, unit: str = "", sub: str = "", accent: bool = False) -> str:
    """Génère le HTML d'une carte indicateur."""
    css = "kpi accent" if accent else "kpi"
    return f"""
    <div class="{css}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "raw", "train.csv")
PRED_PATH = os.path.join(DATA_DIR, "processed", "predictions_test.csv")


@st.cache_data
def load_data():
    """Charge l'historique des ventes et les prédictions du modèle."""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(PRED_PATH):
        return None, None
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    preds = pd.read_csv(PRED_PATH, parse_dates=["date"])
    return train, preds


train_df, preds_df = load_data()

if train_df is None or preds_df is None:
    st.error(
        "Fichiers de données introuvables. Exécutez le notebook 03-modelling "
        "pour générer data/processed/predictions_test.csv."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sélection")
    selected_store = st.selectbox("Magasin", sorted(train_df["store"].unique()), index=0)
    selected_item = st.selectbox("Article", sorted(train_df["item"].unique()), index=0)

    st.markdown("---")
    st.markdown("### Hypothèses financières")
    marge_unitaire = st.slider(
        "Marge perdue par article manquant (€)", 5.0, 50.0, 15.0, 1.0,
        help="Ce que le magasin perd quand un client veut acheter mais que le rayon est vide.",
    )
    cout_stockage = st.slider(
        "Coût d'un article invendu en stock (€)", 0.5, 10.0, 2.0, 0.5,
        help="Ce que coûte un article commandé en trop (place, immobilisation d'argent).",
    )

# ---------------------------------------------------------------------------
# Préparation des données filtrées
# ---------------------------------------------------------------------------
hist = train_df[
    (train_df["store"] == selected_store)
    & (train_df["item"] == selected_item)
    & (train_df["date"] >= "2017-07-01")
].copy()

pred = preds_df[
    (preds_df["store"] == selected_store) & (preds_df["item"] == selected_item)
].copy()

total_pred = pred["sales_predicted"].sum()
mean_daily = pred["sales_predicted"].mean()
peak = pred.loc[pred["sales_predicted"].idxmax()]
peak_date = peak["date"]

# Même période l'année précédente (T1 2017) pour comparer
same_period_ly = train_df[
    (train_df["store"] == selected_store)
    & (train_df["item"] == selected_item)
    & (train_df["date"] >= "2017-01-01")
    & (train_df["date"] <= "2017-03-31")
]["sales"].sum()
evolution = (total_pred / same_period_ly - 1) * 100 if same_period_ly else 0

# Stock de sécurité : 3 jours de ventes moyennes + marge liée à la variabilité récente
std_recent = hist.tail(30)["sales"].std()
stock_securite = int(np.ceil(mean_daily * 3 + 1.65 * std_recent))

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>Prévision des ventes et réapprovisionnement</h1>
        <p>Magasin {selected_store} &nbsp;·&nbsp; Article {selected_item}
        &nbsp;·&nbsp; Horizon : janvier à mars 2018 (90 jours)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Indicateurs clés
# ---------------------------------------------------------------------------
signe = "+" if evolution >= 0 else ""
c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card("Ventes prévues sur 90 jours", fmt(total_pred), "articles",
             f"{signe}{evolution:.1f} % vs même période 2017"),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Moyenne par jour", f"{mean_daily:.1f}", "articles",
             "Rythme de vente quotidien attendu"),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card("Journée la plus chargée", f"{peak['sales_predicted']:.0f}", "articles",
             f"{JOURS_FR[peak_date.dayofweek]} {peak_date.strftime('%d/%m/%Y')}"),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("Stock de sécurité conseillé", fmt(stock_securite), "articles",
             "Seuil en dessous duquel il faut recommander", accent=True),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Graphique principal : historique + prévisions
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Évolution des ventes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Ventes réelles du second semestre 2017, puis prévisions du modèle '
    "pour le premier trimestre 2018. Survolez la courbe pour voir le détail jour par jour.</div>",
    unsafe_allow_html=True,
)

chart_df = pd.concat(
    [
        pd.DataFrame({"Date": hist["date"], "Ventes": hist["sales"], "Série": "Ventes réelles (2017)"}),
        pd.DataFrame({"Date": pred["date"], "Ventes": pred["sales_predicted"].round(1),
                      "Série": "Prévisions (2018)"}),
    ]
)

color_scale = alt.Scale(
    domain=["Ventes réelles (2017)", "Prévisions (2018)"],
    range=[COLOR_HIST, COLOR_PRIMARY],
)

lines = (
    alt.Chart(chart_df)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y", labelColor=COLOR_MUTED, domainColor=COLOR_BORDER, grid=False)),
        y=alt.Y("Ventes:Q", title="Articles vendus par jour",
                axis=alt.Axis(labelColor=COLOR_MUTED, titleColor=COLOR_MUTED, gridColor=COLOR_GRID, domainColor=COLOR_BORDER)),
        color=alt.Color("Série:N", scale=color_scale,
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12, labelColor=COLOR_TEXT)),
        tooltip=[alt.Tooltip("Date:T", format="%d/%m/%Y"), "Série:N", alt.Tooltip("Ventes:Q", format=".0f")],
    )
)

seuil = (
    alt.Chart(pd.DataFrame({"y": [stock_securite]}))
    .mark_rule(color=COLOR_ALERT, strokeDash=[6, 4], strokeWidth=1.5)
    .encode(y="y:Q")
)
seuil_label = (
    alt.Chart(pd.DataFrame({"y": [stock_securite], "Date": [hist["date"].min()]}))
    .mark_text(align="left", dx=4, dy=-8, color=COLOR_ALERT, fontSize=11, fontWeight=600)
    .encode(x="Date:T", y="y:Q", text=alt.value(f"Stock de sécurité : {stock_securite} articles"))
)
separation = (
    alt.Chart(pd.DataFrame({"Date": [pd.Timestamp("2018-01-01")]}))
    .mark_rule(color=COLOR_MUTED, strokeDash=[2, 3], strokeWidth=1)
    .encode(x="Date:T")
)

main_chart = (lines + seuil + seuil_label + separation).properties(height=380).configure_view(strokeWidth=0).configure(background="transparent")
st.altair_chart(main_chart, width="stretch")

# ---------------------------------------------------------------------------
# Graphiques secondaires : par jour de la semaine et par mois
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Ventes moyennes selon le jour</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Quels jours le rayon se vide le plus vite.</div>',
                unsafe_allow_html=True)
    dow = pred.assign(Jour=pred["date"].dt.dayofweek.map(lambda d: JOURS_FR[d]))
    dow = dow.groupby("Jour", as_index=False)["sales_predicted"].mean()
    dow_chart = (
        alt.Chart(dow)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=COLOR_PRIMARY)
        .encode(
            x=alt.X("Jour:N", sort=JOURS_FR, title=None, axis=alt.Axis(labelAngle=0, labelColor=COLOR_MUTED, domainColor=COLOR_BORDER)),
            y=alt.Y("sales_predicted:Q", title="Articles / jour",
                    axis=alt.Axis(labelColor=COLOR_MUTED, titleColor=COLOR_MUTED, gridColor=COLOR_GRID, domainColor=COLOR_BORDER)),
            tooltip=["Jour:N", alt.Tooltip("sales_predicted:Q", title="Moyenne", format=".1f")],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0).configure(background="transparent")
    )
    st.altair_chart(dow_chart, width="stretch")

with right:
    st.markdown('<div class="section-title">Volume prévu par mois</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Quantités à prévoir pour chaque mois du trimestre.</div>',
                unsafe_allow_html=True)
    month = pred.assign(Mois=pred["date"].dt.month.map(MOIS_FR))
    month = month.groupby("Mois", as_index=False)["sales_predicted"].sum()
    month_chart = (
        alt.Chart(month)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=COLOR_SECONDARY, size=60)
        .encode(
            x=alt.X("Mois:N", sort=list(MOIS_FR.values()), title=None,
                    axis=alt.Axis(labelAngle=0, labelColor=COLOR_MUTED, domainColor=COLOR_BORDER)),
            y=alt.Y("sales_predicted:Q", title="Articles",
                    axis=alt.Axis(labelColor=COLOR_MUTED, titleColor=COLOR_MUTED, gridColor=COLOR_GRID, domainColor=COLOR_BORDER)),
            tooltip=["Mois:N", alt.Tooltip("sales_predicted:Q", title="Total", format=",.0f")],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0).configure(background="transparent")
    )
    st.altair_chart(month_chart, width="stretch")

# ---------------------------------------------------------------------------
# Recommandation + impact financier
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Décision et impact</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Ce que le responsable du magasin doit retenir.</div>',
            unsafe_allow_html=True)

# Erreur attendue en nombre d'articles = volume x taux d'erreur moyen (WAPE)
# On suppose que l'erreur se répartit moitié en manque, moitié en surplus.
cout_unitaire_erreur = (marge_unitaire + cout_stockage) / 2
erreur_baseline = total_pred * WAPE_BASELINE
erreur_modele = total_pred * WAPE_MODELE
cout_baseline = erreur_baseline * cout_unitaire_erreur
cout_modele = erreur_modele * cout_unitaire_erreur
gain = cout_baseline - cout_modele

d1, d2 = st.columns(2)

with d1:
    st.markdown(
        f"""
        <div class="panel">
            <h4>Recommandation de réapprovisionnement</h4>
            <p>Garder en permanence au moins <b>{stock_securite} articles</b> en réserve.
            Dès que le stock passe sous ce seuil, passer une nouvelle commande.</p>
            <p>Prévoir un renfort de stock avant le <b>{JOURS_FR[peak_date.dayofweek].lower()}
            {peak_date.strftime('%d/%m')}</b>, journée la plus chargée du trimestre
            ({peak['sales_predicted']:.0f} articles attendus).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        f"""
        <div class="panel">
            <h4>Coût des erreurs de prévision sur 90 jours</h4>
            <div class="row"><span>Méthode classique (ventes de l'an dernier)</span>
                <b>{fmt(cout_baseline)} €</b></div>
            <div class="row"><span>Prévisions du modèle</span><b>{fmt(cout_modele)} €</b></div>
            <div class="row"><span>Économie estimée sur cet article</span>
                <span class="gain">{fmt(gain)} €</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    f"Estimation simplifiée : erreur moyenne mesurée en validation ({WAPE_BASELINE:.1%} pour la méthode "
    f"classique, {WAPE_MODELE:.1%} pour le modèle), appliquée au volume prévu, "
    "avec une erreur répartie à parts égales entre manque et surplus."
)
