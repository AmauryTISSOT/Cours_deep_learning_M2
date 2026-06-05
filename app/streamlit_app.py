"""Dashboard de démonstration — détection de SMS spam / scam (multilingue).

Lancement (depuis la racine du projet) :
    streamlit run app/streamlit_app.py

Charge les modèles exportés depuis ../artifacts et permet de tester un SMS
en direct : verdict + probabilité + jauge, choix du modèle, seuil ajustable,
exemples cliquables et comparaison des trois modèles.
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from inference import MODEL_LABELS, SpamModels

COULEUR_SPAM = "#d62728"   # rouge
COULEUR_HAM = "#2ca02c"    # vert

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Exemples cliquables (multilingues) : (libellé, texte, est_spam_attendu)
EXEMPLES = [
    ("🇫🇷 Arnaque colis", "URGENT : votre colis est bloque. Reglez 1,99EUR de frais de douane ici : http://suivi-colis.fr/payer", True),
    ("🇬🇧 Prize scam", "Congratulations! You've WON a 1000 GBP gift card. Click http://bit.ly/claim-now before it expires!", True),
    ("🇪🇸 Premio falso", "Felicidades! Has ganado un premio. Llama ahora al 900123456 para reclamarlo.", True),
    ("🇫🇷 Message normal", "Je serai un peu en retard ce soir, commence sans moi 😊", False),
    ("🇬🇧 Message normal", "Hey, are we still meeting at 6pm for dinner? Let me know.", False),
    ("🇪🇸 Message normal", "Puedes recogerme en la estacion a las 8? Gracias.", False),
]


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_models() -> SpamModels:
    """Charge les trois modèles une seule fois (mis en cache pour toute la session)."""
    return SpamModels(ARTIFACTS_DIR)


def set_exemple(texte: str) -> None:
    st.session_state.sms_text = texte


def jauge_proba(proba: float, seuil: float) -> go.Figure:
    """Jauge à aiguille : zones colorées + trait noir matérialisant le seuil."""
    couleur = COULEUR_SPAM if proba >= seuil else COULEUR_HAM
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": " %", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": " %"},
            "bar": {"color": couleur},
            "steps": [
                {"range": [0, 50], "color": "#e8f5e9"},
                {"range": [50, 80], "color": "#fff3cd"},
                {"range": [80, 100], "color": "#fdecea"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.85, "value": seuil * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def afficher_verdict(proba: float, seuil: float) -> None:
    """Badge coloré (SPAM / HAM) + jauge de probabilité de spam."""
    if proba >= seuil:
        st.error(f"🔴 **SPAM** — probabilité de spam : **{proba:.1%}**  (seuil {seuil:.0%})")
    else:
        st.success(f"🟢 **HAM** (légitime) — probabilité de spam : **{proba:.1%}**  (seuil {seuil:.0%})")
    st.plotly_chart(jauge_proba(proba, seuil), use_container_width=True)


def barres_comparaison(probas: dict, seuil: float) -> go.Figure:
    """Histogramme horizontal des P(spam) des trois modèles (barres colorées par verdict)."""
    keys = list(probas.keys())
    labels = [MODEL_LABELS[k] for k in keys]
    vals = [probas[k] * 100 for k in keys]
    couleurs = [COULEUR_SPAM if probas[k] >= seuil else COULEUR_HAM for k in keys]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=couleurs,
        text=[f"{v:.0f} %" for v in vals], textposition="auto",
    ))
    fig.add_vline(x=seuil * 100, line_dash="dash", line_color="black",
                  annotation_text=f"seuil {seuil:.0%}", annotation_position="top")
    fig.update_layout(xaxis_title="Probabilité de spam (%)", xaxis_range=[0, 100],
                      height=260, margin=dict(t=30, b=40, l=10, r=10))
    return fig


def barres_explicabilite(contribs: list) -> go.Figure:
    """Barres divergentes des termes les plus influents (rouge = spam, vert = ham)."""
    contribs = list(reversed(contribs))   # le plus influent en haut du graphe horizontal
    termes = [c[0] for c in contribs]
    vals = [c[1] for c in contribs]
    couleurs = [COULEUR_SPAM if v > 0 else COULEUR_HAM for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=termes, orientation="h", marker_color=couleurs))
    fig.update_layout(
        xaxis_title="Contribution au score (tfidf × coefficient)",
        height=max(220, 30 * len(termes)), margin=dict(t=20, b=40, l=10, r=10),
    )
    return fig


# --- Configuration de la page ---
st.set_page_config(page_title="Détecteur de SMS spam", page_icon="📨", layout="centered")
st.title("📨 Détecteur de SMS spam / scam")
st.caption("Démonstration multilingue — comparaison ML · RNN · Transformer (MiniLM)")

if not ARTIFACTS_DIR.exists():
    st.error(
        f"Dossier introuvable : `{ARTIFACTS_DIR}`.\n\n"
        "Exécute d'abord la cellule d'export à la fin du notebook pour générer les modèles."
    )
    st.stop()

models = get_models()

# --- Barre latérale : réglages ---
with st.sidebar:
    st.header("⚙️ Réglages")
    model_key = st.radio(
        "Modèle",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda k: MODEL_LABELS[k],
        index=2,  # Transformer par défaut (le plus performant)
    )
    seuil = st.slider(
        "Seuil de décision (spam si proba ≥ seuil)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    )
    comparer = st.checkbox("Comparer les 3 modèles", value=False)
    st.divider()
    st.caption(f"Calcul sur : **{str(models.device).upper()}**")

# --- Zone de saisie ---
st.session_state.setdefault("sms_text", EXEMPLES[0][1])

st.subheader("Message à analyser")
st.write("Essaie un exemple :")
cols = st.columns(3)
for i, (libelle, texte, _) in enumerate(EXEMPLES):
    cols[i % 3].button(libelle, use_container_width=True, on_click=set_exemple, args=(texte,))

sms = st.text_area("SMS", key="sms_text", height=120, label_visibility="collapsed")
analyser = st.button("🔍 Analyser", type="primary", use_container_width=True)

# --- Résultats ---
if analyser:
    if not sms.strip():
        st.warning("Saisis un message (ou choisis un exemple) avant d'analyser.")
    elif comparer:
        st.subheader("Comparaison des trois modèles")
        probas = models.predict_all(sms)
        st.plotly_chart(barres_comparaison(probas, seuil), use_container_width=True)
    else:
        st.subheader(f"Résultat — {MODEL_LABELS[model_key]}")
        proba = models.predict(model_key, sms)
        afficher_verdict(proba, seuil)

        st.markdown("##### Pourquoi ce verdict ?")
        if model_key == "ml":
            contribs = models.explain_ml(sms, top_k=10)
            if contribs:
                st.plotly_chart(barres_explicabilite(contribs), use_container_width=True)
                st.caption("Rouge = pousse vers *spam*, vert = vers *ham*. "
                           "Calculé à partir des coefficients de la régression logistique.")
            else:
                st.info("Aucun terme connu du modèle dans ce message (que des mots hors vocabulaire).")
        else:
            st.info("Explicabilité par mot disponible uniquement pour le modèle ML "
                    "(le RNN et le Transformer ne sont pas linéaires).")
