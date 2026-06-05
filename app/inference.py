"""Chargement des modèles exportés et inférence.

Module volontairement indépendant de Streamlit (testable / réutilisable). Le cache
applicatif (`@st.cache_resource`) est géré côté `streamlit_app.py`.

Trois modèles entraînés sur le même dataset multilingue (`df_big`) :
- ML          : TF-IDF + régression logistique ElasticNet     (entrée : texte nettoyé)
- RNN         : Embedding -> LSTM -> pooling masqué -> dense   (entrée : texte nettoyé)
- Transformer : MiniLM multilingue fine-tuné                   (entrée : texte BRUT)
"""

import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from preprocessing import clean_text, encode_for_rnn


# --- Architecture du RNN : identique à celle entraînée dans le notebook ---
class RNNSpam(nn.Module):
    """Embedding -> LSTM -> mean-pooling masqué -> LayerNorm -> Dense(ReLU) -> Dropout -> logit.

    On moyenne les sorties du LSTM sur les vrais tokens uniquement (le masque ignore
    le padding) : prendre le dernier état caché diluerait le signal après le padding.
    """

    def __init__(self, taille_vocab, dim_embedding=64, dim_lstm=64, pad=0):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocab, dim_embedding, padding_idx=pad)
        self.lstm = nn.LSTM(dim_embedding, dim_lstm, batch_first=True)
        self.norm = nn.LayerNorm(dim_lstm)
        self.dense = nn.Linear(dim_lstm, 32)
        self.dropout = nn.Dropout(0.3)
        self.sortie = nn.Linear(32, 1)
        self._pad = pad

    def forward(self, x):
        e = self.embedding(x)
        sorties, _ = self.lstm(e)                            # (B, T, H)
        masque = (x != self._pad).unsqueeze(-1).float()      # (B, T, 1)
        z = (sorties * masque).sum(1) / masque.sum(1).clamp(min=1)  # moyenne sur les vrais tokens
        z = self.norm(z)
        z = torch.relu(self.dense(z))
        z = self.dropout(z)
        return self.sortie(z).squeeze(1)                     # logit


# Libellés affichables des modèles (clé interne -> nom lisible)
MODEL_LABELS = {
    "ml": "ML — TF-IDF + ElasticNet",
    "rnn": "RNN — LSTM",
    "transformer": "Transformer — MiniLM",
}


class SpamModels:
    """Charge les trois modèles depuis `artifacts/` et fournit les prédictions.

    Chaque `predict_*` renvoie la **probabilité de spam** (float dans [0, 1]).
    """

    def __init__(self, artifacts_dir, device=None):
        self.dir = Path(artifacts_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self._load_ml()
        self._load_rnn()
        self._load_transformer()

    # --- Chargement ---
    def _load_ml(self):
        self.ml_vectorizer = joblib.load(self.dir / "ml_vectorizer.joblib")
        self.ml_model = joblib.load(self.dir / "ml_baseline.joblib")
        # L'ordre des classes de sklearn n'est pas garanti : on repère l'indice "spam".
        self.ml_spam_idx = list(self.ml_model.classes_).index("spam")

    def _load_rnn(self):
        cfg = json.loads((self.dir / "rnn_config.json").read_text(encoding="utf-8"))
        self.rnn_vocab = json.loads((self.dir / "rnn_vocab.json").read_text(encoding="utf-8"))
        self.rnn_cfg = cfg
        model = RNNSpam(cfg["vocab_size"], cfg["dim_embedding"], cfg["dim_lstm"], pad=cfg["pad"])
        state = torch.load(self.dir / "rnn_state.pt", map_location=self.device)
        model.load_state_dict(state)
        self.rnn_model = model.to(self.device).eval()

    def _load_transformer(self):
        # Import local : transformers est lourd, on ne le charge que si nécessaire.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tdir = self.dir / "transformer"
        self.tf_cfg = json.loads((self.dir / "transformer_config.json").read_text(encoding="utf-8"))
        self.tf_tokenizer = AutoTokenizer.from_pretrained(tdir)
        self.tf_model = (
            AutoModelForSequenceClassification.from_pretrained(tdir).to(self.device).eval()
        )

    # --- Inférence (probabilité de spam) ---
    def predict_ml(self, text: str) -> float:
        vec = self.ml_vectorizer.transform([clean_text(text)])
        return float(self.ml_model.predict_proba(vec)[0, self.ml_spam_idx])

    def predict_rnn(self, text: str) -> float:
        x = encode_for_rnn(
            clean_text(text), self.rnn_vocab, self.rnn_cfg["max_len"],
            pad=self.rnn_cfg["pad"], oov=self.rnn_cfg["oov"],
        ).to(self.device)
        with torch.no_grad():
            logit = self.rnn_model(x)
        return float(torch.sigmoid(logit)[0])

    def predict_transformer(self, text: str) -> float:
        # Texte BRUT : la tokenisation sous-mots exploite casse, ponctuation et emojis.
        enc = self.tf_tokenizer(
            [text], truncation=True, padding="max_length",
            max_length=self.tf_cfg["max_len"], return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.tf_model(**enc).logits
        # Index 1 = "spam" (cf. transformer_config.json)
        return float(torch.softmax(logits, dim=1)[0, 1])

    def predict(self, model_key: str, text: str) -> float:
        return {
            "ml": self.predict_ml,
            "rnn": self.predict_rnn,
            "transformer": self.predict_transformer,
        }[model_key](text)

    def predict_all(self, text: str) -> dict:
        return {k: self.predict(k, text) for k in ("ml", "rnn", "transformer")}

    # --- Explicabilité (modèle ML linéaire uniquement) ---
    def explain_ml(self, text: str, top_k: int = 10):
        """Contribution de chaque terme du message au verdict du modèle ML.

        Pour une régression logistique, le log-odds de la classe spam est une somme
        linéaire : contribution(terme) = poids_tfidf(terme) × coefficient(terme).
        Renvoie la liste [(terme, contribution_signée)] des `top_k` termes les plus
        influents (en valeur absolue). Positif -> pousse vers spam, négatif -> vers ham.
        """
        vec = self.ml_vectorizer.transform([clean_text(text)]).tocoo()
        coefs = self.ml_model.coef_[0]            # poids de la classe positive (classes_[1])
        signe = 1.0 if self.ml_spam_idx == 1 else -1.0   # garantit : positif = vers spam
        feats = self.ml_vectorizer.get_feature_names_out()
        contribs = [(feats[i], signe * float(v * coefs[i])) for i, v in zip(vec.col, vec.data)]
        contribs.sort(key=lambda c: abs(c[1]), reverse=True)
        return contribs[:top_k]
