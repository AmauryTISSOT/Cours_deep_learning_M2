"""Prétraitement partagé entre le notebook (entraînement) et le dashboard (inférence).

Cohérence train/inférence : ML et RNN doivent recevoir EXACTEMENT le même texte
qu'à l'entraînement, sinon les prédictions se dégradent silencieusement.
- `clean_text` est repris à l'identique de la cellule de nettoyage du notebook.
- `encode_for_rnn` réplique la tokenisation `encoder_sequences` (lower + \\w+).
Le Transformer, lui, n'utilise rien de tout cela : il prend le texte BRUT.
"""

import re
import unicodedata

import torch


def clean_text(text: str) -> str:
    """Nettoyage robuste d'un message SMS (identique au notebook).

    On normalise la *structure* du texte (encodage, espaces, caractères de
    contrôle) SANS détruire les signaux utiles au classifieur (URLs, numéros,
    symboles monétaires £/$, MAJUSCULES...), souvent de bons indices de spam.
    """
    # 1. Garantir une chaîne (robustesse face aux NaN / types inattendus)
    if not isinstance(text, str):
        return ""
    # 2. Normalisation Unicode (NFKC) : unifie les variantes équivalentes d'un caractère
    text = unicodedata.normalize("NFKC", text)
    # 3. Suppression des caractères de contrôle invisibles (on garde l'espace)
    text = "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] != "C")
    # 4. Uniformisation des espaces (tabs, retours ligne, espaces multiples -> 1 espace)
    text = re.sub(r"\s+", " ", text)
    # 5. Suppression des espaces en début / fin
    return text.strip()


def encode_for_rnn(text: str, vocab: dict, max_len: int, pad: int = 0, oov: int = 1) -> torch.Tensor:
    """Texte -> tenseur d'entiers (1, max_len), réplique de `encoder_sequences`.

    Le vocabulaire est appris sur le train ; les mots inconnus deviennent OOV,
    et la séquence est tronquée / complétée par du padding à `max_len`.
    """
    ids = [vocab.get(m, oov) for m in re.findall(r"\w+", str(text).lower())][:max_len]
    ids += [pad] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)
