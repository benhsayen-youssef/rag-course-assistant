#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/embed.py — Étape 2 du workshop RAG (version légère, sans ChromaDB)

Ce script :
1. Lit les chunks depuis data/chunks.json (créé par ingest.py)
2. Construit un vocabulaire simple à partir de tous les chunks
3. Transforme chaque chunk en vecteur (bag-of-words normalisé)
4. Sauvegarde tout dans data/vectors.json

Aucune base de données externe n'est nécessaire. Tout est stocké dans un fichier JSON.

Utilisation :
    python src/embed.py
"""

import json
import math
import re
from pathlib import Path

DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"
VECTORS_FILE = DATA_DIR / "vectors.json"


def tokenize(text):
    """Découpe un texte en mots (minuscules, ponctuation retirée)."""
    text = text.lower()
    words = re.findall(r"\w+", text)
    return words


def build_vocabulary(chunks):
    """Construit la liste des mots uniques présents dans tous les chunks."""
    vocab = set()
    for chunk in chunks:
        vocab.update(tokenize(chunk["text"]))
    return sorted(vocab)


def chunk_to_vector(text, vocab):
    """
    Transforme un texte en vecteur de fréquences de mots.
    Retourne une liste de nombres de même longueur que le vocabulaire.
    """
    words = tokenize(text)
    total = len(words)
    if total == 0:
        return [0.0] * len(vocab)

    vector = []
    for word in vocab:
        count = words.count(word)
        vector.append(count / total)  # fréquence normalisée
    return vector


def cosine_similarity(v1, v2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def main():
    print("🔢 Étape 2 : Création des vecteurs (version légère, sans ChromaDB)...")

    if not CHUNKS_FILE.exists():
        print(f"❌ Le fichier {CHUNKS_FILE} est introuvable.")
        print("   Lance d'abord : python src/ingest.py")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"   {len(chunks)} chunks chargés.")

    # 1. Construire le vocabulaire
    vocab = build_vocabulary(chunks)
    print(f"   Vocabulaire : {len(vocab)} mots uniques.")

    # 2. Vectoriser chaque chunk
    records = []
    for i, chunk in enumerate(chunks):
        vector = chunk_to_vector(chunk["text"], vocab)
        records.append({
            "id": f"chunk_{i}",
            "vector": vector,
            "text": chunk["text"],
            "source": chunk.get("source", "inconnu"),
            "page": chunk.get("page", "?"),
        })

    # 3. Sauvegarder
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "vocab": vocab,
            "records": records,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(records)} vecteurs créés et sauvegardés dans : {VECTORS_FILE.resolve()}")
    print("👉 Tu peux maintenant lancer : python src/rag.py")


if __name__ == "__main__":
    main()