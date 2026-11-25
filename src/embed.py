#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/embed.py — Étape 2 du workshop RAG

Ce script :
1. Charge les chunks depuis data/chunks.json (créé par ingest.py)
2. Transforme chaque chunk en vecteur (embedding) qui représente son sens
3. Stocke les vecteurs dans un index local (dossier db/)
4. Persiste l'index pour pouvoir le réutiliser plus tard

Utilisation :
    python src/embed.py
"""

import json
from pathlib import Path

# Dossiers et fichiers
DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"
DB_DIR = Path("db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks():
    """Charge les chunks depuis le fichier JSON."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"❌ Le fichier {CHUNKS_FILE} est introuvable.\n"
            "   Lance d'abord : python src/ingest.py"
        )
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("🔢 Étape 2 : Création des embeddings...")

    # 1. Charger les chunks
    chunks = load_chunks()
    print(f"   {len(chunks)} chunks chargés depuis {CHUNKS_FILE}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "❌ La bibliothèque 'sentence-transformers' est nécessaire.\n"
            "   Installe-la avec : pip install sentence-transformers"
        )

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    DB_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "model": EMBEDDING_MODEL,
        "chunks": [
            {
                **chunk,
                "embedding": embedding.tolist(),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ],
    }
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"\n✅ {len(chunks)} embeddings créés et stockés.")
    print(f"💾 Index persisté dans : {EMBEDDINGS_FILE.resolve()}")
    print("\n👉 Tu peux maintenant lancer : python src/rag.py")


if __name__ == "__main__":
    main()