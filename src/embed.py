#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/embed.py — Étape 2 du workshop RAG

Ce script :
1. Charge les chunks depuis data/chunks.json (créé par ingest.py)
2. Transforme chaque chunk en vecteur (embedding) qui représente son sens
3. Stocke les vecteurs dans une base vectorielle locale : ChromaDB (dossier db/)
4. Persiste la base pour pouvoir la réutiliser plus tard

Utilisation :
    python src/embed.py
"""

import json
import os
from pathlib import Path

# Dossiers et fichiers
DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"
DB_DIR = Path("db")
COLLECTION_NAME = "cours"


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
    print("🔢 Étape 2 : Création des embeddings + base vectorielle ChromaDB...")

    # 1. Charger les chunks
    chunks = load_chunks()
    print(f"   {len(chunks)} chunks chargés depuis {CHUNKS_FILE}")

    # 2. Initialiser ChromaDB (en local, pas de serveur nécessaire)
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "❌ La bibliothèque 'chromadb' est nécessaire.\n"
            "   Installe-la avec : pip install chromadb"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Créer ou récupérer la collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"   Collection ChromaDB '{COLLECTION_NAME}' prête (dossier {DB_DIR}/)")

    # 3. Créer les embeddings et ajouter à ChromaDB
    # ChromaDB peut calculer les embeddings automatiquement avec un modèle par défaut.
    # On ajoute les documents un par un (ou par petit lot) avec leurs métadonnées.
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\n✅ {len(documents)} embeddings créés et stockés dans ChromaDB.")
    print(f"💾 Base vectorielle persistée dans : {DB_DIR.resolve()}")
    print("\n👉 Tu peux maintenant lancer : python src/rag.py")


if __name__ == "__main__":
    main()