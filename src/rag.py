#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/rag.py — Étape 3 du workshop RAG

Ce script implémente le pipeline RAG complet :
1. Pose une question (saisie utilisateur)
2. Transforme la question en vecteur et recherche les passages les plus proches
   dans la base vectorielle ChromaDB
3. Construit un prompt avec les passages trouvés + consigne de citation des sources
4. Appelle un LLM (via une API gratuite, par exemple OpenRouter)
5. Affiche la réponse avec les références (cours + page)

Utilisation :
    python src/rag.py

Configuration :
    Crée un fichier .env à la racine avec :
        OPENROUTER_API_KEY=ton_cle_api_gratuite
    (Optionnel : si aucune clé n'est fournie, le script affiche les passages
     trouvés sans appeler le LLM, pour montrer le principe.)
"""

import json
import os
from pathlib import Path

# Dossiers et fichiers
DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"
DB_DIR = Path("db")
COLLECTION_NAME = "cours"

# Charger les variables d'environnement si python-dotenv est disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def search_chroma(question, top_k=3):
    """
    Recherche les `top_k` passages les plus proches de la question
    dans la base vectorielle ChromaDB.
    """
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "❌ La bibliothèque 'chromadb' est nécessaire.\n"
            "   Installe-la avec : pip install chromadb"
        )

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"❌ Le dossier {DB_DIR} est introuvable.\n"
            "   Lance d'abord : python src/embed.py"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def build_prompt(question, context_chunks):
    """
    Construit le prompt envoyé au LLM.
    La consigne demande de répondre UNIQUEMENT à partir des extraits
    et de citer les sources.
    """
    context_parts = []
    for chunk, meta in zip(context_chunks["documents"][0], context_chunks["metadatas"][0]):
        source = meta.get("source", "source inconnue")
        page = meta.get("page", "?")
        context_parts.append(f"- [{source}, page {page}] {chunk}")

    context = "\n".join(context_parts)

    prompt = f"""Tu es un assistant pédagogique. Réponds à la question de l'étudiant
UNIQUEMENT à partir des extraits de cours ci-dessous.

Règles importantes :
- Cite TOUJOURS le cours et la page (ou la minute pour un audio) à la fin de chaque affirmation.
- Si les extraits ne contiennent pas assez d'information pour répondre, dis-le clairement : "L'information ne se trouve pas dans les documents fournis."
- N'invente rien. Ne sors pas du contexte des extraits.

EXTRAITS DE COURS :
{context}

QUESTION DE L'ÉTUDIANT :
{question}

RÉPONSE (en français, avec sources) :
"""
    return prompt


def call_llm(prompt):
    """
    Appelle un LLM via l'API OpenRouter (modèle gratuit).
    Si aucune clé API n'est configurée, renvoie un message indiquant
    que le LLM n'a pas pu être appelé.
    """
    if not OPENROUTER_API_KEY:
        return (
            "[Clé API LLM non configurée.]\n"
            "Pour activer la génération automatique, crée un fichier .env à la racine\n"
            "avec OPENROUTER_API_KEY=ta_cle_gratuite (ex: OpenRouter).\n"
            "\nEn attendant, voici les passages pertinents trouvés :\n"
        )

    try:
        import requests
    except ImportError:
        return "[Bibliothèque 'requests' manquante. Installe avec : pip install requests]"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Erreur lors de l'appel au LLM : {e}]"


def main():
    print("🤖 Étape 3 : Pipeline RAG — Pose ta question !")
    print("=" * 50)

    # Vérifier que la base vectorielle existe
    if not DB_DIR.exists():
        print(f"❌ Le dossier {DB_DIR} est introuvable.")
        print("   Lance d'abord : python src/embed.py")
        return

    # Saisie de la question
    question = input("\n❓ Ta question : ").strip()
    if not question:
        print("❌ Question vide. Relance le script.")
        return

    # Recherche dans ChromaDB
    print("\n🔍 Recherche des passages pertinents...")
    try:
        results = search_chroma(question, top_k=3)
    except Exception as e:
        print(f"❌ Erreur lors de la recherche : {e}")
        return

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        print("❌ Aucun passage pertinent trouvé dans les cours.")
        return

    print(f"\n✅ {len(chunks)} passage(s) trouvé(s) :")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        source = meta.get("source", "?")
        page = meta.get("page", "?")
        print(f"\n  [{i}] {source} — page {page}")
        print(f"      {chunk[:200]}...")

    # Construire le prompt et appeler le LLM
    print("\n" + "=" * 50)
    print("🧠 Génération de la réponse...")
    prompt = build_prompt(question, results)
    reponse = call_llm(prompt)

    print("\n" + "=" * 50)
    print("💬 RÉPONSE DE L'ASSISTANT :")
    print("=" * 50)
    print(reponse)
    print("=" * 50)


if __name__ == "__main__":
    main()