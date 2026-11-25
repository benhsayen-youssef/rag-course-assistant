#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/rag.py — Étape 3 du workshop RAG

Ce script implémente le pipeline RAG complet :
1. Pose une question (saisie utilisateur)
2. Transforme la question en vecteur et recherche les passages les plus proches
   dans l'index local
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
DB_DIR = Path("db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.json"

# Charger les variables d'environnement si python-dotenv est disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def search_embeddings(question, top_k=3):
    """
    Recherche les `top_k` passages les plus proches de la question.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "❌ La bibliothèque 'sentence-transformers' est nécessaire.\n"
            "   Installe-la avec : pip install sentence-transformers"
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"❌ L'index {EMBEDDINGS_FILE} est introuvable.\n"
            "   Lance d'abord : python src/embed.py"
        )

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    model = SentenceTransformer(index["model"])
    query_embedding = model.encode(question, normalize_embeddings=True)
    ranked_chunks = sorted(
        index["chunks"],
        key=lambda chunk: sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_embedding, chunk["embedding"])
        ),
        reverse=True,
    )[:top_k]

    return {
        "documents": [[chunk["text"] for chunk in ranked_chunks]],
        "metadatas": [[
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in ranked_chunks
        ]],
    }


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

    # Vérifier que l'index existe
    if not EMBEDDINGS_FILE.exists():
        print(f"❌ L'index {EMBEDDINGS_FILE} est introuvable.")
        print("   Lance d'abord : python src/embed.py")
        return

    # Saisie de la question
    question = input("\n❓ Ta question : ").strip()
    if not question:
        print("❌ Question vide. Relance le script.")
        return

    # Recherche dans l'index local
    print("\n🔍 Recherche des passages pertinents...")
    try:
        results = search_embeddings(question, top_k=3)
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