#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/rag.py — Étape 3 du workshop RAG (version légère, sans ChromaDB)

Ce script implémente le pipeline RAG complet :
1. Pose une question (saisie utilisateur)
2. Vectorise la question avec le même vocabulaire que embed.py
3. Calcule la similarité cosinus avec tous les chunks stockés
4. Récupère les top-k passages les plus pertinents
5. Construit un prompt avec consigne de citation des sources
6. Appelle un LLM (OpenRouter, gratuit) si une clé API est configurée
   Sinon : affiche les passages trouvés (mode dégradé)
7. Affiche la réponse avec les références (cours + page)

Utilisation :
    python src/rag.py

Configuration optionnelle :
    Crée un fichier .env à la racine avec :
        OPENROUTER_API_KEY=ta_cle_gratuite
"""

import json
import math
import os
import re
from pathlib import Path

DATA_DIR = Path("data")
VECTORS_FILE = DATA_DIR / "vectors.json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def tokenize(text):
    text = text.lower()
    return re.findall(r"\w+", text)


def chunk_to_vector(text, vocab):
    words = tokenize(text)
    total = len(words)
    if total == 0:
        return [0.0] * len(vocab)
    vector = []
    for word in vocab:
        count = words.count(word)
        vector.append(count / total)
    return vector


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_vectors():
    if not VECTORS_FILE.exists():
        raise FileNotFoundError(
            f"❌ Le fichier {VECTORS_FILE} est introuvable.\n"
            "   Lance d'abord : python src/embed.py"
        )
    with open(VECTORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search(question, data, top_k=3):
    vocab = data["vocab"]
    records = data["records"]

    question_vector = chunk_to_vector(question, vocab)

    scored = []
    for rec in records:
        sim = cosine_similarity(question_vector, rec["vector"])
        scored.append((sim, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def build_prompt(question, results):
    context_parts = []
    for sim, rec in results:
        source = rec.get("source", "source inconnue")
        page = rec.get("page", "?")
        context_parts.append(f"- [{source}, page {page}] (pertinence : {sim:.2f}) {rec['text']}")

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

    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
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

    try:
        data = load_vectors()
    except FileNotFoundError as e:
        print(e)
        return

    question = input("\n❓ Ta question : ").strip()
    if not question:
        print("❌ Question vide. Relance le script.")
        return

    print("\n🔍 Recherche des passages pertinents...")
    results = search(question, data, top_k=3)

    if not results:
        print("❌ Aucun passage pertinent trouvé dans les cours.")
        return

    print(f"\n✅ {len(results)} passage(s) trouvé(s) :")
    for i, (sim, rec) in enumerate(results, 1):
        source = rec.get("source", "?")
        page = rec.get("page", "?")
        print(f"\n  [{i}] {source} — page {page} (pertinence : {sim:.2f})")
        print(f"      {rec['text'][:200]}...")

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