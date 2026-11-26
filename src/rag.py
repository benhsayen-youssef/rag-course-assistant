#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/rag.py — Étape 3 du workshop RAG

Pipeline RAG complet :
1. Pose une question (saisie utilisateur)
2. Recherche les passages les plus proches dans ChromaDB
3. Construit un prompt avec les passages + consigne de citation des sources
4. Appelle un LLM gratuit via OpenRouter
5. Affiche la réponse avec les références (cours + page)

Utilisation :
    python src/rag.py

Configuration (fichier .env à la racine du projet) :
    OPENROUTER_API_KEY=sk-or-v1-...
    # Optionnel : forcer un modèle précis
    # OPENROUTER_MODEL=nom/du-modele:free

Si OPENROUTER_MODEL n'est pas défini, le script récupère lui-même la liste
des modèles gratuits actuellement disponibles sur OpenRouter (les noms de
modèles gratuits changent souvent, c'était la cause de l'erreur 404).
"""

import os
from pathlib import Path

# Racine du projet (le script marche même lancé depuis un autre dossier)
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_DIR = ROOT_DIR / "db"
COLLECTION_NAME = "cours"

# Charger le fichier .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# .strip() enlève espaces et guillemets accidentels autour de la clé
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()

API_BASE = "https://openrouter.ai/api/v1"

# Si la distance du meilleur passage dépasse ce seuil, on considère que la
# question est hors sujet et on n'appelle pas le LLM. None = désactivé.
# Astuce : affiche les distances (voir main) puis règle ce seuil.
MAX_DISTANCE = None


def search_chroma(question, top_k=3):
    """Recherche les `top_k` passages les plus proches dans ChromaDB."""
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "La bibliothèque 'chromadb' est nécessaire.\n"
            "Installe-la avec : pip install chromadb"
        )

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier {DB_DIR} est introuvable.\n"
            "Lance d'abord : python src/embed.py"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # get_collection (et non get_or_create) : erreur claire si la collection n'existe pas
    collection = client.get_collection(name=COLLECTION_NAME)

    return collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def build_prompt(question, context_chunks):
    """Construit le prompt : répondre UNIQUEMENT avec les extraits + citer les sources."""
    context_parts = []
    for chunk, meta in zip(context_chunks["documents"][0], context_chunks["metadatas"][0]):
        source = meta.get("source", "source inconnue")
        page = meta.get("page", "?")
        context_parts.append(f"- [{source}, page {page}] {chunk}")

    context = "\n".join(context_parts)

    return f"""Tu es un assistant pédagogique. Réponds à la question de l'étudiant
UNIQUEMENT à partir des extraits de cours ci-dessous.

Règles importantes :
- Cite TOUJOURS le cours et la page à la fin de chaque affirmation, au format [fichier.pdf, page N].
- Si les extraits ne contiennent pas assez d'information pour répondre, réponds exactement : "L'information ne se trouve pas dans les documents fournis."
- N'invente rien. Ne sors pas du contexte des extraits.

EXTRAITS DE COURS :
{context}

QUESTION DE L'ÉTUDIANT :
{question}

RÉPONSE (en français, avec sources) :
"""


def get_free_models(requests, limit=6):
    """Récupère la liste des modèles gratuits actuellement disponibles sur OpenRouter."""
    try:
        r = requests.get(f"{API_BASE}/models", timeout=15)
        r.raise_for_status()
        models = r.json().get("data", [])
    except Exception:
        return []

    free = []
    for m in models:
        pricing = m.get("pricing", {}) or {}
        try:
            is_free = float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
        except (TypeError, ValueError):
            is_free = False
        if is_free and m.get("id"):
            free.append((m.get("context_length") or 0, m["id"]))

    # Les modèles avec le plus grand contexte d'abord
    free.sort(reverse=True)
    return [model_id for _, model_id in free[:limit]]


def call_llm(prompt):
    """
    Appelle un LLM gratuit via OpenRouter.
    Essaie plusieurs modèles si l'un est indisponible (404) ou saturé (429).
    """
    if not OPENROUTER_API_KEY:
        return (
            "[Clé API LLM non configurée.]\n"
            "Crée un fichier .env à la racine avec OPENROUTER_API_KEY=ta_cle\n"
            "En attendant, les passages pertinents sont affichés ci-dessus."
        )

    try:
        import requests
    except ImportError:
        return "[Bibliothèque 'requests' manquante. Installe avec : pip install requests]"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Liste des modèles à essayer
    if OPENROUTER_MODEL:
        candidates = [OPENROUTER_MODEL]
    else:
        candidates = ["openrouter/free"] + get_free_models(requests)

    errors = []
    for model in candidates:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=data,
                timeout=60,
            )
        except Exception as e:
            errors.append(f"{model} : {e}")
            continue

        if response.status_code == 200:
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError):
                errors.append(f"{model} : réponse vide ou inattendue")
                continue
            if content and content.strip():
                return f"{content.strip()}\n\n(modèle utilisé : {model})"
            errors.append(f"{model} : réponse vide")
            continue

        if response.status_code in (401, 403):
            return (
                f"[Erreur {response.status_code} : clé API refusée par OpenRouter.]\n"
                "Vérifie la ligne OPENROUTER_API_KEY=... dans .env (sans espaces ni guillemets),\n"
                "ou crée une nouvelle clé sur https://openrouter.ai/keys"
            )

        # 404 (modèle disparu), 429 (trop de requêtes), 5xx... : on essaie le suivant
        errors.append(f"{model} : HTTP {response.status_code} - {response.text[:150]}")

    return "[Aucun modèle n'a répondu. Détails :]\n  - " + "\n  - ".join(errors)


def ask(question):
    """Traite une question : recherche, affichage des passages, appel du LLM."""
    print("\n🔍 Recherche des passages pertinents...")
    try:
        results = search_chroma(question, top_k=3)
    except Exception as e:
        print(f"❌ Erreur lors de la recherche : {e}")
        return

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        print("❌ Aucun passage pertinent trouvé dans les cours.")
        return

    print(f"\n✅ {len(chunks)} passage(s) trouvé(s) :")
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), 1):
        source = meta.get("source", "?")
        page = meta.get("page", "?")
        print(f"\n  [{i}] {source} — page {page}  (distance : {dist:.3f})")
        print(f"      {chunk[:200]}...")

    # Filtre hors-sujet (optionnel) : évite d'envoyer des passages sans rapport au LLM
    if MAX_DISTANCE is not None and distances[0] > MAX_DISTANCE:
        print("\n" + "=" * 50)
        print("💬 L'information ne se trouve pas dans les documents fournis.")
        print("=" * 50)
        return

    print("\n" + "=" * 50)
    print("🧠 Génération de la réponse...")
    reponse = call_llm(build_prompt(question, results))

    print("\n" + "=" * 50)
    print("💬 RÉPONSE DE L'ASSISTANT :")
    print("=" * 50)
    print(reponse)
    print("=" * 50)


def main():
    print("🤖 Étape 3 : Pipeline RAG — Pose ta question !")
    print("   (tape 'q' ou laisse vide pour quitter)")
    print("=" * 50)

    if not DB_DIR.exists():
        print(f"❌ Le dossier {DB_DIR} est introuvable.")
        print("   Lance d'abord : python src/embed.py")
        return

    # Boucle : on peut poser plusieurs questions sans relancer le script
    while True:
        question = input("\n❓ Ta question : ").strip()
        if not question or question.lower() in ("q", "quit", "exit"):
            print("👋 À bientôt !")
            break
        ask(question)


if __name__ == "__main__":
    main()