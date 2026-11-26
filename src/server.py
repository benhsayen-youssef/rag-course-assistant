#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/server.py — Étape 4 du workshop RAG : le site web

Ce script ne remplace pas rag.py : il le réutilise.
Il fait le lien entre la page web (index.html) et le pipeline RAG :
    page web  --(question)-->  server.py  -->  rag.py (ChromaDB + LLM)
    page web  <--(réponse)---  server.py  <--

Utilisation :
    pip install flask
    python src/server.py
puis ouvre http://localhost:8000 dans ton navigateur.
"""

from flask import Flask, jsonify, request, send_from_directory

import rag  # le fichier src/rag.py (même dossier que ce script)

# Dossier qui contient index.html, style.css et script.js
WEB_DIR = rag.ROOT_DIR / "interface"
PORT = 8000
TOP_K = 3  # nombre de passages envoyés au LLM

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.get("/")
def accueil():
    """Affiche la page web."""
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/ask")
def ask():
    """Reçoit { "question": "..." } et renvoie { "reponse": "...", "sources": [...] }."""
    donnees = request.get_json(silent=True) or {}
    question = (donnees.get("question") or "").strip()
    if not question:
        return jsonify({"erreur": "La question est vide."}), 400

    # 1. Recherche des passages les plus proches dans ChromaDB
    try:
        resultats = rag.search_chroma(question, top_k=TOP_K)
    except Exception as e:
        return jsonify({"erreur": f"Erreur lors de la recherche : {e}"}), 500

    documents = resultats["documents"][0]
    metadatas = resultats["metadatas"][0]
    distances = resultats["distances"][0]

    if not documents:
        return jsonify({
            "reponse": "Aucun passage pertinent trouvé dans les cours.",
            "sources": [],
        })

    # 2. Filtre hors-sujet (actif seulement si MAX_DISTANCE est défini dans rag.py)
    if rag.MAX_DISTANCE is not None and distances[0] > rag.MAX_DISTANCE:
        return jsonify({
            "reponse": "L'information ne se trouve pas dans les documents fournis.",
            "sources": [],
        })

    # 3. Appel du LLM avec les passages trouvés
    reponse = rag.call_llm(rag.build_prompt(question, resultats))

    # 4. Liste des sources (sans doublons, dans l'ordre)
    sources = []
    for meta in metadatas:
        ligne = f"{meta.get('source', '?')}, page {meta.get('page', '?')}"
        if ligne not in sources:
            sources.append(ligne)

    return jsonify({"reponse": reponse, "sources": sources})


if __name__ == "__main__":
    print(f"🌐 Assistant de cours : ouvre http://localhost:{PORT}")
    print("   (Ctrl+C pour arrêter le serveur)")
    app.run(host="127.0.0.1", port=PORT)
