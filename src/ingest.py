#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/ingest.py — Étape 1 du workshop RAG

Ce script :
1. Lit tous les PDF du dossier data/
2. Extrait le texte de chaque page
3. Découpe le texte en petits passages ("chunks") d'environ 200 mots
4. Sauvegarde les chunks avec leurs métadonnées (nom du fichier, numéro de page)
   dans un fichier data/chunks.json

Utilisation :
    python src/ingest.py
"""

import json
import os
from pathlib import Path

# Dossiers
DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"

# Taille approximative d'un chunk (en mots)
CHUNK_SIZE = 200


def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF.
    Essaie d'utiliser pypdf. Si ce n'est pas installé, affiche une erreur claire.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "La bibliothèque 'pypdf' est nécessaire pour lire les PDF.\n"
            "Installe-la avec : pip install pypdf"
        )

    reader = PdfReader(str(pdf_path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append({"page": i + 1, "text": text})
    return pages_text


def split_into_chunks(text, source_name, page_number, chunk_size=CHUNK_SIZE):
    """
    Découpe un texte en chunks d'environ `chunk_size` mots.
    Chaque chunk garde sa source et son numéro de page.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 0:
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "page": page_number,
            })
    return chunks


def main():
    print("🔍 Étape 1 : Ingestion des documents PDF...")
    print(f"Dossier source : {DATA_DIR.resolve()}")

    if not DATA_DIR.exists():
        print(f"❌ Le dossier {DATA_DIR} n'existe pas. Crée-le et ajoute des PDFs.")
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ Aucun fichier PDF trouvé dans le dossier data/.")
        print("   Ajoute des PDFs de cours puis relance ce script.")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  📄 Traitement de {pdf_path.name}...")
        try:
            pages = extract_text_from_pdf(pdf_path)
            for page_info in pages:
                page_chunks = split_into_chunks(
                    page_info["text"],
                    source_name=pdf_path.name,
                    page_number=page_info["page"],
                )
                all_chunks.extend(page_chunks)
        except Exception as e:
            print(f"  ⚠️  Erreur avec {pdf_path.name} : {e}")
            print("     Si c'est un PDF scanné, tu as besoin de l'OCR (Tesseract).")

    if not all_chunks:
        print("❌ Aucun chunk n'a pu être extrait.")
        return

    # Sauvegarde
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Ingestion terminée : {len(all_chunks)} chunks créés.")
    print(f"💾 Sauvegardé dans : {CHUNKS_FILE.resolve()}")


if __name__ == "__main__":
    main()