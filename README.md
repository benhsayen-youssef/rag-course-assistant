# 🎓 Mon Assistant de Cours RAG — Template de Workshop

Dépôt template pour le workshop **"Mon premier assistant de cours avec RAG"**.

## 📁 Structure du projet

```
rag-course-assistant/
├── .gitignore
├── README.md
├── requirements.txt
├── data/                 # ⚠️ Mets ici tes PDFs de cours (non committé si volumineux)
├── src/
│   ├── ingest.py         # Étape 1 : extrait le texte des PDF et découpe en chunks
│   ├── embed.py          # Étape 2 : crée les embeddings + base vectorielle ChromaDB
│   └── rag.py            # Étape 3 : pipeline RAG (question → réponse avec sources)
├── interface/
│   ├── index.html        # Page d'accueil (frontpage HTML/CSS)
│   └── style.css         # Mise en forme de la page
└── db/                   # Créé automatiquement par ChromaDB (ne pas committer)
```

## 🚀 Démarrage rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/bwanox/rag-course-assistant.git
cd rag-course-assistant
```

### 2. Créer et activer l'environnement Python
```bash
# Sur macOS / Linux :
python -m venv venv
source venv/bin/activate

# Sur Windows (PowerShell) :
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter tes documents de cours
Dépose 3 à 5 fichiers PDF dans le dossier `data/`.

### 5. Lancer le pipeline
```bash
# Étape 1 : ingestion des PDF
python src/ingest.py

# Étape 2 : embeddings + base vectorielle
python src/embed.py

# Étape 3 : poser une question à ton assistant
python src/rag.py
```

### 6. Ouvrir la page web
Ouvre `interface/index.html` directement dans ton navigateur (double-clic).

## 📋 Livrables attendus par équipe

- [ ] Dépôt GitHub avec historique de commits
- [ ] `README.md` expliquant comment lancer le projet
- [ ] Dossier `data/` avec au moins 3 documents de cours
- [ ] Scripts `src/ingest.py`, `src/embed.py`, `src/rag.py` fonctionnels
- [ ] Base vectorielle locale générée (dossier `db/`)
- [ ] Page `interface/index.html` stylisée avec `interface/style.css`
- [ ] Au moins une Pull Request ouverte, reviewée et mergeée

## 🔒 Sécurité

- Ne jamais committer de clés API, mots de passe ou données personnelles.
- Ne pas mettre de documents contenant des noms réels, numéros de téléphone ou adresses e-mail dans `data/`.

---

*Template pour le workshop LLM & RAG — Étudiants débutants bienvenus !*