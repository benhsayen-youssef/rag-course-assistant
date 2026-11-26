// ===== Réglage : adresse du serveur =====
// "/ask" = le serveur Python (src/server.py) qui sert aussi cette page.
const API_URL = "/ask";

// ===== On récupère les éléments de la page grâce à leur id =====
const formulaire = document.getElementById("formulaire");
const champQuestion = document.getElementById("question");
const bouton = document.getElementById("envoyer");
const texteReponse = document.getElementById("texte-reponse");
const blocSources = document.getElementById("sources");
const listeSources = document.getElementById("liste-sources");

// ===== Quand on envoie le formulaire (clic ou Entrée) =====
formulaire.addEventListener("submit", async (event) => {
    event.preventDefault(); // empêche la page de se recharger

    const question = champQuestion.value.trim();
    if (!question) return;

    // État "en attente"
    bouton.disabled = true;
    bouton.textContent = "Réflexion en cours...";
    texteReponse.classList.remove("erreur");
    texteReponse.textContent = "Recherche dans tes cours...";
    blocSources.hidden = true;

    try {
        // On envoie la question au serveur Python
        const reponse = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question }),
        });

        if (!reponse.ok) {
            // Le serveur a répondu avec une erreur : on lit son message
            let message = "Erreur du serveur (code " + reponse.status + ").";
            try {
                const erreurServeur = await reponse.json();
                if (erreurServeur.erreur) message = erreurServeur.erreur;
            } catch (e) {
                // pas de JSON : on garde le message par défaut
            }
            throw new Error(message);
        }

        const donnees = await reponse.json();

        // On affiche la réponse
        texteReponse.textContent = donnees.reponse;

        // On affiche les sources s'il y en a
        listeSources.innerHTML = "";
        const sources = donnees.sources || [];
        sources.forEach((source) => {
            const li = document.createElement("li");
            li.textContent = source; // textContent est plus sûr que innerHTML
            listeSources.appendChild(li);
        });
        blocSources.hidden = sources.length === 0;

    } catch (erreur) {
        console.error(erreur);
        texteReponse.classList.add("erreur");
        if (erreur instanceof TypeError) {
            // fetch n'a pas pu joindre le serveur du tout
            texteReponse.textContent =
                "Impossible de joindre le serveur. Vérifie que « python src/server.py » est lancé " +
                "et ouvre la page avec http://localhost:8000 (pas en double-cliquant sur index.html).";
        } else {
            texteReponse.textContent = erreur.message;
        }
    } finally {
        // Dans tous les cas, on réactive le bouton
        bouton.disabled = false;
        bouton.textContent = "Poser la question";
    }
});

