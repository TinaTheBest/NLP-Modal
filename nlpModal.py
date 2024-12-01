import spacy
import re
import json  


nlp = spacy.load("fr_core_news_md")


def preprocess_text(text):
    text = re.sub(r'[^\w\s\d\.\,mg]', ' ', text)  # Conserver chiffres, lettres, ".", ",", "mg"
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces multiples
    return text


texte = """
Le patient a reçu une ordonnance pour du Doliprane et de l'. 
Madame Dupont Germaine
59 ans, 64 kg
- Amoxicilline 500 mg, CLAMOXYL 500 mg , 2 gélules matin et soir.
LOSEC 20 mg, 1 gélule le matin pendant 1 mois.
QSP 8 jours
AR 3 fois, NS
- LACTIBIANE, 1 sachet dans un verre d'eau le matin pendant 10 jours.
NR
"""

# Prétraitement du texte
texte_nettoye = preprocess_text(texte)
print("Texte nettoyé :")
print(texte_nettoye)

# Analyse du texte avec SpaCy
doc = nlp(texte_nettoye)

# Extraction des médicaments détectés
medicaments_detectes = []
print("\nMédicaments détectés :")
for ent in doc.ents:
    # Filtrer les entités de type pertinents
    if ent.label_ in ("MISC", "PRODUCT"):  # Ajouter d'autres types si nécessaire
        medicaments_detectes.append(ent.text)
        print(f"- {ent.text} (Type : {ent.label_})")

# Préparation des données pour le fichier JSON
json_data = {
    "medicaments_detectes": list(set(medicaments_detectes))  # Suppression des doublons
}

# Écriture dans un fichier JSON
with open("medicaments_detectes.json", "w", encoding="utf-8") as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# Afficher les résultats pour vérification
print("\nRésultats sauvegardés dans 'medicaments_detectes.json' :")
print(json.dumps(json_data, ensure_ascii=False, indent=4))
