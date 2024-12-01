import spacy
import re
import json
import requests

# Charger le modèle de langue française
nlp = spacy.load("fr_core_news_md")

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = re.sub(r'[^\w\s\d\.\,mg]', ' ', text)  # Conserver chiffres, lettres, ".", ",", "mg"
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces multiples
    return text

# API OCR.space configuration
def extract_text_from_image(image_path, api_key):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'language': 'fre',  # Spécifie la langue française
        'isOverlayRequired': True,
        'OCREngine': 2,  # Utilise OCR Engine 2 pour une meilleure qualité
    }
    with open(image_path, 'rb') as f:
        response = requests.post(url, files={'file': f}, data=payload)
    return response.json()

# Configuration
image_path = 'dataset/image.png'  # Chemin vers l'image
api_key = 'K84545075888957'  # Clé API pour OCR.space

# Extraction du texte à partir de l'image
ocr_result = extract_text_from_image(image_path, api_key)
if 'ParsedResults' in ocr_result:
    extracted_text = ocr_result['ParsedResults'][0]['ParsedText']
    print("Texte extrait :")
    print(extracted_text)
else:
    print("Erreur dans la réponse de l'API :", ocr_result)
    extracted_text = ""

# Si du texte a été extrait, continuer l'analyse
if extracted_text:
    # Prétraitement du texte
    texte_nettoye = preprocess_text(extracted_text)
    print("\nTexte nettoyé :")
    print(texte_nettoye)

    # Analyse du texte avec SpaCy
    doc = nlp("Marseille, le 4 février 2020 Madame Dupont Germaine 59 ans, 64 kg amoxilline 500 mg, CLAMOXYL 500 mg , 2 gélules matin et soir. LOSEC 20 mg, 1 gélule le matin pendant 1 mois. QSP 8 jours AR 3 fois, NS LACTIBIANE, 1 sachet dans un verre d eau le matin pendant 10 jours.")

    # Extraction des médicaments détectés
    medicaments_detectes = []
    print("\nMédicaments détectés :")
    for ent in doc.ents:
        # Filtrer les entités de type pertinents
        if ent.label_ in ("MISC", "ORG", "PRODUCT", "LOC"):  # Ajouter d'autres types si nécessaire
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
