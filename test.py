from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le modèle BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Pipeline NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Texte à analyser
texte = "Le patient prend AMOXICILLINE 500 mg et CLAMOXYL. Il doit aussi prendre LACTIBIANE."

# Exécuter le pipeline
results = ner_pipeline(texte)

# Post-traitement pour combiner les sous-mots
def combine_subwords(results):
    entities = []
    temp_entity = ""
    temp_start = None
    temp_end = None
    for res in results:
        if res['word'].startswith("##"):  # Sous-mot
            temp_entity += res['word'][2:]  # Retirer le préfixe "##"
            temp_end = res['end']
        else:
            if temp_entity:  # Sauvegarder l'entité précédente si elle existe
                entities.append({"text": temp_entity, "start": temp_start, "end": temp_end})
            temp_entity = res['word']  # Nouvelle entité
            temp_start = res['start']
            temp_end = res['end']
    if temp_entity:  # Ajouter la dernière entité
        entities.append({"text": temp_entity, "start": temp_start, "end": temp_end})
    return entities

# Appliquer le post-traitement
final_entities = combine_subwords(results)

# Filtrer les entités pertinentes (produits)
medicaments = [ent["text"] for ent in final_entities if 'PRODUCT' in ent.get('entity', '')]
print("Médicaments détectés :", medicaments)
