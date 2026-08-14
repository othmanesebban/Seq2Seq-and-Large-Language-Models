import json
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from nltk.translate.meteor_score import meteor_score

class MockCOCO:
    def __init__(self, refs):
        """
        Initialize the mock COCO object with references.
        """
        self.refs = refs
        self.imgIds = list(refs.keys())  # List of image/video IDs
        self.imgToAnns = self._create_imgToAnns(refs)  # Create imgToAnns structure

    def _create_imgToAnns(self, refs):
        """
        Create the imgToAnns attribute from the references.
        """
        imgToAnns = {}
        for img_id, annotations in refs.items():
            imgToAnns[img_id] = [{"image_id": img_id, "caption": ann["caption"]} for ann in annotations]
        return imgToAnns

    def getImgIds(self):
        """
        Return the list of image IDs.
        """
        return self.imgIds

    def loadRes(self, preds):
        """
        Mock method to load results as annotations.
        """
        imgToAnns = {}
        for img_id, captions in preds.items():
            imgToAnns[img_id] = [{"image_id": img_id, "caption": caption["caption"]} for caption in captions]
        return MockCOCO(imgToAnns)

# Charger les fichiers JSON
with open("dataset/vatex_test_cider.json", "r") as ref_file:
    references = json.load(ref_file)

with open("dataset/videoID_captions.json", "r") as pred_file:
    predictions = json.load(pred_file)

# Adapter les données
def prepare_coco_format(predictions, references):
    """
    Prépare les données pour qu'elles respectent le format COCO.
    """
    # Normaliser les références
    references_dict = {}
    if isinstance(references, list):  # Si références est une liste
        for item in references:
            image_id = item["image_id"]
            if image_id not in references_dict:
                references_dict[image_id] = []
            references_dict[image_id].append({"caption": item["caption"]})  # Ajouter comme dictionnaire

    # Normaliser les prédictions
    predictions_dict = {}
    if isinstance(predictions, dict):  # Si prédictions est un dict
        for video_id, captions in predictions.items():
            if isinstance(captions, list) and captions:  # Vérifiez que captions est une liste non vide
                predictions_dict[video_id] = [{"caption": captions[0]}]  # Utilisez la première prédiction pour chaque clé

    return predictions_dict, references_dict

# Synchroniser les clés entre les références et les prédictions
def synchronize_keys(refs, preds):
    """
    Synchronise les clés entre les références et les prédictions.
    Supprime les IDs qui ne correspondent pas.
    """
    common_keys = set(refs.keys()).intersection(set(preds.keys()))
    synced_refs = {k: refs[k] for k in common_keys}
    synced_preds = {k: preds[k] for k in common_keys}
    return synced_refs, synced_preds

# Reformater les données après tokenisation
def reformat_tokenized_output(tokenized_output):
    """
    Reformate la sortie de PTBTokenizer pour qu'elle respecte la structure attendue.
    """
    reformatted = {}
    for img_id, captions in tokenized_output.items():
        # Transformez chaque chaîne dans `captions` en un dictionnaire avec la clé "caption"
        reformatted[img_id] = [{"caption": caption} for caption in captions]
    return reformatted

# Reformater les données pour qu'elles soient compatibles avec le module BLEU
def format_for_bleu(input_data):
    """
    Reformate les données pour qu'elles soient une liste de chaînes.
    """
    formatted_data = {}
    for img_id, annotations in input_data.items():
        formatted_data[img_id] = [ann["caption"] for ann in annotations]
    return formatted_data

# Calculer METEOR
def compute_meteor(refs, preds):
    """
    Calcule le score METEOR pour les prédictions et les références.
    """
    meteor_scores = []
    for img_id, ref_annotations in refs.items():
        # Tokeniser les références et les hypothèses
        reference_list = [ref["caption"].split() for ref in ref_annotations]
        hypothesis = preds[img_id][0]["caption"].split()  # Utilisez la première hypothèse

        # Calculer le score METEOR
        score = meteor_score(reference_list, hypothesis)
        meteor_scores.append(score)

    # Retourner la moyenne des scores METEOR
    return sum(meteor_scores) / len(meteor_scores)

# Préparer les données
preds_coco, refs_coco = prepare_coco_format(predictions, references)

# Tokenizer
tokenizer = PTBTokenizer()
tokenized_refs = tokenizer.tokenize(refs_coco)
tokenized_preds = tokenizer.tokenize(preds_coco)

# Reformater les données après tokenisation
tokenized_refs = reformat_tokenized_output(tokenized_refs)
tokenized_preds = reformat_tokenized_output(tokenized_preds)

# Synchroniser les données
tokenized_refs, tokenized_preds = synchronize_keys(tokenized_refs, tokenized_preds)

# Créer des objets COCO factices
mock_coco = MockCOCO(tokenized_refs)
mock_res = mock_coco.loadRes(tokenized_preds)

# Reformater les données pour BLEU
mock_coco.imgToAnns = format_for_bleu(mock_coco.imgToAnns)
mock_res.imgToAnns = format_for_bleu(mock_res.imgToAnns)

# Évaluation manuelle des métriques
metrics = {
    "Bleu": Bleu(4),  # Évaluation pour BLEU-1 à BLEU-4
    #"ROUGE_L": Rouge(),
    "CIDEr": Cider(),
    # "SPICE": Spice(),  # Activez si nécessaire
}
results = []

# Calculer chaque métrique
for metric_name, scorer in metrics.items():
    print(f"Computing {metric_name} score...")
    score, _ = scorer.compute_score(mock_coco.imgToAnns, mock_res.imgToAnns)
    if metric_name == "Bleu":
        # BLEU produit une liste de scores
        results.append(f"{metric_name}: {', '.join([f'{s:.4f}' for s in score])}")
    else:
        # Autres métriques produisent un score unique
        results.append(f"{metric_name}: {score:.4f}")

# Calculer METEOR
print("Computing METEOR score...")
meteor_avg = compute_meteor(tokenized_refs, tokenized_preds)
results.append(f"METEOR: {meteor_avg:.4f}")

# Calculer ROUGE-L
print("Computing ROUGE_L score...")
rouge_score, _ = Rouge().compute_score(mock_coco.imgToAnns, mock_res.imgToAnns)
results.append(f"ROUGE_L: {rouge_score:.4f}")

# Écrire les résultats dans un fichier texte
with open("results.txt", "w") as result_file:
    result_file.write("\n".join(results))

# Afficher les résultats
print("\nÉvaluation terminée. Résultats écrits dans 'results.txt'.")
for line in results:
    print(line)
