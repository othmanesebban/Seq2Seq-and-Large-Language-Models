import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoImageProcessor
from tqdm.auto import tqdm
import openai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

# Configuration de l'API OpenAI
openai.api_key = "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"  # Remplacez par votre clé API OpenAI

# Détection du périphérique (CPU ou GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le fichier JSON
def load_json_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

# Classe personnalisée pour GPT-4 interactions via l'API OpenAI
class GPT4Decoder:
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def generate_caption(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a video captioning assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

# Calcul des métriques d'évaluation
class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.meteor = Meteor()
        self.cider = Cider()

    def compute_bleu(self, reference, hypothesis):
        smoothing = SmoothingFunction().method4
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing)

    def compute_rouge(self, reference, hypothesis):
        scores = self.rouge_scorer.score(reference, hypothesis)
        return scores

    def compute_meteor(self, reference, hypothesis):
        return self.meteor.compute_score([reference], [hypothesis])[0]

    def compute_cider(self, references, hypotheses):
        return self.cider.compute_score({0: references}, {0: hypotheses})[0]

    def evaluate(self, reference, hypothesis):
        bleu = self.compute_bleu(reference, hypothesis)
        rouge = self.compute_rouge(reference, hypothesis)
        meteor = self.compute_meteor(reference, hypothesis)
        return {"BLEU": bleu, "ROUGE": rouge, "METEOR": meteor}

# Pipeline d'évaluation
def evaluate_pipeline(json_path, max_examples=5):
    dataset = load_json_dataset(json_path)
    print(f"Dataset loaded with {len(dataset)} examples.")

    gpt4_decoder = GPT4Decoder()
    evaluator = Evaluator()

    print("\n--- STARTING EVALUATION ---\n")
    results = []

    for i, example in enumerate(tqdm(dataset[:max_examples])):
        image_id = example.get("image_id", f"image_{i}")
        reference_caption = example.get("caption", "No caption available.")

        # Générer une légende
        generated_caption = gpt4_decoder.generate_caption(reference_caption)

        if generated_caption:
            metrics = evaluator.evaluate(reference_caption, generated_caption)
            results.append({"image_id": image_id, "metrics": metrics})

            # Afficher les résultats pour cet exemple
            print(f"Image ID: {image_id}")
            print(f"Reference Caption: {reference_caption}")
            print(f"Generated Caption: {generated_caption}")
            print(f"Metrics: {metrics}")
            print("-" * 50)
        else:
            print(f"Failed to generate caption for {image_id}")

    return results

# Script principal
if __name__ == "__main__":
    # Chemin vers votre fichier JSON
    json_file_path = "dataset/captions/vatex_test_captions.json"  # Remplacez par le chemin réel

    # Lancer le pipeline d'évaluation
    results = evaluate_pipeline(json_file_path, max_examples=5)

    # Sauvegarder les résultats
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation completed and results saved to evaluation_results.json.")
