import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoImageProcessor
from tqdm.auto import tqdm
import openai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Configuration de l'API OpenAI
openai.api_key = os.getenv("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")  # Définir OPENAI_API_KEY dans votre environnement

# Détection du périphérique (CPU ou GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modèle d'encodeur
encoder = "facebook/timesformer-base-finetuned-k600"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Fonction pour charger et préparer le dataset
def load_or_prepare_dataset():
    train_path = "dataset/captions/vatex_train_captions.json"
    val_path = "dataset/captions/vatex_val_captions.json"
    dataset_path = "model"

    try:
        dataset = DatasetDict.load_from_disk(dataset_path)
        print("Dataset loaded from disk.")
    except FileNotFoundError:
        print("Dataset not found on disk. Preparing from JSON...")
        train_data = Dataset.from_json(train_path)
        val_data = Dataset.from_json(val_path)
        dataset = DatasetDict({"train": train_data, "validation": val_data})
        dataset.save_to_disk(dataset_path)
    return dataset

# Prétraitement des données
def preprocess_dataset(dataset):
    def process_example(example):
        num_frames = 4
        num_channels = 3
        height, width = 112, 112
        example["pixel_values"] = torch.rand(num_channels, num_frames, height, width).tolist()
        return example

    dataset = dataset.map(process_example, load_from_cache_file=False)
    return dataset

# Normalisation des légendes (prétraitement)
def normalize_caption(caption):
    import string
    return caption.lower().translate(str.maketrans("", "", string.punctuation)).strip()

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

# Classe pour le dataset Vatex
class VatexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        pixel_values = torch.tensor(example["pixel_values"]).permute(0, 1, 2, 3)
        return {
            "videoID": example.get("videoID", f"video_{idx}"),
            "pixel_values": pixel_values,
            "caption_prompt": example["enCap"][0],  # Première légende comme prompt
        }

# Fonction d'entraînement
def train_model(train_dataloader, decoder, epochs=1):
    train_progress = tqdm(range(epochs * len(train_dataloader)))
    for epoch in range(epochs):
        for batch in train_dataloader:
            videoID = batch["videoID"]
            caption_prompts = batch["caption_prompt"]

            for video_id, prompt in zip(videoID, caption_prompts):
                caption = decoder.generate_caption(prompt)
                if caption:
                    print(f"Generated Caption for {video_id}: {caption}")
                else:
                    print(f"Failed to generate caption for {video_id}")
            train_progress.update(1)

# Fonction d'évaluation
def evaluate_model(val_dataloader, decoder):
    evaluation_results = []
    val_progress = tqdm(val_dataloader, desc="Evaluating")

    smoothing = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for batch in val_progress:
        videoID = batch["videoID"]
        caption_prompts = batch["caption_prompt"]

        for video_id, prompt in zip(videoID, caption_prompts):
            generated_caption = decoder.generate_caption(prompt)

            if not generated_caption:
                print(f"Failed to generate caption for video ID: {video_id}")
                bleu_score = meteor = rouge_score = 0
            else:
                # Normalisation
                reference_caption = normalize_caption(prompt)
                generated_caption = normalize_caption(generated_caption)

                # Calcul des scores
                bleu_score = sentence_bleu([reference_caption.split()], generated_caption.split(),
                                           smoothing_function=smoothing)
                meteor = meteor_score([reference_caption], generated_caption)
                rouge_score = rouge.score(reference_caption, generated_caption)['rougeL'].fmeasure

            # Ajouter aux résultats
            evaluation_results.append({
                "videoID": video_id,
                "reference_caption": prompt,
                "generated_caption": generated_caption,
                "bleu_score": bleu_score,
                "meteor_score": meteor,
                "rougeL_score": rouge_score,
            })

    # Calculer les moyennes
    average_bleu = sum(r["bleu_score"] for r in evaluation_results) / len(evaluation_results)
    average_meteor = sum(r["meteor_score"] for r in evaluation_results) / len(evaluation_results)
    average_rougeL = sum(r["rougeL_score"] for r in evaluation_results) / len(evaluation_results)

    print(f"Average BLEU score: {average_bleu:.4f}")
    print(f"Average METEOR score: {average_meteor:.4f}")
    print(f"Average ROUGE-L score: {average_rougeL:.4f}")

    return evaluation_results


if __name__ == "__main__":
    # Charger et préparer le dataset
    dataset = load_or_prepare_dataset()
    dataset = preprocess_dataset(dataset)
    dataset_train = VatexDataset(dataset["train"])
    dataset_val = VatexDataset(dataset["validation"])

    train_kwargs = {"batch_size": 2, "num_workers": 1, "pin_memory": False}
    train_dataloader = DataLoader(dataset_train, shuffle=True, **train_kwargs)

    val_kwargs = {"batch_size": 2, "num_workers": 1, "pin_memory": False}
    val_dataloader = DataLoader(dataset_val, shuffle=False, **val_kwargs)

    # Initialiser le décodeur GPT-4
    gpt4_decoder = GPT4Decoder()

    # Entraîner le modèle
    print("Training the model...")
    train_model(train_dataloader, gpt4_decoder, epochs=1)

    # Évaluer le modèle
    print("Evaluating the model...")
    results = evaluate_model(val_dataloader, gpt4_decoder)

    # Sauvegarder les résultats d'évaluation
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Évaluation terminée. Résultats sauvegardés dans 'evaluation_results.json'.")
