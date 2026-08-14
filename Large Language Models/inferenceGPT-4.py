import os
import json
import openai
from tqdm.auto import tqdm

# **Configuration de l'API OpenAI**
openai.api_key = "sk-YOUR-API-KEY"  # Remplacez par votre clé API OpenAI

# **Charger le fichier JSON**
def load_json_dataset(json_path):
    """
    Charge le dataset JSON contenant `image_id` et `caption`.
    Args:
        json_path (str): Chemin vers le fichier JSON.
    Returns:
        list: Liste des exemples du dataset.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

# **Génération de légendes avec GPT-4**
class GPT4Decoder:
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def generate_caption(self, image_id, prompt):
        """
        Génère une légende à l'aide de GPT-4.
        Args:
            image_id (str): ID de l'image (ou vidéo).
            prompt (str): Texte d'entrée pour GPT-4.
        Returns:
            str: Légende générée par GPT-4.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a video captioning assistant."},
                    {"role": "user", "content": f"Describe the video or image with ID '{image_id}'. Context: {prompt}"}
                ],
                temperature=self.temperature,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:  # Remplacer `openai.error.OpenAIError` par une exception générique
            print(f"Error generating caption for {image_id}: {e}")
            return None


# **Pipeline de test**
def test_pipeline(json_path, max_examples=5):
    """
    Teste la génération de légendes avec GPT-4 en utilisant un dataset JSON.
    Args:
        json_path (str): Chemin vers le fichier JSON.
        max_examples (int): Nombre maximum d'exemples à tester.
    """
    # Charger le dataset
    dataset = load_json_dataset(json_path)
    if not dataset:
        print("Dataset is empty or could not be loaded.")
        return

    print(f"Dataset loaded with {len(dataset)} examples.")

    # Initialiser le décodeur GPT-4
    gpt4_decoder = GPT4Decoder()

    # Itérer sur les exemples du dataset
    print("\n--- STARTING TEST ---\n")
    for i, example in enumerate(tqdm(dataset[:max_examples])):
        image_id = example.get("image_id", f"image_{i}")
        caption = example.get("caption", "No caption available.")

        # Générer une légende
        generated_caption = gpt4_decoder.generate_caption(image_id, caption)

        # Afficher les résultats
        print(f"Image ID: {image_id}")
        print(f"Original Caption: {caption}")
        print(f"Generated Caption: {generated_caption}")
        print("-" * 50)

# **Script principal**
if __name__ == "__main__":
    # Chemin vers votre fichier JSON
    json_file_path = "dataset/captions/vatex_test_captions.json"  # Remplacez par le chemin réel

    # Lancer le pipeline de test
    test_pipeline(json_file_path, max_examples=5)
