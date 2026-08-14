import json
import os

from datasets import Dataset, DatasetDict, load_from_disk
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, get_scheduler

print("Transformers fonctionne correctement")

# Détection du périphérique (CPU ou GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL
encoder = "facebook/timesformer-base-finetuned-k600"
decoder = "gpt2"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained(decoder)
tokenizer.pad_token = tokenizer.eos_token

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to(device)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.max_length = 50
model.config.num_beams = 4
model.config.early_stopping = True

# DATASET
def load_or_prepare_dataset():
    dataset_path = "model"  # Update this path as required
    train_path = "dataset/captions/vatex_train_captions.json"  # Update to the actual JSON train file
    val_path = "dataset/captions/vatex_val_captions.json"  # Update to the actual JSON validation file

    try:
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded from disk.")
    except FileNotFoundError:
        print("Dataset not found on disk. Preparing from JSON...")
        train_data = Dataset.from_json(train_path)
        val_data = Dataset.from_json(val_path)
        dataset = DatasetDict({
            "train": train_data,
            "validation": val_data
        })
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}.")
    return dataset

# Configuration du modèle pour correspondre aux frames réduites
model.encoder.config.num_frames = 4  # Frames ajustées pour CPU
model.encoder.config.image_size = 112  # Taille réduite des frames

def preprocess_dataset(dataset):
    # Prétraitement pour générer les données simulées
    def process_example(example):
        num_frames = 4
        num_channels = 3
        height, width = 112, 112

        # Générer des données de pixel_values avec les bonnes dimensions
        pixel_values = torch.rand(num_channels, num_frames, height, width)
        assert pixel_values.shape[0] == 3, f"Invalid number of channels during preprocessing: {pixel_values.shape[0]}"
        assert pixel_values.shape[1] == 4, f"Invalid number of frames during preprocessing: {pixel_values.shape[1]}"

        example["pixel_values"] = pixel_values.tolist()
        return example

    dataset = dataset.map(process_example, load_from_cache_file=False)
    return dataset


class VatexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 10 * len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):  # Gestion des indices par lot
            return [self.__getitem__(i) for i in idx]

        # Si idx est un entier (élément unique)
        video_idx = idx // 10
        caption_idx = idx % 10
        example = self.dataset[video_idx]

        if "pixel_values" not in example:
            raise KeyError("Missing 'pixel_values' key in dataset.")

        pixel_values = torch.tensor(example["pixel_values"]).permute(0, 1, 2, 3)  # [channels, frames, height, width]
        assert pixel_values.shape[0] == 3, f"Invalid number of channels in __getitem__: {pixel_values.shape[0]}"
        assert pixel_values.shape[1] == 4, f"Invalid number of frames in __getitem__: {pixel_values.shape[1]}"

        return {
            "videoID": example["videoID"],
            "pixel_values": pixel_values,
            "labels": tokenizer(
                example["enCap"][caption_idx], padding="max_length", truncation=True, max_length=50
            ).input_ids,
        }


def custom_collate_fn(batch):
    videoIDs = [item["videoID"] for item in batch]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])

    return {
        "videoID": videoIDs,
        "pixel_values": pixel_values,
        "labels": labels,
    }


if __name__ == "__main__":
    dataset = load_or_prepare_dataset()
    dataset = preprocess_dataset(dataset)

    dataset.set_format("torch")
    dataset_train = VatexDataset(dataset["train"])
    dataset_val = VatexDataset(dataset["validation"])
    print("DATASET: train - %d, validation - %d" % (len(dataset_train), len(dataset_val)))

    kwargs = {
        "batch_size": 2,
        "drop_last": True,
        "num_workers": 1,
        "pin_memory": False,
    }

    train_dataloader = DataLoader(dataset_train, collate_fn=custom_collate_fn, shuffle=True, **kwargs)
    val_dataloader = DataLoader(dataset_val, collate_fn=custom_collate_fn, **kwargs)

    OUTPUT_DIR = "model"
    EPOCHS = 2

    scaler = GradScaler(enabled=False)  # Désactivé sur CPU
    optimizer = AdamW(model.parameters(), lr=1e-5)
    training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps)

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))
    train_progress = tqdm(range(training_steps))

    for epoch in range(EPOCHS):
        train_loss, val_loss = 0, 0

        model.train()
        for batch in train_dataloader:
            # Déplacer les données sur le bon périphérique
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Vérification des dimensions des données
            assert pixel_values.shape[1] == 3, "Number of channels must be 3"
            assert pixel_values.shape[2] == 4, "Number of frames must be 4"

            # Réinitialiser les gradients
            optimizer.zero_grad()

            # Passer les données dans le modèle
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Calcul de la perte
            loss = outputs.loss
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_progress.update(1)

        # Validation après l'entraînement
        model.eval()
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    labels=batch["labels"].to(device),
                )
            val_loss += outputs.loss.item()

        # Logs pour TensorBoard
        writer.add_scalar("Loss/train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/val", val_loss / len(val_dataloader), epoch)
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint_{epoch + 1}"))

    print("Entraînement terminé.")