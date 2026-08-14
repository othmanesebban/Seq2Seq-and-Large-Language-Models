
import json
import os

from datasets import Dataset, DatasetDict, load_from_disk
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, default_data_collator, get_scheduler

print("Transformers fonctionne correctement")

device = "cpu"

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
        # Try to load the dataset from disk
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded from disk.")
    except FileNotFoundError:
        print("Dataset not found on disk. Preparing from JSON...")

        # Load from JSON files
        train_data = Dataset.from_json(train_path)
        val_data = Dataset.from_json(val_path)

        # Combine into DatasetDict
        dataset = DatasetDict({
            "train": train_data,
            "validation": val_data
        })

        # Save for future use
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}.")

    return dataset


def preprocess_dataset(dataset):
    # à l'aide de GPU
    def process_example(example):
        num_frames = 16  # Par exemple, 8 ou 16 images par vidéo
        num_channels = 3  # Canaux RGB
        height, width = 224, 224  # Dimensions des images
        # Générer des valeurs fictives pour une vidéo
        example["pixel_values"] = torch.rand(num_frames, num_channels, height, width).tolist()  # Simuler une vidéo
        return example

    # à l'aide de CPU
    #def process_example(example):
        #num_frames = 8  # Réduction du nombre de frames
        #num_channels = 3
        #height, width = 112, 112  # Réduction de la résolution
        #example["pixel_values"] = torch.rand(num_frames, num_channels, height, width)  # Reste un tenseur
        #return example

    # Applique le prétraitement
    dataset = dataset.map(process_example, load_from_cache_file=False)
    return dataset  # Retourne le dataset modifié




class VatexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 10 * len(self.dataset)

    def __getitems__(self, idxs):
        items = []
        for idx in idxs:
            video_idx = idx // 10
            caption_idx = idx % 10
            example = self.dataset[video_idx]
            if "pixel_values" not in example:
                raise KeyError("Missing 'pixel_values' key in dataset.")
            items.append({
                "videoID": example["videoID"],
                "pixel_values": example["pixel_values"],
                "labels": example["enCap"][caption_idx]
            })
        return items


def val_collator(examples):
    videoID, pixel_values, labels = [], [], []
    for example in examples:
        videoID.append(example["videoID"])
        pixel_values.append(torch.tensor(example["pixel_values"]))
        labels.append(tokenizer(example["labels"], padding="max_length", truncation=True, max_length=50).input_ids)

    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels)
    return {"videoID": videoID, "pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    # Load the dataset
    dataset = load_or_prepare_dataset()

    # Preprocess the dataset to ensure 'pixel_values' exists
    dataset = preprocess_dataset(dataset)

    dataset.set_format("torch")
    dataset_train = VatexDataset(dataset["train"])
    dataset_val = VatexDataset(dataset["validation"])
    print("DATASET: train - %d, validation - %d" % (len(dataset_train), len(dataset_val)))

    #kwargs = {
        #"batch_size": 6,
        #"drop_last": True,
        #"num_workers": 0,  # Set to 0 to avoid multiprocessing issues on Windows
        #"pin_memory": True,
    #}
    kwargs = {
        "batch_size": 6,  # Taille de batch réduite
        "drop_last": True,
        "num_workers": 8,  # Parallélisme pour accélérer
        "pin_memory": False,
    }

    train_dataloader = DataLoader(dataset_train, collate_fn=default_data_collator, shuffle=True, **kwargs)
    val_dataloader = DataLoader(dataset_val, collate_fn=val_collator, **kwargs)

    # TRAINING
    OUTPUT_DIR = "model"  # Update this path as required
    EPOCHS = 2

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=5e-7)
    training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )

    # VALIDATION
    videoID_captions_path = "dataset/videoID_captions.json"  # Update to the actual JSON file
    with open(videoID_captions_path) as file:
        videoID_captions = json.load(file)

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))
    train_progress = tqdm(range(training_steps))
    for epoch in range(EPOCHS):
        train_loss, val_loss = 0, 0

        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
                train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_progress.update(1)

        model.eval()
        val_progress = tqdm(range(len(val_dataloader)))
        for batch in val_dataloader:
            videoIDs = batch.pop("videoID")
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch)
            val_loss += outputs.loss.item()
            val_progress.update(1)

        writer.add_scalar("Loss/train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/val", val_loss / len(val_dataloader), epoch)
        model.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint_%d" % (epoch + 1)))
