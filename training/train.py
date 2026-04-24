import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from dataset import MultiModalDataset
from pathlib import Path
from types import SimpleNamespace
from act import ACT
import yaml
from act import _to_batched_time_major
from tqdm import tqdm

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
model_device = get_device()
#This is the training loop --- it should include a function that can be used to compute a loss

dataset_path = "/Users/rahulavasarala/Desktop/ForceWM/data_storage/no_coll_dataset_v1_extracted"
contract_path = "/Users/rahulavasarala/Desktop/ForceWM/universal_contract.yaml"
checkpoint_dir = Path("/Users/rahulavasarala/Desktop/ForceWM/training/checkpoints")

# ---------------- Training Hyperparameters ---------------------

EPOCHS = 100
LR=3.0e-4
BATCH_SIZE = 4

# ----------------- Training Hyperparameters ---------------------

def compute_loss(output, train_data):

    ground_truth = _to_batched_time_major(
            torch.concat(
                (train_data["actions"]["eef_pos"], train_data["actions"]["eef_ori"]),
                dim=-1,
            ),
            "action",
        ).to(device=model_device)

    return torch.nn.functional.mse_loss(output, ground_truth)

def load_act_model():

    config_path = Path("/Users/rahulavasarala/Desktop/ForceWM/training/act_config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)
    model = ACT(config).to(model_device)

    return model

def act_train_loop(): 

    train_dataset = MultiModalDataset(dataset_path, contract_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    act = load_act_model()
    act.train()
    optimizer = torch.optim.Adam(act.parameters(), lr=LR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch", leave=False)
        for batch_idx, train_data in enumerate(batch_bar, start=1):
            output, _ = act(train_data)
            loss = compute_loss(output, train_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = running_loss / max(len(train_loader), 1)
        epoch_bar.set_postfix(loss=f"{average_loss:.4f}")

        checkpoint_path = checkpoint_dir / f"act_epoch_{epoch + 1:04d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": act.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
            },
            checkpoint_path,
        )

        

if __name__ == "__main__":
    act_train_loop()




# for key, value in train_data.items():
#             if isinstance(value, dict):
#                 print("here")
#                 for sub_key, sub_value in value.items():
#                     print(f"{key} {sub_key}: {sub_value.shape}")
#             else:
#                 print("here")
#                 print(f"{key}: {value.shape}")
