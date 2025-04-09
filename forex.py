from pathlib import Path

model_id = "hello"
current_dir = Path(__file__).parent / "models"
model_path = current_dir / f"{model_id}" / "model"
print(model_path)
