from __future__ import annotations

import json
from pathlib import Path

from transformers import Trainer


class SparseMemoryTrainer(Trainer):
    def __init__(self, *args, sparse_controller=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_controller = sparse_controller

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        self.sparse_controller.reset_batch_counts()
        outputs = model(**inputs)
        loss = outputs.loss
        self.sparse_controller.update_masks_from_current_batch()
        return (loss, outputs) if return_outputs else loss


def save_log_history(trainer: Trainer, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = trainer.state.log_history
    (output_dir / "trainer_log_history.json").write_text(json.dumps(payload, indent=2) + "\n")

