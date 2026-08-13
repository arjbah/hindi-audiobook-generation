from datetime import datetime
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class RetrievalEvaluation(pl.Callback):
    def __init__(
        self,
        distance: str = "cosine",
        log_path: str = "train_log.txt",
        dataset_names: List[str] | None = None,
    ):
        self.distance = distance
        self.log_path = Path(log_path)
        self.dataset_names = dataset_names or ["Rasa"]

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            self._write(f"Total params: {total_params:,}")
            self._write(f"Trainable params: {trainable_params:,}")
            self._write(f" {self._timestamp()} | Using device: {trainer.strategy.root_device.type}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            pl_module.reset_retrieval_dict()
            return
        self._gather_log_and_reset(trainer, pl_module, f"Epoch [{trainer.current_epoch + 1}]", "val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._gather_log_and_reset(trainer, pl_module, "Test", "test")

    def _gather_log_and_reset(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        label: str,
        step_name: str,
    ) -> None:
        device = next(pl_module.parameters()).device
        retrieval_dicts = self._gather_retrieval_dict(pl_module.retrieval_dict, device)

        for dataset_idx, retrieval_dict in enumerate(retrieval_dicts):
            dataset_name = (
                self.dataset_names[dataset_idx]
                if dataset_idx < len(self.dataset_names)
                else f"Dataset {dataset_idx}"
            )
            for output_name, embeddings in retrieval_dict.items():
                if not embeddings["audio"]:
                    continue

                audio = torch.cat(embeddings["audio"]).cpu()
                text = torch.cat(embeddings["text"]).cpu()
                similarities = self._similarities(audio, text)
                a2t = self.compute_metrics(similarities)
                t2a = self.compute_metrics(similarities.t())

                output_slug = output_name.lower().replace(" ", "_")
                metric_prefix = f"Retrieval/{step_name}_{dataset_idx}/{output_slug}"
                logged_metrics = {
                    f"{metric_prefix}/A2T/{key}": value for key, value in a2t.items()
                }
                logged_metrics.update({
                    f"{metric_prefix}/T2A/{key}": value for key, value in t2a.items()
                })
                pl_module.log_dict(logged_metrics)

                if trainer.is_global_zero:
                    self._write(
                        f" {self._timestamp()} | {label} | {dataset_name} {output_name.lower()} metrics | "
                        f"T2A: {self._format_metrics(t2a)} | A2T: {self._format_metrics(a2t)}"
                    )

        pl_module.reset_retrieval_dict()

    def _similarities(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if audio.shape[0] != text.shape[0]:
            raise ValueError(
                "VoiceCLAP-style retrieval metrics require one matching text per audio, "
                f"but got {audio.shape[0]} audio and {text.shape[0]} text embeddings."
            )
        if self.distance == "cosine":
            return F.normalize(audio, dim=-1) @ F.normalize(text, dim=-1).t()
        if self.distance == "euclidean":
            return -torch.cdist(audio, text, p=2)
        raise ValueError(f"Unsupported retrieval distance: {self.distance}")

    @staticmethod
    def compute_metrics(similarities: torch.Tensor) -> dict[str, float]:
        targets = torch.arange(similarities.shape[0]).unsqueeze(1)
        sorted_indices = similarities.argsort(dim=1, descending=True)
        ranks = (sorted_indices == targets).long().argmax(dim=1) + 1

        return {
            "R@1": (ranks <= 1).float().mean().item(),
            "R@5": (ranks <= 5).float().mean().item(),
            "R@10": (ranks <= 10).float().mean().item(),
            "MedR": ranks.median().float().item(),
            "MeanR": ranks.float().mean().item(),
            "mAP": ranks.float().reciprocal().mean().item(),
        }

    def _gather_retrieval_dict(self, retrieval_dicts: List[dict], device: torch.device) -> List[dict]:
        retrieval_dicts = [self._normalize_retrieval_dict(item) for item in retrieval_dicts]
        if not torch.distributed.is_initialized():
            return retrieval_dicts

        gathered_dicts = []
        for retrieval_dict in retrieval_dicts:
            gathered = {
                name: {
                    modality: [self._gather_tensor(tensor, device) for tensor in tensors]
                    for modality, tensors in embeddings.items()
                }
                for name, embeddings in retrieval_dict.items()
            }
            gathered_dicts.append(gathered)
        return gathered_dicts

    @staticmethod
    def _normalize_retrieval_dict(retrieval_dict: dict) -> dict:
        if "A->T" in retrieval_dict:
            return {
                "Projections": {
                    "audio": retrieval_dict["A->T"]["query"],
                    "text": retrieval_dict["A->T"]["key"],
                }
            }
        return retrieval_dict

    @staticmethod
    def _gather_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = tensor.to(device)
        gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered)

    @staticmethod
    def _format_metrics(metrics: dict[str, float]) -> str:
        return (
            f"R@1={metrics['R@1']:.3f}, R@5={metrics['R@5']:.3f}, "
            f"R@10={metrics['R@10']:.3f}, MedR={metrics['MedR']:.3f}, "
            f"MeanR={metrics['MeanR']:.3f}, mAP={metrics['mAP']:.3f}"
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d at %H:%M:%S")

    def _write(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
