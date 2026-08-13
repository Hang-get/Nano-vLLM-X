from dataclasses import dataclass
from glob import glob
import os

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class WeightLoadReport:
    consumed: tuple[str, ...]
    skipped: tuple[str, ...]
    injected: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def map_eagle3_weight_name(name: str):
    if "t2d" in name:
        return None, None
    if "d2t" in name:
        return "draft_id_to_target_id", None
    name = name.replace("midlayer.", "model.layers.0.")
    for source, target, shard in (
        ("q_proj", "qkv_proj", "q"),
        ("k_proj", "qkv_proj", "k"),
        ("v_proj", "qkv_proj", "v"),
        ("gate_proj", "gate_up_proj", 0),
        ("up_proj", "gate_up_proj", 1),
    ):
        if source in name:
            return name.replace(source, target), shard
    if not name.startswith("lm_head.") and not name.startswith("model."):
        name = "model." + name
    return name, None


def _required_parameter_parts(model) -> set[tuple[str, str | int | None]]:
    required = set()
    for name, _ in model.named_parameters():
        if name == "model.embed_tokens.weight":
            continue
        if ".qkv_proj." in name:
            required.update((name, shard) for shard in ("q", "k", "v"))
        elif ".gate_up_proj." in name:
            required.update((name, shard) for shard in (0, 1))
        else:
            required.add((name, None))
    return required


def _load_parameter_part(param, tensor, shard, checkpoint_name: str) -> None:
    if tensor.dtype != param.dtype:
        raise ValueError(
            f"{checkpoint_name}: dtype expected {param.dtype}, got {tensor.dtype}"
        )
    loader = getattr(param, "weight_loader", None)
    if shard is None:
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"{checkpoint_name}: shape expected {tuple(param.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if loader is None:
            param.data.copy_(tensor)
        else:
            loader(param, tensor)
        return
    if loader is None:
        raise ValueError(f"{checkpoint_name}: packed parameter has no weight loader")
    try:
        loader(param, tensor, shard)
    except (AssertionError, RuntimeError) as exc:
        raise ValueError(
            f"{checkpoint_name}: incompatible packed tensor shape {tuple(tensor.shape)}"
        ) from exc


def _validate_injected_embedding(model, target_embedding) -> None:
    injected = model.model.embed_tokens
    if injected is not target_embedding or injected.weight is not target_embedding.weight:
        raise ValueError("draft embedding must alias the loaded target embedding")
    reference = model.model.fc.weight
    if injected.weight.size(1) != model.config.hidden_size:
        raise ValueError("target embedding hidden size mismatch")
    if injected.weight.size(0) != model.config.vocab_size:
        raise ValueError("target embedding vocabulary size mismatch")
    if injected.weight.dtype != reference.dtype:
        raise ValueError("target embedding dtype mismatch")
    if injected.weight.device != reference.device:
        raise ValueError("target embedding device mismatch")


def _validate_d2t(model) -> None:
    offsets = model.draft_id_to_target_id
    if offsets.numel() != model.config.draft_vocab_size:
        raise ValueError(
            "d2t length expected "
            f"{model.config.draft_vocab_size}, got {offsets.numel()}"
        )
    draft_ids = torch.arange(
        model.config.draft_vocab_size,
        device=offsets.device,
        dtype=offsets.dtype,
    )
    target_ids = (draft_ids + offsets).to(torch.long)
    if bool(((target_ids < 0) | (target_ids >= model.config.vocab_size)).any()):
        raise ValueError("d2t target IDs must be within target vocabulary")
    if target_ids.unique().numel() != target_ids.numel():
        raise ValueError("d2t target IDs must be unique")


def load_eagle3_weights(model, path: str, target_embedding) -> WeightLoadReport:
    _validate_injected_embedding(model, target_embedding)
    checkpoint_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not checkpoint_files:
        raise ValueError(f"draft checkpoint {path}: no safetensors files found")

    parameters = dict(model.named_parameters())
    required = _required_parameter_parts(model)
    loaded_parts: set[tuple[str, str | int | None]] = set()
    consumed = []
    skipped = []
    unexpected = []

    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework="pt", device="cpu") as weights:
            for checkpoint_name in weights.keys():
                parameter_name, shard = map_eagle3_weight_name(checkpoint_name)
                if parameter_name is None:
                    skipped.append(checkpoint_name)
                    continue
                if parameter_name == "model.embed_tokens.weight":
                    unexpected.append(checkpoint_name)
                    continue
                key = (parameter_name, shard)
                if parameter_name not in parameters or key not in required:
                    unexpected.append(checkpoint_name)
                    continue
                if key in loaded_parts:
                    unexpected.append(checkpoint_name)
                    continue
                _load_parameter_part(
                    parameters[parameter_name],
                    weights.get_tensor(checkpoint_name),
                    shard,
                    checkpoint_name,
                )
                loaded_parts.add(key)
                consumed.append(checkpoint_name)

    missing_parts = required - loaded_parts
    missing = tuple(
        sorted(
            name if shard is None else f"{name}[{shard}]"
            for name, shard in missing_parts
        )
    )
    unexpected_tuple = tuple(sorted(unexpected))
    if missing or unexpected_tuple:
        raise ValueError(
            f"strict EAGLE3 weight loading failed: missing={missing}, "
            f"unexpected={unexpected_tuple}"
        )

    _validate_d2t(model)
    return WeightLoadReport(
        consumed=tuple(sorted(consumed)),
        skipped=tuple(sorted(skipped)),
        injected=("model.embed_tokens.weight",),
        missing=(),
        unexpected=(),
    )
