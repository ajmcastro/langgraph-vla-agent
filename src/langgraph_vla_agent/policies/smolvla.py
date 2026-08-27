"""SmolVLAPolicyAdapter — wraps lerobot/smolvla_base behind the RobotPolicy Protocol.

Optional dependency: install the [vla] extra to use the real model.
    uv sync --extra dev --extra vla

Without the [vla] extra, SmolVLAPolicyAdapter can still be instantiated by
injecting a stub model via the ``_model`` parameter — used in unit tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation

try:
    import torch  # noqa: F401 — presence check only; used inside methods
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SmolVLAPolicy

    _vla_available = True
except ImportError:
    _vla_available = False


def vla_available() -> bool:
    """Return True if lerobot (and torch) are installed."""
    return _vla_available


# ---------------------------------------------------------------------------
# Internal protocol — the interface our adapter expects from the underlying model.
# Both SmolVLAPolicy (real) and _StubSmolVLAModel (test) satisfy this.
# ---------------------------------------------------------------------------


@runtime_checkable
class _SmolVLAModel(Protocol):
    def reset(self) -> None: ...

    def select_action(self, batch: dict[str, Any]) -> Any: ...  # noqa: ANN401


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE_H = 480
_DEFAULT_IMAGE_W = 640


class SmolVLAPolicyAdapter:
    """Wraps lerobot/SmolVLAPolicy behind the RobotPolicy Protocol.

    SmolVLA is a Vision-Language-Action model that takes a camera image,
    proprioceptive state, and a natural-language instruction and predicts
    a chunk of future actions. The chunk is buffered inside SmolVLAPolicy;
    select_action() returns one action per call.

    For unit tests, inject a stub via the ``_model`` parameter so the [vla]
    extra is not required:

        adapter = SmolVLAPolicyAdapter(_model=_StubSmolVLAModel())

    Parameters
    ----------
    model_id:
        HuggingFace Hub model ID (default: ``"lerobot/smolvla_base"``).
        Ignored when ``_model`` is supplied.
    device:
        Torch device string (``"cpu"``, ``"mps"``, ``"cuda"``). Defaults to
        ``"cpu"`` for safe cross-platform operation.
    image_key:
        Key into ``RobotObservation.images`` for the camera to use.
        Defaults to ``"front"`` (matches the SO-100 dataset convention).
    image_size:
        ``(height, width)`` of the dummy image created when no camera image
        is present in the observation (e.g., fixture-based unit tests).
    _model:
        Inject a pre-built model object. Skips the lerobot import and Hub
        download entirely. Intended for testing only.
    """

    def __init__(
        self,
        model_id: str = "lerobot/smolvla_base",
        *,
        device: str = "cpu",
        image_key: str = "front",
        image_size: tuple[int, int] = (_DEFAULT_IMAGE_H, _DEFAULT_IMAGE_W),
        _model: _SmolVLAModel | None = None,
    ) -> None:
        if _model is not None:
            self._model: _SmolVLAModel = _model
            self._tokenizer: Any = None
        elif _vla_available:
            loaded = _SmolVLAPolicy.from_pretrained(model_id)
            loaded = loaded.to(device)
            loaded.eval()
            self._model = loaded
            # Cache the tokenizer from the model's internal VLM processor.
            # Used in _build_batch to produce observation.language.tokens.
            try:
                self._tokenizer = loaded.model.vlm_with_expert.processor.tokenizer
            except AttributeError:
                self._tokenizer = None
        else:
            raise ImportError(
                "lerobot is required for SmolVLAPolicyAdapter. "
                "Install the [vla] extra:  uv sync --extra dev --extra vla\n"
                "For unit tests, inject a stub:  SmolVLAPolicyAdapter(_model=stub)"
            )
        self._model_id = model_id
        self._device = device
        self._image_key = image_key
        self._image_size = image_size

    @property
    def model_id(self) -> str:
        return self._model_id

    def reset(self, context: PolicyContext) -> None:
        """Reset the action-chunk buffer inside the underlying model."""
        self._model.reset()

    def act(self, observation: RobotObservation, instruction: str) -> RobotAction:
        """Predict the next action given a robot observation and instruction.

        Calls ``model.select_action(batch)`` once. SmolVLA manages the action
        chunk buffer internally — this returns the next buffered action or
        triggers a new forward pass when the buffer is empty.

        The observation's camera image is used if available; otherwise a dummy
        black image of ``image_size`` is created (see class docstring).
        """
        batch = self._build_batch(observation, instruction)
        raw = self._model.select_action(batch)
        return self._to_robot_action(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_batch(self, observation: RobotObservation, instruction: str) -> dict[str, Any]:
        if not _vla_available:
            # Stub mode: the stub model ignores the batch format entirely.
            return {
                "observation.state": observation.state,
                "task": [instruction],
            }

        import torch  # available because _vla_available is True

        state_tensor = torch.tensor(observation.state, dtype=torch.float32).unsqueeze(0)
        batch: dict[str, Any] = {
            "observation.state": state_tensor.to(self._device),
        }

        # Tokenize the instruction — SmolVLA's select_action reads
        # observation.language.tokens and observation.language.attention_mask
        # directly from the batch (pre-tokenized, not raw strings).
        if self._tokenizer is not None:
            tok = self._tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch["observation.language.tokens"] = tok["input_ids"].to(self._device)
            # SmolVLA's attention mechanism expects a boolean mask, not Long (0/1)
            batch["observation.language.attention_mask"] = (
                tok["attention_mask"].bool().to(self._device)
            )
        else:
            # Fallback for models without an introspectable tokenizer
            batch["task"] = [instruction]

        # Introspect the loaded checkpoint's expected image keys and shapes so
        # the batch matches regardless of which model is loaded.  Falls back to
        # the constructor's image_key / image_size when config is absent.
        image_features: dict[str, Any] = {}
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            image_features = getattr(cfg, "image_features", {}) or {}

        if image_features:
            for key, feature in image_features.items():
                c, h, w = feature.shape  # (C, H, W) e.g. (3, 256, 256)
                # Match observation image by full key, short camera name, or image_key hint
                short_key = key.split(".")[-1]  # "camera1" from "observation.images.camera1"
                raw = (
                    observation.images.get(key)
                    or observation.images.get(short_key)
                    or observation.images.get(self._image_key)
                )
                if raw is None:
                    img_t = torch.zeros(1, c, h, w, dtype=torch.float32)
                else:
                    img_t = (
                        torch.tensor(raw, dtype=torch.float32)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .div(255.0)
                    )
                batch[key] = img_t.to(self._device)
        else:
            # Fallback: single image under the constructor's image_key
            h, w = self._image_size
            raw_img: npt.NDArray[Any] | None = observation.images.get(self._image_key)
            if raw_img is None:
                raw_img = np.zeros((h, w, 3), dtype=np.uint8)
            img_t = (
                torch.tensor(raw_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255.0)
            )
            batch[f"observation.images.{self._image_key}"] = img_t.to(self._device)

        return batch

    def _to_robot_action(self, raw: Any) -> RobotAction:  # noqa: ANN401
        """Convert model output (torch.Tensor or numpy array) to RobotAction."""
        if isinstance(raw, np.ndarray):
            values: npt.NDArray[Any] = raw.flatten().astype(np.float32)
        elif hasattr(raw, "cpu"):
            # torch.Tensor path — detach from autograd graph, move to CPU
            values = raw.cpu().detach().numpy().flatten().astype(np.float32)
        else:
            raise TypeError(
                f"Unexpected action type from underlying model: {type(raw).__name__}. "
                "Expected torch.Tensor or numpy.ndarray."
            )
        return RobotAction(values=values)


# ---------------------------------------------------------------------------
# Stub for unit tests
# ---------------------------------------------------------------------------


class _StubSmolVLAModel:
    """Minimal stub that satisfies _SmolVLAModel for unit tests.

    Returns a zero-vector of configurable length. No torch required.
    """

    def __init__(self, action_dim: int = 6) -> None:
        self.action_dim = action_dim
        self.reset_call_count: int = 0
        self.act_call_count: int = 0

    def reset(self) -> None:
        self.reset_call_count += 1

    def select_action(self, batch: dict[str, Any]) -> npt.NDArray[Any]:
        self.act_call_count += 1
        return np.zeros(self.action_dim, dtype=np.float32)
