import warnings
from pathlib import Path

import torch

from .features import compute_terraseg_features
from .model import build_terraseg
from .norm import replace_bn1d_with_gn

# Default URI scheme used to refer to a checkpoint hosted on the Hugging Face Hub.
HF_URI_PREFIX: str = "hf://"

# Default Hugging Face repo holding the released TerraSeg weights.
DEFAULT_HF_REPO_ID: str = "TedLentsch/TerraSeg"


def resolve_checkpoint_path(
    checkpoint_path: str | Path,
    hf_revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a checkpoint reference into a local filesystem path.

    Accepts either a regular local path or a Hugging Face URI of the form
    ``hf://<repo_id>/<filename>`` (e.g. ``hf://TedLentsch/TerraSeg/terraseg_s.pth``).
    For HF URIs, the file is downloaded once and cached via ``huggingface_hub``.

    Args:
        checkpoint_path (str or Path) : Either a local path or an ``hf://`` URI.
        hf_revision (str or None) : Optional revision (branch, tag, or commit hash) when
            downloading from Hugging Face. Default: ``None`` (latest on ``main``).
        cache_dir (str or Path or None) : Optional override of the Hugging Face cache directory.

    Returns:
        local_path (Path) : Filesystem path to the resolved checkpoint.
    """
    s = str(checkpoint_path)
    if not s.startswith(HF_URI_PREFIX):
        return Path(s)

    # Lazy import so non-HF users do not need huggingface_hub installed.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Loading TerraSeg weights from Hugging Face requires the 'huggingface_hub' "
            "package. Install it with `uv add huggingface_hub` or `pip install huggingface_hub`."
        ) from e

    spec = s[len(HF_URI_PREFIX) :]
    parts = spec.split("/", 2)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid Hugging Face URI '{s}'. Expected format "
            f"'hf://<user-or-org>/<repo>/<filename>'."
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]

    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=hf_revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local)


class TerraSegPredictor:
    """
    Single-frame TerraSeg ground / non-ground predictor for deployment.

    Loads a trained TerraSeg checkpoint once at construction time, applies the
    BatchNorm-to-GroupNorm swap (paper section 3.4), moves the model to the requested device, and
    exposes a :meth:`predict` method that takes a single (N, 3) point cloud and returns
    per-point binary labels.

    The predictor loads the model weights under the ``"model_state_dict"`` key, falling
    back to treating the checkpoint as a bare ``state_dict``. The decision threshold is
    loaded from the checkpoint if present.

    TerraSeg runs in FP32 only. Lower-precision dtypes (BF16, FP16) are not supported because
    PTv3's sparse-convolution path becomes numerically unstable at reduced precision -- the
    forward pass does not error, but training diverges and inference quality degrades
    silently. The dtype is therefore hard-wired to ``torch.float32`` and not user-configurable.

    Args:
        variant (str) : Either ``"B"`` or ``"S"``; must match the checkpoint architecture.
        checkpoint_path (str or Path) : Either a local filesystem path to a ``best.pth``
            produced by ``terraseg_train.py``, or a Hugging Face URI of the form
            ``hf://<user>/<repo>/<filename>`` (e.g. ``hf://TedLentsch/TerraSeg/terraseg_s.pth``).
            HF URIs are downloaded and cached via ``huggingface_hub``.
        device (str or torch.device or None) : Device on which to run inference. Default:
            ``"cuda:0"`` if available else ``"cpu"``.
        decision_thres (float or None) : Decision threshold applied to the sigmoid of the
            logits. If None, the threshold stored in the checkpoint is used (falling back to
            0.5 if absent).
        compile_model (bool) : If True, wrap the model with :func:`torch.compile`. Adds a
            one-time compilation cost, warmed up at construction. Note that PTv3's spconv
            path is opaque to inductor, so the gain is limited; eager is the recommended
            default. Default: False.
        tf32 (bool) : If True (default), enable TF32 tensor-core matmuls for a ~20% speedup
            on Ampere+ GPUs. Slightly reduces matmul precision versus strict FP32; set False
            to reproduce the paper's exact numerics.
        hf_revision (str or None) : Optional revision (branch, tag, or commit hash) when
            loading weights via an ``hf://`` URI. Default: ``None`` (latest on ``main``).
        cache_dir (str or Path or None) : Optional override of the Hugging Face cache
            directory. Default: ``None`` (use the platform default).

    Attributes:
        model (terraseg.TerraSeg) : Loaded model in eval mode on the target device.
        decision_thres (float) : Decision threshold used by :meth:`predict`.
        device (torch.device) : Device the model lives on.
    """

    def __init__(
        self,
        variant: str,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        decision_thres: float | None = None,
        compile_model: bool = False,
        tf32: bool = True,
        hf_revision: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

        # TF32 routes FP32 matmuls onto Ampere+ tensor cores for a sizeable speedup.
        # It is not bit-identical FP32 (reduced matmul mantissa), so it is exposed as a
        # flag: default on for deployment speed, set tf32=False to reproduce the paper's
        # exact FP32 numerics. Safe no-op on pre-Ampere GPUs and CPU.
        if tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Resolve checkpoint location (local path or Hugging Face URI).
        local_ckpt = resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            hf_revision=hf_revision,
            cache_dir=cache_dir,
        )
        if not local_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local_ckpt}")

        checkpoint = torch.load(str(local_ckpt), map_location="cpu", weights_only=False)

        model = build_terraseg(variant=variant, input_dim=3, num_classes=1)
        model = replace_bn1d_with_gn(model=model, groups_default=32)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

        # TerraSeg runs in FP32: PTv3's sparse-conv path is unstable at lower precision.
        model = model.to(self.device)
        model.eval()

        self.model = model

        # Set decision threshold before any warm-up forward pass.
        if decision_thres is not None:
            self.decision_thres = float(decision_thres)
        else:
            self.decision_thres = float(checkpoint.get("decision_thres", 0.5))

        if compile_model:
            eager_model = self.model
            self.model = torch.compile(eager_model)
            # torch.compile is lazy: the backend only runs on the first forward. Trigger
            # it here inside a try/except so a compile failure degrades to eager instead
            # of crashing on the first real scan. Also pays the one-time cost up front.
            try:
                warmup_coord = torch.randn(2048, 3, device=self.device) * 20.0
                _ = self.predict_probs(coord=warmup_coord)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"torch.compile warm-up failed ({e!r}); using eager mode.",
                    stacklevel=2,
                )
                self.model = eager_model

    @torch.inference_mode()
    def predict(
        self,
        coord: torch.Tensor,
        grid_size: float = 0.05,
        source_dataset: str = "deploy",
    ) -> torch.Tensor:
        """
        Run a single-frame prediction.

        Args:
            coord (torch.Tensor) : (N, 3) per-point XYZ coordinates in the TerraSeg-standardized
                frame (z = 0 approximately ground-aligned, +x forward). Unit: meters.
            grid_size (float) : PTv3 voxel grid size. Unit: meters. Default: 0.05.
            source_dataset (str) : Per-scan dataset tag passed to PTv3 as ``condition``. With
                ``pdnorm_conditions=None`` (TerraSeg's domain-agnostic setting) this value does
                not affect normalization, but PTv3 still expects a string here.

        Returns:
            pred_labels (torch.Tensor) : (N,) per-point binary labels of dtype ``torch.uint8``,
                with 0 = ground and 1 = non-ground, on the predictor's device.
        """
        pred_probs = self.predict_probs(
            coord=coord, grid_size=grid_size, source_dataset=source_dataset
        )
        pred_labels = (pred_probs > self.decision_thres).to(torch.uint8)
        return pred_labels

    @torch.inference_mode()
    def predict_probs(
        self,
        coord: torch.Tensor,
        grid_size: float = 0.05,
        source_dataset: str = "deploy",
    ) -> torch.Tensor:
        """
        Run a single-frame prediction returning continuous probabilities instead of labels.

        Args:
            coord (torch.Tensor) : (N, 3) per-point XYZ coordinates. Unit: meters.
            grid_size (float) : PTv3 voxel grid size. Unit: meters. Default: 0.05.
            source_dataset (str) : Per-scan dataset tag passed to PTv3 as ``condition``.

        Returns:
            pred_probs (torch.Tensor) : (N,) per-point non-ground probabilities in [0, 1]
                of dtype ``torch.float32``, on the predictor's device.
        """
        assert coord.ndim == 2 and coord.shape[1] == 3, (
            f"coord must have shape (N, 3), got {tuple(coord.shape)}!"
        )

        coord = coord.to(self.device, dtype=torch.float32, non_blocking=True)
        feat = compute_terraseg_features(coord=coord)

        offset = torch.tensor([coord.shape[0]], dtype=torch.long, device=self.device)
        batch = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": float(grid_size),
            "condition": [source_dataset],
        }
        pred_logits = self.model(batch)
        pred_probs = torch.sigmoid(pred_logits.float())
        return pred_probs
