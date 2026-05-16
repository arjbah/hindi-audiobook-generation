"""
Quick single-sentence IndicF5 smoke test.
Run: python test_indicf5.py
Output: test_output.wav in the project root.
"""
import faulthandler, sys, types
faulthandler.enable()  # dumps a native stack on segfault

# pyarrow-on-Windows access-violation workaround: shim `datasets` before
# IndicF5/f5_tts pulls it in.  Inference doesn't need it.
if "datasets" not in sys.modules:
    class _Stub:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, name): return _Stub()
    _fake = types.ModuleType("datasets")
    for _n in ("Dataset", "IterableDataset", "DatasetDict", "IterableDatasetDict",
               "load_dataset", "load_from_disk", "concatenate_datasets",
               "Features", "Value", "Audio", "Sequence", "ClassLabel", "Array2D"):
        setattr(_fake, _n, _Stub)
    sys.modules["datasets"] = _fake

# Disable accelerate.init_empty_weights meta-device context so IndicF5's
# Vocos+MelSpectrogram construction doesn't hit a meta/cpu tensor mismatch.
import contextlib
_fake_ctx = lambda *a, **k: contextlib.nullcontext()
for _mod_name in ("accelerate.big_modeling", "accelerate", "transformers.modeling_utils"):
    try:
        _mod = __import__(_mod_name, fromlist=["init_empty_weights"])
        if hasattr(_mod, "init_empty_weights"):
            _mod.init_empty_weights = _fake_ctx
    except ImportError:
        pass

# Backstop: any DeviceContext("meta") becomes DeviceContext("cpu")
import torch as _torch_for_patch
import torch.utils._device as _dev_mod
_orig_dc_init = _dev_mod.DeviceContext.__init__
def _patched_dc_init(self, device):
    dev = _torch_for_patch.device(device)
    if dev.type == "meta":
        dev = _torch_for_patch.device("cpu")
    _orig_dc_init(self, dev)
_dev_mod.DeviceContext.__init__ = _patched_dc_init

# IndicF5 calls f5_tts.load_model without ckpt_path; new f5_tts requires it.
try:
    import f5_tts.infer.utils_infer as _ui
    _orig_load_model = _ui.load_model
    def _patched_load_model(model_cls, model_cfg, ckpt_path=None,
                            mel_spec_type=None, vocab_file="", ode_method=None,
                            use_ema=True, device=None):
        if mel_spec_type is None: mel_spec_type = _ui.mel_spec_type
        if ode_method is None: ode_method = _ui.ode_method
        if device is None: device = _ui.device
        if ckpt_path:
            return _orig_load_model(model_cls, model_cfg, ckpt_path,
                                    mel_spec_type, vocab_file, ode_method,
                                    use_ema, device)
        from importlib.resources import files as _files
        from f5_tts.model import CFM as _CFM
        from f5_tts.model.utils import get_tokenizer as _get_tok
        if not vocab_file:
            vocab_file = str(_files("f5_tts").joinpath("infer/examples/vocab.txt"))
        _vmap, _vsize = _get_tok(vocab_file, "custom")
        return _CFM(
            transformer=model_cls(**model_cfg, text_num_embeds=_vsize, mel_dim=_ui.n_mel_channels),
            mel_spec_kwargs=dict(
                n_fft=_ui.n_fft, hop_length=_ui.hop_length, win_length=_ui.win_length,
                n_mel_channels=_ui.n_mel_channels, target_sample_rate=_ui.target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
            odeint_kwargs=dict(method=ode_method),
            vocab_char_map=_vmap,
        ).to(device)
    _ui.load_model = _patched_load_model
except ImportError:
    pass

try:
    from transformers.modeling_utils import PreTrainedModel as _PTM
    _PTM._move_missing_keys_from_meta_to_device = lambda self, *a, **k: None
    _PTM.tie_weights = lambda self, *a, **k: None
    _PTM.all_tied_weights_keys = {}
except (ImportError, AttributeError):
    pass

print(f"Python: {sys.version}", flush=True)

import numpy as np
import soundfile as sf
import torch
print(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free={free/1e9:.2f}GB / total={total/1e9:.2f}GB", flush=True)

from transformers import AutoModel
print("transformers imported", flush=True)

DEVICE = "cpu"   # force CPU to rule out GPU OOM
REF_AUDIO = "Audios/story120reference.wav"
REF_TEXT  = "हे गाइस, आज की हमारी कहानी है गरीब किसान और सोने का घड़ा।"
TEXT      = "गाड़ी ने लाहौर का स्टेशन छोड़ा तो एकबारगी मेरा मन काँप उठा।"
OUT       = "test_output.wav"

import sys, traceback

print(f"Loading IndicF5 on {DEVICE} ...")
sys.stdout.flush()

try:
    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False,
    )
    print("from_pretrained OK")
    sys.stdout.flush()
    model = model.to(DEVICE).eval()
    print(f"Model ready on {DEVICE}.")
    sys.stdout.flush()
except Exception:
    print("ERROR during model load:")
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"Synthesizing: {TEXT}")
    sys.stdout.flush()
    with torch.no_grad():
        audio = model(TEXT, ref_audio_path=REF_AUDIO, ref_text=REF_TEXT)
    print(f"Raw audio type={type(audio)}, shape={getattr(audio, 'shape', 'N/A')}")
    sys.stdout.flush()
except Exception:
    print("ERROR during synthesis:")
    traceback.print_exc()
    sys.exit(1)

audio = np.asarray(audio, dtype=np.float32)
if np.max(np.abs(audio)) > 1.5:
    audio = audio / 32768.0

sf.write(OUT, audio, samplerate=24000)
print(f"Done → {OUT}  ({len(audio)/24000:.1f}s)")
