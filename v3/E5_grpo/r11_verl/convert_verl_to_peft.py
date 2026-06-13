"""Convert verl FSDP ckpt → PEFT LoRA adapter.

verl key:  base_model.model.model.layers.X.q_proj.lora_A.default.weight
PEFT key:  base_model.model.model.layers.X.q_proj.lora_A.weight
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file


# Self-contained adapter_config — matches our R11 training (r=64, α=32, all-linear)
DEFAULT_ADAPTER_CFG = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": "/mnt/d/fine-tuning/models/gemma-2-2b-it",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 32,
    "lora_bias": False,
    "lora_dropout": 0.0,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 64,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}


def convert(verl_ckpt_dir: Path, out_dir: Path, ref_adapter_cfg: Path = None, base_model: str = None):
    actor = verl_ckpt_dir / "actor"
    pt_files = list(actor.glob("model_world_size_*_rank_*.pt*"))
    assert len(pt_files) == 1, f"expected 1 model.pt, got {len(pt_files)}: {pt_files}"
    pt = pt_files[0]

    print(f"loading {pt} ({pt.stat().st_size / 1e9:.1f} GB)…")
    sd = torch.load(pt, map_location="cpu", weights_only=False)
    print(f"  {len(sd)} keys total")

    lora_sd = {}
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        # FSDP2 saves DTensor — unwrap to local tensor (mesh is 1-device for our run)
        if hasattr(v, "to_local"):
            v = v.to_local()
        new_k = k.replace(".default.", ".")
        lora_sd[new_k] = v.detach().contiguous()
    print(f"  extracted {len(lora_sd)} LoRA keys")
    assert len(lora_sd) > 0, "no LoRA keys found — was lora_rank set during training?"

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(lora_sd, out_dir / "adapter_model.safetensors")
    print(f"  wrote {out_dir / 'adapter_model.safetensors'}")

    if ref_adapter_cfg and ref_adapter_cfg.exists():
        cfg = json.loads(ref_adapter_cfg.read_text())
    else:
        cfg = dict(DEFAULT_ADAPTER_CFG)
    if base_model:
        cfg["base_model_name_or_path"] = base_model
    cfg["r"] = 64
    cfg["lora_alpha"] = 32
    cfg["target_modules"] = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    cfg["inference_mode"] = True
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {out_dir / 'adapter_config.json'}")

    hf_dir = actor / "huggingface"
    if hf_dir.exists():
        for name in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "chat_template.jinja"]:
            src = hf_dir / name
            if src.exists():
                shutil.copy(src, out_dir / name)
        print(f"  copied tokenizer files from {hf_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to global_step_N/")
    p.add_argument("--out", required=True, help="output PEFT adapter dir")
    p.add_argument("--ref-cfg", default=None, help="optional ref adapter_config.json (else use built-in defaults)")
    p.add_argument("--base-model", default=None, help="override base_model_name_or_path in cfg")
    a = p.parse_args()
    ref = Path(a.ref_cfg) if a.ref_cfg else None
    convert(Path(a.ckpt), Path(a.out), ref, a.base_model)
