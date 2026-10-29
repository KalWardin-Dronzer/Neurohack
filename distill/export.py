"""Export the trained adapter for serving: fold it into the base weights, convert to GGUF and register with Ollama.

Run in the GPU environment after training (the GPU is not used):

    .venv-train\\Scripts\\python -m distill.export --llama-cpp C:\\path\\to\\llama.cpp

The untuned base model goes through the same steps, so the two can be compared at the same precision. Under the
run directory this writes `student.q8_0.gguf`, `base.q8_0.gguf`, a Modelfile for each and `export.json` with
their hashes; the intermediate safetensors copies are deleted.

Ollama renders prompts with the chat template stored in the GGUF, which is Qwen3.5's own. With thinking off, that
template produces exactly the training format, so clients must send `reasoning_effort: "none"`
(LLM_REASONING_EFFORT=none). Without it, Ollama opens a think block the student never saw in training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import BASE_NAME, RUN_DIR, STUDENT_NAME
from .prompt import END_OF_TURN

GGUF_TYPE = "q8_0"


def modelfile(gguf_name: str) -> str:
    """Greedy decoding with no repeat penalty, since memories legitimately repeat names and keys.

    There is no TEMPLATE: Ollama 0.40 uses the GGUF's chat template even when the Modelfile gives one.
    """
    return "\n".join(
        [
            f"FROM ./{gguf_name}",
            "PARAMETER temperature 0",
            "PARAMETER repeat_penalty 1",
            "PARAMETER num_ctx 8192",
            f'PARAMETER stop "{END_OF_TURN}"',
            "",
        ]
    )


def save_hf(out_dir: Path, base_model: str, revision: str, adapter_dir: Path | None) -> None:
    """The text model as safetensors, with the adapter merged in when one is given."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=torch.bfloat16)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_model, revision=revision).save_pretrained(str(out_dir))


def to_gguf(hf_dir: Path, gguf_path: Path, llama_cpp: Path) -> None:
    script = llama_cpp / "convert_hf_to_gguf.py"
    subprocess.run([sys.executable, str(script), str(hf_dir), "--outtype", GGUF_TYPE, "--outfile", str(gguf_path)], check=True)


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(run_dir: Path, llama_cpp: Path, register: bool) -> dict:
    training = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
    base_model, revision = training["config"]["base_model"], training["config"]["base_revision"]
    llama_cpp_commit = subprocess.run(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest: dict = {"base_model": base_model, "base_revision": revision, "gguf_type": GGUF_TYPE, "llama_cpp_commit": llama_cpp_commit}
    for role, name, adapter in (("student", STUDENT_NAME, run_dir / "adapter"), ("base", BASE_NAME, None)):
        hf_dir, gguf_path = run_dir / f"{role}-hf", run_dir / f"{role}.{GGUF_TYPE}.gguf"
        save_hf(hf_dir, base_model, revision, adapter)
        to_gguf(hf_dir, gguf_path, llama_cpp)
        shutil.rmtree(hf_dir)
        recipe = run_dir / f"Modelfile.{role}"
        recipe.write_text(modelfile(gguf_path.name), encoding="utf-8", newline="\n")
        if register:
            # An absolute path: given a Modelfile it cannot find, `ollama create` silently imports the whole folder.
            subprocess.run(["ollama", "create", name, "-f", str(recipe.resolve())], check=True, cwd=run_dir)
        manifest[role] = {"ollama_model": name, "gguf": gguf_path.name, "gguf_sha256": sha256_of(gguf_path), "bytes": gguf_path.stat().st_size}
        print(f"{role}: {gguf_path} ({gguf_path.stat().st_size / 2**20:.0f} MiB)" + (f", registered as {name}" if register else ""))
    (run_dir / "export.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m distill.export", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--llama-cpp", type=Path, required=True, help="a llama.cpp checkout, for convert_hf_to_gguf.py")
    parser.add_argument("--no-ollama", action="store_true", help="write the GGUF files and Modelfiles without registering them")
    args = parser.parse_args(argv)
    export(args.run_dir, args.llama_cpp, register=not args.no_ollama)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
