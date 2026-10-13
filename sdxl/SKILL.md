---
name: python-script
description: "Use this skill for ANY task involving Python files: writing a new .py script or module, creating a Python package or CLI tool, refactoring or cleaning up existing Python code, fixing bugs in Python scripts, adding new functionality to existing Python modules, or expanding a Python project with new files. Trigger this skill whenever the user says 'write a script', 'create a Python file', 'fix my script', 'improve/refactor this code', 'add a function to', 'expand this module', or pastes Python code they want improved. Also trigger when the user is working on ML training scripts, data pipelines, automation scripts, or any Python-based tooling — even if they don't say 'Python' explicitly but the context is clear. Prefer this skill over ad-hoc code generation to ensure proper file output, code quality, and verification."
---

# Python Script Skill

Covers creating, improving, and expanding Python scripts and files with production-quality output.

---

## Step 0: Mode — Decide Before Acting

| Signal | Mode |
|---|---|
| No existing file, user describes what they want | **Create** |
| User pastes/uploads code that needs fixes, cleanup, or optimization | **Improve** |
| User wants new functionality added to existing code | **Expand** |

If files are provided (uploaded or referenced), `view` them before writing a single line.

---

## Create Mode

**Scope decision first:**
- Single file → write directly to `/mnt/user-data/outputs/<name>.py`
- Multi-file (package, CLI with subcommands, >~250 lines) → build in `/home/claude/<name>/`, then copy final files to `/mnt/user-data/outputs/`

**Always include:**
- Module docstring
- Type hints on all public function signatures
- Docstrings on classes and non-trivial functions
- `if __name__ == "__main__":` guard on runnable scripts
- Explicit error handling for I/O, parsing, and network calls

**Dependencies:** if third-party packages are needed, create `requirements.txt` alongside the script.

**Verify:** run via `bash_tool` with a quick smoke test when feasible (no GPU, no network, no hardware required). Fix errors before presenting.

---

## Improve Mode

1. Read the full file — don't skim.
2. Categorize issues before touching anything:
   - **Bugs**: wrong logic, uncaught exceptions, type mismatches, edge cases
   - **Performance**: O(n²) loops, redundant computation, unnecessary copies
   - **Readability**: poor naming, magic numbers, deeply nested logic, dead code
   - **Style**: PEP 8 violations, inconsistent conventions, missing type hints
3. Use `str_replace` for targeted edits on large files. Only rewrite the whole file if it's small or the user asks.
4. Preserve the existing interface unless explicitly told to change it. If a signature must change, note it clearly.
5. Summarize changes as a brief bulleted list after the edit — one item per logical change, not per line.

---

## Expand Mode

1. Read all related files before adding anything. Note: naming conventions, import style, error handling patterns, class/function organization.
2. Match the existing style exactly — don't impose preferences.
3. Place new code near related existing code, not appended at the bottom.
4. Add, don't modify existing interfaces unless asked. New functions alongside old ones.
5. Update imports at the top of the file.
6. If adding a new file to a package, update `__init__.py` if relevant.

---

## Code Quality Standards

Apply unconditionally:

```
Type hints        Required on all public functions
Docstrings        Required on classes; one-liner minimum on non-trivial functions
Exceptions        Specific types with meaningful messages; never bare except:
Constants         UPPER_SNAKE_CASE at module level, not inline magic numbers
File I/O          pathlib.Path over os.path; context managers (with open(...))
Logging           logging module for scripts meant to be reused; print() OK for one-offs
Formatting        Black-compatible style (88-char lines, trailing commas in multi-line)
```

Don't over-engineer. A 30-line script doesn't need ABCs, factories, or plugin registries.

---

## Useful Patterns

**Argument parsing (CLI scripts):**
```python
import argparse

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="...")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output"))
    return p.parse_args()
```

**Logging setup:**
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

**Dataclass for structured data** (prefer over plain dicts when shape is known):
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    lr: float = 1e-4
    epochs: int = 5
    output_dir: Path = Path("checkpoints")
```

**Safe JSON load with fallback:**
```python
def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot load config from {path}: {e}") from e
```

---

## ML / Training Script Context

For machine learning scripts (PyTorch, diffusers, PEFT, Accelerate, etc.):

- Separate concerns: data loading, model definition, training loop, checkpointing — each in its own function or section with a clear header comment
- Log loss and metrics at every step (or every N steps), not just epoch end
- Checkpoint early; don't wait for epoch boundary. Include step number in filename: `adapter_epoch{epoch}_step{step}.pt`
- `torch.no_grad()` belongs only in validation/inference; never wrap the forward pass inside training
- Use `tqdm` for progress bars on long loops
- For multi-GPU (`DataParallel` / `Accelerate`): always unwrap the model before saving state dict — `model.module.state_dict()` or `accelerator.unwrap_model(model)`
- LR scheduler: call `scheduler.step()` after `optimizer.step()`, never before
- For resuming from checkpoint: restore optimizer state, scheduler state, and scaler state (for AMP) — not just model weights

---

## Output

1. Save final file(s) to `/mnt/user-data/outputs/`
2. Call `present_files` with the output path(s)
3. Add 1–2 sentences max: what it does, how to run it (if non-obvious)

Don't narrate the code. Trust the user to read it.
