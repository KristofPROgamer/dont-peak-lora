"""
cipher.py — Semantic Prompt Cipher  (permutation edition)
==========================================================
Random alphabet permutation + word rotation.

Why this beats the original Caesar shift
-----------------------------------------
Caesar has 25 possible keys — any LLM or human can brute-force all of them
in under a second.  A random permutation has 26! ≈ 4×10²⁶ possible keys.
Short prompts can't be cracked by frequency analysis, and ChatGPT has nothing
to anchor on without the exact key string.  The LoRA learns it just as easily
because it's still a fixed, consistent letter-to-letter mapping.

Key format
----------
A 26-character string that is a shuffled version of a-z, e.g.:
    qmjztgfkpwlsboxncryevhiadu
Generate one with:  python cipher.py --generate-key
Store it somewhere safe — it must match the KEY value used in the notebook.

Usage
-----
  # Generate a fresh key (do this once, save the output)
  python cipher.py --generate-key

  # Encode a prompt
  python cipher.py "a cat on a mat" --key qmjztgfkpwlsboxncryevhiadu

  # Decode it back
  python cipher.py "q ejq ox q ebq" --key qmjztgfkpwlsboxncryevhiadu --decrypt

  # Interactive REPL
  python cipher.py --interactive --key qmjztgfkpwlsboxncryevhiadu

  # Demo with a given key
  python cipher.py --demo --key qmjztgfkpwlsboxncryevhiadu
"""

import sys
import random
import argparse
import os

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# ── Key utilities ─────────────────────────────────────────────────────────────

def generate_key(seed: int = None) -> str:
    """Return a random 26-char permutation of a-z.  Optionally seeded."""
    rng = random.Random(seed)
    perm = list(ALPHABET)
    rng.shuffle(perm)
    return "".join(perm)


def validate_key(key: str) -> str:
    """Raise ValueError if key is not a valid permutation of a-z."""
    key = key.strip().lower()
    if sorted(key) != list(ALPHABET):
        raise ValueError(
            f"Key must be a permutation of all 26 lowercase letters.\n"
            f"Got: {key!r}\n"
            f"Tip: generate one with --generate-key"
        )
    return key


# ── Cipher core ───────────────────────────────────────────────────────────────

def _make_tables(key: str):
    """Build str.translate tables for both directions."""
    enc = str.maketrans(ALPHABET + ALPHABET.upper(),
                        key      + key.upper())
    dec = str.maketrans(key      + key.upper(),
                        ALPHABET + ALPHABET.upper())
    return enc, dec


def _rotate(word: str) -> str:
    """Move last character to front (encrypt step 2)."""
    return (word[-1] + word[:-1]) if len(word) > 1 else word


def _unrotate(word: str) -> str:
    """Move first character to back (decrypt step 1)."""
    return (word[1:] + word[0]) if len(word) > 1 else word


def encrypt(text: str, key: str) -> str:
    """
    Encrypt a full prompt.
    For each word: (1) substitute every letter via the permutation key,
                   (2) rotate: last character → front.
    Non-alpha characters (digits, punctuation) pass through untouched.
    Whitespace structure is preserved.
    """
    enc, _ = _make_tables(key)
    return " ".join(_rotate(w.translate(enc)) for w in text.split())


def decrypt(text: str, key: str) -> str:
    """Reverse of encrypt: (1) unrotate, (2) substitute back."""
    _, dec = _make_tables(key)
    return " ".join(_unrotate(w).translate(dec) for w in text.split())


# ── CLI helpers ───────────────────────────────────────────────────────────────

_DEMO_PROMPTS = [
    "a cat on a mat",
    "a beautiful sunset over the ocean",
    "futuristic city at night with neon lights",
    "an astronaut riding a horse on Mars",
    "portrait of a woman in renaissance style",
]


def _demo(key: str) -> None:
    w = 60
    print(f"\n{'─'*w}")
    print(f"  Cipher demo  (key = {key})")
    print(f"{'─'*w}")
    for p in _DEMO_PROMPTS:
        c = encrypt(p, key)
        d = decrypt(c, key)
        ok = "✓" if d == p else "✗ ROUND-TRIP FAILED"
        print(f"  plain : {p}")
        print(f"  cipher: {c}")
        print(f"  back  : {d}  {ok}")
        print()


def _interactive(key: str) -> None:
    print(f"\nInteractive cipher  —  key: {key}")
    print("Type a prompt (plain or cipher) and press Enter.  Ctrl-C to quit.\n")
    try:
        while True:
            raw = input("plain > ").strip()
            if not raw:
                continue
            c = encrypt(raw, key)
            d = decrypt(c, key)
            print(f"cipher: {c}")
            print(f"back  : {d}  {'✓' if d == raw else '✗'}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Allow key via env var so you don't have to retype it every call:
    #   export CIPHER_KEY=qmjztgfkpwlsboxncryevhiadu
    env_key = os.environ.get("CIPHER_KEY", "")

    parser = argparse.ArgumentParser(
        description="Semantic prompt cipher — random permutation + word rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text", nargs="?",
                        help="Prompt to encode or decode")
    parser.add_argument("--key", default=env_key,
                        help="26-char permutation key.  Can also be set via "
                             "the CIPHER_KEY env variable.")
    parser.add_argument("--decrypt",      action="store_true",
                        help="Decrypt instead of encrypt")
    parser.add_argument("--interactive",  action="store_true",
                        help="Interactive REPL — type prompts one at a time")
    parser.add_argument("--demo",         action="store_true",
                        help="Show built-in examples and exit")
    parser.add_argument("--generate-key", action="store_true",
                        help="Print a fresh random key and exit")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Optional integer seed for reproducible key generation")
    args = parser.parse_args()

    # ── Key generation doesn't need an existing key ───────────────────────────
    if args.generate_key:
        k = generate_key(args.seed)
        print(k)
        print("\nStore this somewhere safe — add it to your notebook as KEY = "
              f'"{k}"', file=sys.stderr)
        return

    # ── Everything else requires a key ────────────────────────────────────────
    if not args.key:
        parser.error(
            "No key supplied.  Either pass --key <26chars> or set the "
            "CIPHER_KEY environment variable.\n"
            "To generate a fresh key run:  python cipher.py --generate-key"
        )

    key = validate_key(args.key)

    if args.demo:
        _demo(key)
        return

    if args.interactive:
        _interactive(key)
        return

    if args.text is None:
        _demo(key)
        return

    fn     = decrypt if args.decrypt else encrypt
    result = fn(args.text, key)
    print(result)


if __name__ == "__main__":
    import argparse
    import sys

    # Hardcoded default key
    DEFAULT_KEY = "cixmpuqbstreyofwlkzhgnavdj"

    parser = argparse.ArgumentParser(description="Semantic Prompt Cipher")
    parser.add_argument("text", nargs="?", default=None,
                        help="Text to encrypt/decrypt (optional)")
    parser.add_argument("--key", default=DEFAULT_KEY,
                        help="26-char substitution key")
    parser.add_argument("--decrypt",      action="store_true",
                        help="Decrypt the text instead of encrypting")
    parser.add_argument("--interactive",  action="store_true",
                        help="Interactive REPL — type prompts one at a time")
    parser.add_argument("--demo",         action="store_true",
                        help="Show built-in examples and exit")
    parser.add_argument("--generate-key", action="store_true",
                        help="Print a fresh random key and exit")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Optional integer seed for reproducible key generation")
    args = parser.parse_args()

    # ── Key generation ──────────────────────────────────────────────
    if args.generate_key:
        k = generate_key(args.seed)
        print(k)
        print("\nStore this somewhere safe — add it to your notebook as KEY = "
              f'"{k}"', file=sys.stderr)
        sys.exit(0)

    # ── Validation ──────────────────────────────────────────────────
    key = validate_key(args.key)

    if args.demo:
        _demo(key)
        sys.exit(0)

    # ── Single-shot mode (if you pass text directly) ────────────────
    if args.text is not None:
        fn = decrypt if args.decrypt else encrypt
        print(fn(args.text, key))
        sys.exit(0)

    # ── Default behavior: Interactive Loop ──────────────────────────
    # If no text and no specific flags are passed, just boot the loop
    print(f"Loaded Default Key: {DEFAULT_KEY}")
    _interactive(key)