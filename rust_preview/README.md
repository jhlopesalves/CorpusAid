# corpus_preview Extension

This crate exposes two helpers for CorpusAid:

- `load_preview(path, limit)` streams a fixed-size preview of a document without blocking the GUI.
- `scan_directory(path)` walks the directory tree and returns the list of `.txt` files quickly.

## Building

```
python -m pip install maturin
maturin develop --release
```

The command builds the native extension and installs it into the current Python environment. On
Windows you may need to have the Rust toolchain (rustup) and a suitable MSVC build environment
installed.

If the extension is unavailable at runtime, CorpusAid automatically falls back to the pure-Python
preview loader.
