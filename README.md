# revolver

Automatically picks a free GPU so you don't have to. Use it as `CUDA_VISIBLE_DEVICES=$(revolver)` — if all GPUs are busy, it blocks with a live status line until one frees up.

Install with `cargo install --path .`
