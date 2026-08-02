# SGLang model server

The SGLang server runs independently from the zero-shot pipeline. This package
only turns Hydra configuration into an SGLang command; it does not install
SGLang, download models, manage SSH, or restart server processes.

Preview the default Qwen 3.6 command without starting a server:

```bash
python -m zeroshot.server.serve_model dry_run=true
```

On the GPU host, activate the conda environment, install `sglang`, and then launch it:

```bash
conda activate drawing2cad
pip3 install sglang
python -m zeroshot.server.serve_model
```

Choose another model profile or override hardware-dependent settings with
Hydra:

```bash
python -m zeroshot.server.serve_model \
  model=qwen3_6_27b_fp8 \
  server.tensor_parallel_size=1 \
  server.context_length=65536
```

The default host is `127.0.0.1`. From the laptop, expose it through an SSH
tunnel rather than binding an unauthenticated endpoint publicly:

```bash
ssh -N -L 30000:localhost:30000 <gpu-host>
```

Then select the existing OpenAI-compatible pipeline client:

```bash
ZEROSHOT_LIVE_MODEL_CONFIG=qwen3_6_sglang \
  python -m pytest tests/zeroshot/live/test_openai_compatible.py

python -m zeroshot.run_pipeline model=qwen3_6_sglang
```

If the server uses the 27B profile, set the client-visible model name too:

```bash
export SGLANG_MODEL=Qwen/Qwen3.6-27B-FP8
```
