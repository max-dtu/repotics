# How to use the app

## Starting the app

### Mac/Linux
```bash
python -m venv .venv
source .venv/bin/activate
python main.py
```

## Qwen2-VL EV3 loop

Install the vision-language model dependencies:

```bash
pip install -U transformers==4.47.1 accelerate torch qwen-vl-utils torchvision==0.17.2 "numpy<2"
```

Run one dry loop first so the model can download and you can inspect the command:

```bash
python qwen_ev3_loop.py --dry-run --max-steps 1
```

Then point it at the EV3 TCP server:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1
```

On MPS, the script loads Qwen with `float16` by default because MPS does not
support `bfloat16`. On CPU, it uses `float32` by default. For Intel Macs, start
with the most reliable path:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device cpu
```

If your PyTorch build exposes the AMD GPU through MPS, you can try:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device mps --torch-dtype float16
```

The prompt is intentionally strict: the model output is reduced to one of
`left`, `right`, `forward`, or `backward`. Keep `--hz` around `1` or `2`, and
leave the image size near `320x240` unless you enjoy lag with extra garnish.
