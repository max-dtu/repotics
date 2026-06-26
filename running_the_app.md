# Running the EV3 Robot VLM Loops

This guide explains how to set up the project, start the main GUI application, and run both local and cloud-based Vision-Language Model (VLM) control loops for the EV3 robot.

---

## 1. Setup and Virtual Environment

First, prepare your environment and install core requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the main dashboard application:
```bash
python main.py
```

---

## 2. Running Local VLM Loops (In-Memory Inference)

Local inference runs the model weights entirely on your MacBook. This is best for offline settings but requires significant CPU/GPU resources.

### Installation
Install PyTorch and the Hugging Face Transformers vision dependencies:
```bash
pip install -U transformers==4.47.1 accelerate torch qwen-vl-utils torchvision==0.17.2 "numpy<2"
```

### Execution
1. Run a dry run first to download the default model weights (`Qwen/Qwen2.5-VL-3B-Instruct`) and verify setup:
   ```bash
   python local_ev3_loop.py --dry-run --max-steps 1
   ```

2. Run on the EV3 TCP server (use `--device cpu` on Intel Macs, or `--device mps` on Apple Silicon):
   * **CPU Mode (Intel Macs):**
     ```bash
     python local_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1.0 --device cpu
     ```
   * **MPS GPU Mode (Apple Silicon):**
     ```bash
     python local_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1.0 --device mps --torch-dtype float16
     ```

*Note: Keep `--hz` around `1.0` and width/height around `320x240` to maintain responsive local generation performance.*

---

## 3. Running Cloud VLM Loops (Gemini & OpenAI)

Cloud loops query models via external API endpoints (e.g. Gemini 2.0 Flash or GPT-4o-mini). This offloads processing, yielding ultra-low latency (200ms - 400ms) and superior spatial reasoning.

### Environment Setup
Export your API keys in your terminal:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

### Execution
* **Using Gemini 2.0 Flash (Recommended):**
  ```bash
  python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 1.0 --model gemini-2.0-flash
  ```
  *(Note: If using the Gemini Free Tier, consider running with `--hz 0.25` to stay within the 15 requests-per-minute rate limit).*
  
* **Using GPT-4o-mini:**
  ```bash
  python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 1.0 --model gpt-4o-mini
  ```

---

## 4. Using the Model Harness for Local Servers

The `cloud_ev3_loop.py` script also functions as a **Model Harness**. Instead of loading model weights in PyTorch, it connects to local inference servers (Ollama, llama.cpp, LM Studio) that expose OpenAI-compatible endpoints.

This offers:
* **Zero startup lag:** No loading of large models in Python.
* **Fast generation:** Leverages quantized GGUF weights optimized for macOS Metal.

### Running with a Local Server (e.g., Ollama)
Start your local model (e.g. `qwen2.5-vl:7b`) in Ollama, and direct requests to its local API port using `--api-base`:
```bash
python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 1.0 --model qwen2.5-vl:7b --api-base http://localhost:11434/v1
```

### Running with llama.cpp (llama-server)
For local performance, download and serve quantized GGUF weights via `llama-server`:

1. **Download the model & projector files:**
   ```bash
   huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --local-dir /Users/kkhomefolder/llama.cpp/models --local-dir-use-symlinks False
   huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-GGUF mmproj-F16.gguf --local-dir /Users/kkhomefolder/llama.cpp/models --local-dir-use-symlinks False
   ```

2. **Launch llama-server:**
   ```bash
   /usr/local/bin/llama-server \
     -m /Users/kkhomefolder/llama.cpp/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
     --mmproj /Users/kkhomefolder/llama.cpp/models/mmproj-F16.gguf \
     --port 8080 \
     -c 2048 \
     --host 0.0.0.0
   ```

3. **Run the model harness loop:**
   Direct requests to the local `llama-server` port:
   ```bash
   python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 1.0 --model qwen2.5-vl-7b --api-base http://localhost:8080/v1
   ```

### Intel CPU vs. AMD GPU Inference (Intel Macs)

Since your system is an **Intel Core i9 MacBook Pro 16** with a discrete **AMD Radeon Pro 5500M GPU (8 GB VRAM)**, here is how you should configure inference:

#### 1. Running on CPU (Intel Core i9) - Recommended for Stability
By default, the pre-built `llama.cpp` binaries for Intel macOS will run on the CPU using Apple's Accelerate framework (`BLAS`).
* **Pros:** 100% stable, no compilation required, plenty of room in your 32 GB RAM.
* **Cons:** Slower image encoding (~60 seconds per frame).
* **Launch Command:** Use the standard launch command shown in step 2 above.

#### 2. Running on AMD GPU (Radeon Pro 5500M) - For Performance
Your AMD GPU has **8 GB VRAM**, which is sufficient to fit the model (~4.68 GB) and the projector (~1.30 GB).
* **Note on Metal Support:** The standard `Metal` backend in `llama.cpp` is optimized specifically for Apple Silicon (M-series unified memory) and often fails or runs slowly on Intel Macs with discrete GPUs.
* **Solution:** To use your AMD GPU, you should compile `llama.cpp` with the **Vulkan** backend (which sits on top of Metal via MoltenVK on macOS):

##### Compile llama.cpp with Vulkan support:
```bash
# Install Vulkan and compilation dependencies (including vulkan-loader)
brew install cmake git libomp vulkan-headers vulkan-loader glslang molten-vk shaderc

# Clone and build llama.cpp with Vulkan enabled (and Metal disabled)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=OFF -DGGML_VULKAN=1 \
  -DVulkan_INCLUDE_DIR=$(brew --prefix)/opt/vulkan-headers/include \
  -DVulkan_LIBRARY=$(brew --prefix)/lib/libvulkan.dylib \
  -DOpenMP_ROOT=$(brew --prefix)/opt/libomp \
  -DVulkan_GLSLC_EXECUTABLE=$(brew --prefix)/opt/shaderc/bin/glslc \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=$(brew --prefix)/opt/glslang/bin/glslangValidator

cmake --build build --config Release
```

##### Launch llama-server offloading to AMD GPU:
Use the compiled binary and specify `-ngl` to offload layers to the GPU:
```bash
./build/bin/llama-server \
  -m /Users/kkhomefolder/llama.cpp/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj /Users/kkhomefolder/llama.cpp/models/mmproj-F16.gguf \
  --port 8080 \
  -c 2048 \
  -ngl 28 \
  --host 0.0.0.0
```

##### GPU Offloading Guidelines & Recommendations (`-ngl`)

Since the **AMD Radeon Pro 5500M** has **8 GB VRAM**, memory allocation is tight. Exceeding 8 GB VRAM will trigger macOS swap memory, resulting in severe lag and execution speeds slower than CPU-only mode.

* **Layer Count:** The Qwen2.5-VL-7B model contains exactly **28 layers**.
* **Offloading Recommendations:**
  * **Full Offloading (`-ngl 28`):** Offloads all 28 language layers to the GPU. The total VRAM usage (Model ~4.68 GB + Projector ~1.30 GB + KV cache + display output overhead) is around **~6.1 GB**. This is recommended if you do not have heavy graphical applications running in the background.
  * **Partial Offloading (`-ngl 15` to `-ngl 20`):** Splits processing between the GPU and CPU. Use this if you experience UI stuttering or memory warnings.
  * **Disabling Projector Offload (`--no-mmproj-offload`):** If you run out of memory during the initial image-encoding step, keep the large 1.30 GB projector weights on the CPU and offload only language layers:
    ```bash
    ./build/bin/llama-server -m ... --mmproj ... -ngl 28 --no-mmproj-offload
    ```


### Auto-Correction Validation Loop
The harness includes a compilation-style feedback loop that validates VLM decisions and forces corrections in real-time (up to 3 retries):
1. **Level 1 (JSON Syntax):** Rejects invalid JSON structures.
2. **Level 2 (Physical constraints):** Rejects illegal commands (e.g. trying to `open_gripper` when the gripper is already open).
3. **Level 3 (Logical Consistency):** Detects contradictions (e.g., evaluating progress as `degraded` but choosing to repeat the exact same command). 

If a check fails, the harness automatically re-prompts the model with the compiler error to force self-correction.