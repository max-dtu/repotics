# Models for EV3 Robot Control

This document analyzes the trade-offs of local Vision-Language Models (VLMs) versus cloud-based models for real-time control of the EV3 Mindstorms robot on an Intel MacBook Pro.

---

## 1. Local Vision-Language Models (VLMs)

When running models locally, performance is heavily constrained by the host GPU's VRAM. Exceeding VRAM capacity forces PyTorch/Accelerate to offload weights to the CPU, increasing latency from ~1 second to ~165 seconds per step.

| Model | Size | Minimum VRAM (FP16) | Target Hardware | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-72B-Instruct** | 72B | ~144 GB | Mac Studio (128GB+ RAM) or Multi-GPU Server | Currently the absolute best open-weights VLM. Matches GPT-4o on many visual benchmarks. |
| **Llama-3.2-90B-Vision-Instruct** | 90B | ~180 GB | Enterprise Server | Extremely strong reasoning and general multimodal capability. |
| **Llama-3.2-11B-Vision-Instruct** | 11B | ~22 GB | Mac Studio or RTX 3090/4090 (24GB) | Great mid-sized option for general reasoning and layout parsing. |
| **Qwen2.5-VL-7B-Instruct** | 7B | ~14 GB | Mac (16GB+ RAM) or RTX 3080/4070 (16GB) | The gold standard for mid-sized VLM task execution, spatial reasoning, and agentic control. |
| **Qwen2.5-VL-3B-Instruct** | 3B | ~6 GB | Mid-range laptops (like your 8GB AMD MBP) | The frontier for **ultra-lightweight** models. Excellent balance of JSON formatting, visual attention, and local performance. |

---

## 2. Cloud-Based Vision-Language Models

Using a cloud API (e.g., Gemini 2.0 Flash or GPT-4o-mini) offloads all heavy computation to external servers.

### Pros
* **Low Latency (Fast Control Loop)**: 
  * Running a 3B model locally on an Intel MBP takes 1–2 seconds.
  * Querying a fast cloud VLM (like **Gemini 2.0 Flash** or **GPT-4o-mini**) via API typically takes **200ms to 400ms** round-trip. This easily satisfies a `1 Hz` to `2 Hz` control loop target.
* **No Hardware Bottlenecks**: 
  * No GPU thermal throttling, no CPU thread starvation, and no memory swapping/paging lag on the MacBook.
* **Superior Spatial & Logical Reasoning**:
  * Large cloud models are extremely robust at visual grounding (returning precise bounding boxes or relative coordinates) and consistently adhere to JSON response formats.
* **Lighter Codebase**:
  * Eliminates the need for heavy local dependencies like `torch`, `transformers`, and `accelerate`.

### Cons
* **Internet Dependency**: The MacBook must have an active internet connection. If the EV3 and MacBook are connected via a local offline Wi-Fi router or peer-to-peer connection without external internet access, cloud queries will fail.
* **Operational Cost**: API queries cost money (though models like Gemini 2.0 Flash or GPT-4o-mini cost less than $0.005 per image request).
* **Privacy**: Camera frames are sent to a cloud provider.

---

## Summary Verdict
* **For offline/local setups**: Use **`Qwen2.5-VL-3B-Instruct`** as it fits entirely within the 8 GB VRAM limit of the AMD GPU on the Intel MacBook Pro.
* **For best responsiveness & accuracy**: Use **Gemini 2.0 Flash** or **GPT-4o-mini** via API, which delivers sub-second latency and excellent spatial reasoning.
