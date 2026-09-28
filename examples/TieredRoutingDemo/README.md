# Tiered Routing Demonstration

This is a complete, runnable console application demonstrating how to use `TIEG.SystemOne.LLamaSharp` to route decisions based on a model's internal confidence metrics.

## Hardware Configuration
By default, this project is configured to use the **CPU backend** (`LLamaSharp.Backend.Cpu`) so it can compile and run on any machine. 

To enable hardware acceleration, edit `TieredRoutingDemo.csproj` and replace the CPU backend with the backend that matches your system (e.g., `LLamaSharp.Backend.Cuda12`).

## Required Models
To run this demo, you need two standard GGUF models:

1. **Primary Routing Model (Fast):** [Plumb-4B-v5 (Q8_0)](https://huggingface.co/TheIntelligentEnterpriseGroup/Plumb-4B). This is our purpose-built System One routing model, calibrated to output precise statistical probabilities. *(Note: While the engine API supports standard models like Qwen or Llama here if you change the calibration temperature to `1.0f`, they are not optimized for this specific triage task).*
2. **Fallback Generative Model (Heavy):** Any standard instruction model (e.g., Llama-3.2-8B-Instruct or Qwen-2.5-7B-Instruct).

## Running the Demo

Pass the absolute or relative paths to your two downloaded models as command-line arguments:

```bash
dotnet run -- "C:\models\plumb-4b-v5-Q8_0.gguf" "C:\models\qwen2.5-7b-instruct.gguf"

```

The application will process two transactions:

1. **A routine transaction** (Automated instantly by the fast System One model).
2. **A borderline transaction** (The fast model hesitates; the engine safely unloads it from VRAM and routes the context to the heavy fallback model).
