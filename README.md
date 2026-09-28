# TIEG.SystemOne.LLamaSharp

[![NuGet Version](https://img.shields.io/nuget/v/TIEG.SystemOne.LLamaSharp.svg?style=flat-square)](https://www.nuget.org/packages/TIEG.SystemOne.LLamaSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

A high-performance, deterministic routing engine for local AI models, built for `LLamaSharp`. Developed by **The Intelligent Enterprise Group (TIEG)**.

This library implements a "System One" architectural pattern. Instead of generating text autoregressively (which takes seconds), it extracts raw logits from a single parallel batch pass to determine the probability of specific categories. 

This allows specialized smaller models to execute deterministic classification in **< 100ms** while saving your VRAM for larger fallback generative models.

## Features
* **Instant Execution:** Bypass token-by-token generation for sub-100ms classification.
* **Universal Thresholds:** Uses Margin and Dominance Ratios to evaluate the model's internal hesitation, ensuring mathematical safety across 2-option and 10-option questions.
* **VRAM Safety:** Includes thread-locking and explicit `Unload()` memory management to prevent out-of-memory exceptions when hot-swapping models.

## Model Compatibility & Calibration
While this engine is built on standard `llama.cpp` logit extraction and technically works natively with any instruction-tuned causal language model (including Llama 3, Qwen, Mistral, and Phi), **it is heavily optimized for use with purpose-built routing models.**

**Recommended Model:** [Plumb-4B](https://huggingface.co/TheIntelligentEnterpriseGroup/Plumb-4B) is our specialized System One routing model, explicitly trained and calibrated for this engine.

**Setting the Calibration Temperature:**
The constructor accepts a `calibrationTemperature` parameter which adjusts the strictness of the Softmax probability distribution:
* **`2.07f` (Plumb Default):** The default temperature is specifically mathematically calibrated for the **Plumb-4B** routing model to align its raw logits with true statistical probability.
* **`1.0f` (Standard):** Use this baseline if you are experimenting with general-purpose models like Qwen 2.5 or Llama 3 (though they are not optimized for strict triage tasks).

## Quick Start

### 1. Installation
Install the core routing engine alongside the `LLamaSharp` backend that matches your hardware (e.g., CUDA 12 for NVIDIA GPUs):

```bash
dotnet add package TIEG.SystemOne.LLamaSharp
dotnet add package LLamaSharp.Backend.Cuda12
```

### 2. Usage

```csharp
using TIEG.SystemOne.LLamaSharp;

// Initialize the engine (set calibrationTemperature: 1.0f if not using Plumb-4B)
using var engine = new DecisionEngine("path/to/model.gguf");

// Execute a classification in < 100ms
var result = engine.Decide(
    systemContext: "Determine the risk level of the transaction.",
    inputData: "Transaction: $12 Netflix subscription.",
    options: new[] { "High Risk", "Medium Risk", "Low Risk" }
);

if (result.CanAutomate(minMargin: 0.20f, minDominanceRatio: 3.0f))
{
    Console.WriteLine($"Automated: {result.SelectedOption} (Margin: {result.Margin:P1})");
}
else
{
    engine.Unload(); // Free VRAM for the fallback model
    Console.WriteLine("Routing to heavy fallback model...");
    // ... route to a heavy fallback model
}
```

## Running the Example Project

A complete, runnable tiered-routing pipeline demonstration is located in [`examples/TieredRoutingDemo`](examples/TieredRoutingDemo):

```bash
cd examples/TieredRoutingDemo
dotnet run -- path/to/primary-model.gguf path/to/fallback-model.gguf
```
