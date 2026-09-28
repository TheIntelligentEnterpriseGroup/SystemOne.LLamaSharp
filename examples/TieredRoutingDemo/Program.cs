using System.Diagnostics;
using System.Text;
using LLama;
using LLama.Common;
using LLama.Sampling;
using Microsoft.Extensions.Logging;
using TIEG.SystemOne.LLamaSharp;

namespace TieredRoutingDemo
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            // Configure model paths via CLI arguments or default relative locations
            string primaryModelPath = args.Length > 0 ? args[0] : @"models/plumb-4b-v5-Q8_0.gguf";
            string fallbackModelPath = args.Length > 1 ? args[1] : @"models/qwen2.5-7b-instruct.gguf";

            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder
                    .AddSimpleConsole(options =>
                    {
                        options.IncludeScopes = false;
                        options.SingleLine = true;
                        options.TimestampFormat = "[HH:mm:ss] ";
                    })
                    .SetMinimumLevel(LogLevel.Information);
            });

            var logger = loggerFactory.CreateLogger<Program>();
            var engineLogger = loggerFactory.CreateLogger<DecisionEngine>();

            if (!File.Exists(primaryModelPath))
            {
                logger.LogError("Primary decision model not found at '{Path}'.", primaryModelPath);
                logger.LogInformation("Usage: dotnet run -- <PathToPrimaryModel.gguf> [PathToFallbackModel.gguf]");
                return;
            }

            logger.LogInformation("Initializing Tier-1 DecisionEngine with: {ModelPath}", primaryModelPath);

            using var decisionEngine = new DecisionEngine(
                primaryModelPath,
                calibrationTemperature: 2.07f, // Calibrated for Plumb-4B (use 1.0f for standard models)
                idleMinutes: 1,
                logger: engineLogger);

            var options = new List<string> { "High Risk", "Medium Risk", "Low Risk" };

            // Scenario 1: Unambiguous input -> Passes confidence & margin thresholds
            logger.LogInformation("\n=== Processing Transaction 1 (Routine / Unambiguous) ===");
            string clearTx = "Transaction: $12 payment for monthly Netflix subscription from home IP.";
            var result1 = decisionEngine.Decide(
                systemContext: "Determine the risk level of the transaction.",
                inputData: clearTx,
                options: options
            );
            await EvaluateDecisionAsync(clearTx, options, result1, decisionEngine, fallbackModelPath, logger);

            // Scenario 2: Borderline input -> Fails threshold, unloads VRAM, escalates to generative fallback
            logger.LogInformation("\n=== Processing Transaction 2 (Borderline / Ambiguous) ===");
            string ambiguousTx = "Transaction: $2,800 electronics store purchase from new device, but within billing zip code at 11:30 PM.";
            var result2 = decisionEngine.Decide(
                systemContext: "Determine the risk level of the transaction.",
                inputData: ambiguousTx,
                options: options
            );
            await EvaluateDecisionAsync(ambiguousTx, options, result2, decisionEngine, fallbackModelPath, logger);
        }

        private static async Task EvaluateDecisionAsync(
            string inputData,
            IReadOnlyList<string> options,
            DecisionResult decision,
            DecisionEngine engine,
            string fallbackModelPath,
            ILogger logger)
        {
            logger.LogInformation("Tier-1 Decision: '{SelectedOption}' (Confidence: {Confidence:P1}, Margin: {Margin:P1}, Latency: {Latency}ms)",
                decision.SelectedOption, decision.Confidence, decision.Margin, decision.LatencyMs);

            if (decision.CanAutomate(minMargin: 0.20f, minDominanceRatio: 3.0f))
            {
                logger.LogInformation("=> [AUTOMATION APPROVED] Metric thresholds satisfied. Executing programmatic action immediately.");
            }
            else
            {
                logger.LogWarning("=> [AUTOMATION REJECTED] Margin ({Margin:P1}) insufficient. Escalating to Generative Fallback Review...", decision.Margin);

                // Evict fast model from memory to allocate VRAM for heavy reasoning model
                engine.Unload();

                var stopwatch = Stopwatch.StartNew();
                string secondaryReview = await RouteToFallbackReviewAsync(inputData, options, decision, fallbackModelPath);
                stopwatch.Stop();

                logger.LogInformation("=> [SECONDARY REVIEW OUTCOME] (Fallback took {FallbackMs}ms vs Tier-1 took {Tier1Ms}ms):\n{ReviewResult}",
                    stopwatch.ElapsedMilliseconds, decision.LatencyMs, secondaryReview);
            }
        }

        private static async Task<string> RouteToFallbackReviewAsync(
            string inputData,
            IReadOnlyList<string> options,
            DecisionResult preliminaryDecision,
            string fallbackModelPath)
        {
            if (!File.Exists(fallbackModelPath))
            {
                return $"Fallback model not found at '{fallbackModelPath}'. Fallback review skipped.";
            }

            var modelParams = new ModelParams(fallbackModelPath)
            {
                ContextSize = 2048,
                GpuLayerCount = -1,
                MainGpu = 0
            };

            using var secondaryWeights = LLamaWeights.LoadFromFile(modelParams);
            var executor = new StatelessExecutor(secondaryWeights, modelParams);

            var inferenceParams = new InferenceParams
            {
                MaxTokens = 256,
                SamplingPipeline = new DefaultSamplingPipeline
                {
                    Temperature = 0.1f,
                    TopP = 0.95f
                },
                AntiPrompts = new[] { "<|im_end|>", "<|endoftext|>" }
            };

            string optionsList = string.Join(", ", options);
            string prompt = $"""
            <|im_start|>system
            /no_think
            You are a strict risk arbitrator. A fast triage system marked this request as ambiguous.
            Analyze the input and select EXACTLY ONE of the allowed options: [{optionsList}].
            Rules:
            1. State your chosen option in the first line.
            2. Provide a brief 1-2 sentence justification.
            <|im_end|>
            <|im_start|>user
            Input: {inputData}
            Preliminary Triage: {preliminaryDecision.SelectedOption} (Margin: {preliminaryDecision.Margin:P1})

            Select the final option from [{optionsList}] and justify:
            <|im_end|>
            <|im_start|>assistant
            <think></think>
            """;

            var outputBuilder = new StringBuilder();
            await foreach (var token in executor.InferAsync(prompt, inferenceParams))
            {
                outputBuilder.Append(token);
            }

            return outputBuilder.ToString().Trim();
        }
    }
}