using System.Diagnostics;
using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
using Microsoft.Extensions.Logging;

namespace TIEG.SystemOne.LLamaSharp
{
    /// <summary>
    /// A thread-safe, auto-unloading inference engine designed specifically for 
    /// "System One" decision models (like Plumb-4B). Evaluates unstructured text into typed deterministic options.
    /// </summary>
    public class DecisionEngine : IDisposable
    {
        private readonly string _modelPath;
        private readonly float _temperature;
        private readonly TimeSpan _idleTimeout;
        private readonly ILogger<DecisionEngine>? _logger;
        private readonly string[] _letters = { "A", "B", "C", "D", "E", "F" };

        private LLamaWeights? _weights;
        private LLamaContext? _context;
        private Dictionary<string, int>? _cachedOptionTokens;

        private readonly SemaphoreSlim _inferenceLock = new(1, 1);
        private Timer? _idleTimer;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the DecisionEngine. The model is not loaded into VRAM until the first decision is requested.
        /// </summary>
        /// <param name="modelPath">The absolute or relative path to the GGUF model file.</param>
        /// <param name="calibrationTemperature">The specific calibration temperature for the model (Plumb-4B defaults to 2.07f).</param>
        /// <param name="idleMinutes">Minutes of inactivity before the model automatically drops from VRAM.</param>
        /// <param name="logger">Optional logger for telemetry and debugging.</param>
        public DecisionEngine(
            string modelPath,
            float calibrationTemperature = 2.07f,
            int idleMinutes = 5,
            ILogger<DecisionEngine>? logger = null)
        {
            _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
            _temperature = calibrationTemperature;
            _idleTimeout = TimeSpan.FromMinutes(idleMinutes);
            _logger = logger;
        }

        private void EnsureLoaded()
        {
            if (_context != null) return;

            _logger?.LogInformation("Loading decision weights into VRAM from {ModelPath}...", _modelPath);
            var parameters = new ModelParams(_modelPath)
            {
                ContextSize = 4096,
                GpuLayerCount = -1,
                MainGpu = 0
            };

            _weights = LLamaWeights.LoadFromFile(parameters);
            _context = _weights.CreateContext(parameters);

            // Cache token IDs for option letters once upon load
            _cachedOptionTokens = new Dictionary<string, int>();
            foreach (var letter in _letters)
            {
                int tokenId = (int)_context.Tokenize($" {letter}", special: false)[0];
                _cachedOptionTokens[letter] = tokenId;
            }

            _logger?.LogInformation("Decision engine loaded and ready.");
        }

        private void OnIdleTimeout(object? state)
        {
            if (!_inferenceLock.Wait(0)) return;

            try
            {
                if (_context != null)
                {
                    _logger?.LogInformation("Idle timeout reached. Evicting model from VRAM...");
                    _context.Dispose();
                    _weights?.Dispose();

                    _context = null;
                    _weights = null;
                    _cachedOptionTokens = null;
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error while un-allocating model memory during idle timeout.");
            }
            finally
            {
                _inferenceLock.Release();
            }
        }

        /// <summary>
        /// Evaluates unstructured input data against a provided list of options in a single parallel pass.
        /// </summary>
        /// <param name="systemContext">The core instruction or question (e.g., 'Determine the risk level').</param>
        /// <param name="inputData">The unstructured context to analyze.</param>
        /// <param name="options">A list of up to 6 distinct options to score.</param>
        /// <returns>A calibrated DecisionResult containing the winning option and statistical distribution.</returns>
        public DecisionResult Decide(string systemContext, string inputData, IReadOnlyList<string> options)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (options.Count > _letters.Length)
                throw new ArgumentException($"Maximum supported options is {_letters.Length}", nameof(options));

            _inferenceLock.Wait();
            try
            {
                var stopwatch = Stopwatch.StartNew();
                EnsureLoaded();

                // Reset the auto-unload timer
                _idleTimer?.Dispose();
                _idleTimer = new Timer(OnIdleTimeout, null, _idleTimeout, Timeout.InfiniteTimeSpan);

                // Build multiple-choice prompt
                var promptBuilder = new StringBuilder();
                promptBuilder.AppendLine(systemContext);
                promptBuilder.AppendLine($"\nInput: {inputData}\n\nOptions:");

                for (int i = 0; i < options.Count; i++)
                {
                    promptBuilder.AppendLine($"{_letters[i]}) {options[i]}");
                }
                promptBuilder.Append("\nDecision:");

                // 1. Reset the KV cache sequence
                _context!.NativeHandle.MemorySequenceRemove(LLamaSeqId.Zero, -1, -1);

                // 2. Tokenize prompt and populate batch
                var tokens = _context.Tokenize(promptBuilder.ToString(), true);
                var batch = new LLamaBatch();

                for (int i = 0; i < tokens.Length; i++)
                {
                    batch.Add(tokens[i], i, LLamaSeqId.Zero, i == tokens.Length - 1);
                }

                // 3. Single-pass evaluation
                _context.Decode(batch);

                // 4. Extract logits for the target token
                float[] allLogits = _context.NativeHandle.GetLogitsIth(tokens.Length - 1).ToArray();
                var probabilities = new Dictionary<string, float>();
                float sumExp = 0;

                // 5. Softmax over the options using calibrated temperature
                for (int i = 0; i < options.Count; i++)
                {
                    string letter = _letters[i];
                    int tokenId = _cachedOptionTokens![letter];

                    float logit = allLogits[tokenId];
                    float expValue = (float)Math.Exp(logit / _temperature);

                    probabilities.Add(options[i], expValue);
                    sumExp += expValue;
                }

                var sorted = probabilities
                    .Select(kvp => new KeyValuePair<string, float>(kvp.Key, kvp.Value / sumExp))
                    .OrderByDescending(x => x.Value)
                    .ToList();

                var top = sorted[0];
                var runnerUp = sorted.Count > 1 ? sorted[1] : new KeyValuePair<string, float>(string.Empty, 0f);

                stopwatch.Stop();

                var result = new DecisionResult
                {
                    SelectedOption = top.Key,
                    Confidence = top.Value,
                    Margin = top.Value - runnerUp.Value,
                    Distribution = sorted.ToDictionary(k => k.Key, v => v.Value),
                    LatencyMs = stopwatch.ElapsedMilliseconds
                };

                _logger?.LogDebug("Decision computed in {Latency}ms: Top={SelectedOption} ({Confidence:P1}), Margin={Margin:P1}",
                    result.LatencyMs, result.SelectedOption, result.Confidence, result.Margin);

                return result;
            }
            finally
            {
                _inferenceLock.Release();
            }
        }

        /// <summary>
        /// Explicitly evicts the model from VRAM immediately. 
        /// It will automatically reload on the next call to Decide().
        /// </summary>
        public void Unload()
        {
            if (_disposed) return;

            _inferenceLock.Wait();
            try
            {
                if (_context != null)
                {
                    _logger?.LogInformation("Explicitly unloading model to free VRAM for secondary tasks...");
                    _context.Dispose();
                    _weights?.Dispose();

                    _context = null;
                    _weights = null;
                    _cachedOptionTokens = null;
                }
            }
            finally
            {
                _inferenceLock.Release();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _idleTimer?.Dispose();
            _context?.Dispose();
            _weights?.Dispose();
            _inferenceLock.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}

