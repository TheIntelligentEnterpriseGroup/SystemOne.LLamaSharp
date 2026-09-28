using System.Collections.Generic;

namespace TIEG.SystemOne.LLamaSharp
{
    /// <summary>
    /// Represents the statistical outcome of a deterministic decision pass.
    /// </summary>
    public class DecisionResult
    {
        /// <summary>The winning option label.</summary>
        public string SelectedOption { get; init; } = string.Empty;

        /// <summary>Calibrated probability of the winning label (0.0 to 1.0).</summary>
        public float Confidence { get; init; }

        /// <summary>
        /// Difference between the top-1 and top-2 probabilities.
        /// A narrow margin indicates ambiguity between competing options.
        /// </summary>
        public float Margin { get; init; }

        /// <summary>Full normalized probability distribution sorted descending.</summary>
        public IReadOnlyDictionary<string, float> Distribution { get; init; }
            = new Dictionary<string, float>();

        /// <summary>Inference execution time in milliseconds.</summary>
        public long LatencyMs { get; init; }

        /// <summary>
        /// Evaluates if the decision clears universal automation thresholds 
        /// regardless of the total number of options provided.
        /// </summary>
        public bool CanAutomate(float minMargin = 0.35f, float minDominanceRatio = 3.0f)
        {
            bool hasSafeMargin = Margin >= minMargin;
            float runnerUpProb = Confidence - Margin;
            float epsilon = 0.0001f;
            float ratio = Confidence / (runnerUpProb + epsilon);
            bool isDominant = ratio >= minDominanceRatio;

            return hasSafeMargin && isDominant;
        }
    }
}