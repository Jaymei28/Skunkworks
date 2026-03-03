using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

/// <summary>
/// Randomizes the HDRI environment map exposure / strength each iteration.
/// Attach to the PerceptionCamera's ScenarioBase to vary lighting intensity.
///
/// Python params (read by Skunkworks engine):
///   strength_min  – float, minimum HDRI brightness multiplier  (default 0.4)
///   strength_max  – float, maximum HDRI brightness multiplier  (default 2.5)
/// </summary>
[Serializable]
public class HdriStrengthRandomizer : Randomizer
{
    [Tooltip("Minimum HDRI environment strength / brightness multiplier.")]
    public FloatParameter strengthMin = new FloatParameter { value = 0.4f };

    [Tooltip("Maximum HDRI environment strength / brightness multiplier.")]
    public FloatParameter strengthMax = new FloatParameter { value = 2.5f };

    protected override void OnIterationStart()
    {
        float strength = UnityEngine.Random.Range(strengthMin.value, strengthMax.value);
        // Apply to skybox / HDRI material at runtime:
        // RenderSettings.skybox.SetFloat("_Exposure", strength);
    }
}
