using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

/// <summary>
/// Randomizes weather type (clear, rain, fog, dust, overcast) and its intensity.
/// Controls particle systems, fog settings, and post-process volumes.
///
/// Python params (read by Skunkworks engine):
///   weight_clear     – float, relative probability weight for clear sky   (default 0.35)
///   weight_rain      – float, relative probability weight for rain        (default 0.20)
///   weight_fog       – float, relative probability weight for fog         (default 0.20)
///   weight_dust      – float, relative probability weight for dust storm  (default 0.10)
///   weight_overcast  – float, relative probability weight for overcast    (default 0.15)
///   intensity_min    – float, minimum weather effect intensity            (default 0.15)
///   intensity_max    – float, maximum weather effect intensity            (default 0.75)
/// </summary>
[Serializable]
public class WeatherRandomizer : Randomizer
{
    [Header("Weather Type Weights (relative probabilities)")]
    [Tooltip("Higher values make this weather type appear more often. Weights are normalized automatically.")]
    [Min(0f)] public float weightClear    = 0.35f;
    [Min(0f)] public float weightRain     = 0.20f;
    [Min(0f)] public float weightFog      = 0.20f;
    [Min(0f)] public float weightDust     = 0.10f;
    [Min(0f)] public float weightOvercast = 0.15f;

    [Header("Intensity")]
    [Tooltip("Minimum weather effect intensity (0 = off, 1 = full).")]
    [Range(0f, 1f)] public float intensityMin = 0.15f;

    [Tooltip("Maximum weather effect intensity (0 = off, 1 = full).")]
    [Range(0f, 1f)] public float intensityMax = 0.75f;

    [Header("GameObject Tags (optional)")]
    [Tooltip("Tag of rain particle system GameObjects.")]
    public string rainTag     = "Weather_Rain";

    [Tooltip("Tag of dust/sandstorm particle system GameObjects.")]
    public string dustTag     = "Weather_Dust";

    // ── Internal ────────────────────────────────────────────────────────────
    private string[] _types;
    private float[]  _weights;

    protected override void OnScenarioStart()
    {
        _types   = new[] { "clear", "rain", "fog", "dust", "overcast" };
        _weights = new[] { weightClear, weightRain, weightFog, weightDust, weightOvercast };
    }

    protected override void OnIterationStart()
    {
        string weatherType = SampleWeighted(_types, _weights);
        float  intensity   = UnityEngine.Random.Range(intensityMin, intensityMax);

        // ── Fog ─────────────────────────────────────────────────────────────
        bool fogActive = weatherType == "fog" || weatherType == "overcast";
        RenderSettings.fog          = fogActive;
        RenderSettings.fogDensity   = fogActive ? intensity * 0.04f : 0f;
        RenderSettings.fogColor     = weatherType == "overcast"
            ? new Color(0.55f, 0.55f, 0.58f)
            : new Color(0.85f, 0.80f, 0.70f);

        // ── Rain particles ───────────────────────────────────────────────────
        SetParticleActive(rainTag, weatherType == "rain", intensity);

        // ── Dust particles ───────────────────────────────────────────────────
        SetParticleActive(dustTag, weatherType == "dust", intensity);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────
    private static string SampleWeighted(string[] items, float[] weights)
    {
        float total = 0f;
        foreach (var w in weights) total += w;
        float roll  = UnityEngine.Random.Range(0f, total);
        float cumul = 0f;
        for (int i = 0; i < items.Length; i++)
        {
            cumul += weights[i];
            if (roll <= cumul) return items[i];
        }
        return items[0];
    }

    private static void SetParticleActive(string tag, bool active, float intensity)
    {
        foreach (var go in GameObject.FindGameObjectsWithTag(tag))
        {
            go.SetActive(active);
            if (active)
            {
                var ps = go.GetComponent<ParticleSystem>();
                if (ps != null)
                {
                    var main = ps.main;
                    main.maxParticles = Mathf.RoundToInt(Mathf.Lerp(50, 2000, intensity));
                }
            }
        }
    }
}
