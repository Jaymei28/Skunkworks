using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

/// <summary>
/// Randomizes URP / HDRP post-processing overrides each iteration.
/// Attach to the same GameObject as the global Volume.
///
/// Toggle each effect independently and set per-effect min/max ranges.
///
/// Python params (read by Skunkworks engine):
///   exposure_enabled          – bool   (default true)
///   exposure_min              – float  (default -0.5)
///   exposure_max              – float  (default  0.5)
///   bloom_enabled             – bool   (default true)
///   bloom_min                 – float  (default 0.0)
///   bloom_max                 – float  (default 0.3)
///   bloom_threshold           – float  (default 0.7)
///   noise_enabled             – bool   (default true)
///   noise_mode                – string "small" | "large" | "random"  (default "random")
///   noise_min                 – float  (default 0.0)
///   noise_max                 – float  (default 0.06)
///   ao_enabled                – bool   (default true)
///   ao_min                    – float  (default 0.0)
///   ao_max                    – float  (default 0.35)
///   wb_enabled                – bool   (default true)
///   wb_temp_min               – float  (default 3500)
///   wb_temp_max               – float  (default 8500)
///   blur_enabled              – bool   (default true)
///   blur_min                  – float  (default 0.0)
///   blur_max                  – float  (default 1.8)
/// </summary>
[Serializable]
public class PostProcessRandomizer : Randomizer
{
    // ── Exposure ─────────────────────────────────────────────────────────────
    [Header("Exposure")]
    public bool  exposureEnabled = true;
    [Range(-3f, 3f)]  public float exposureMin = -0.5f;
    [Range(-3f, 3f)]  public float exposureMax =  0.5f;

    // ── Bloom ────────────────────────────────────────────────────────────────
    [Header("Bloom")]
    public bool  bloomEnabled   = true;
    [Range(0f, 1f)]   public float bloomMin       = 0.00f;
    [Range(0f, 1f)]   public float bloomMax       = 0.30f;
    [Range(0f, 1f)]   public float bloomThreshold = 0.70f;

    // ── Film Grain / Noise ───────────────────────────────────────────────────
    [Header("Noise (Film Grain)")]
    public bool  noiseEnabled = true;

    [Tooltip("small = fine grain, large = coarse grain, random = pick each frame.")]
    public NoiseMode noiseMode = NoiseMode.Random;

    [Range(0f, 1f)]   public float noiseMin = 0.00f;
    [Range(0f, 1f)]   public float noiseMax = 0.06f;

    // ── Ambient Occlusion ────────────────────────────────────────────────────
    [Header("Ambient Occlusion (SSAO)")]
    public bool  aoEnabled = true;
    [Range(0f, 2f)]   public float aoMin = 0.00f;
    [Range(0f, 2f)]   public float aoMax = 0.35f;

    // ── White Balance ────────────────────────────────────────────────────────
    [Header("White Balance")]
    public bool  wbEnabled = true;
    [Range(1000f, 20000f)] public float wbTempMin = 3500f;
    [Range(1000f, 20000f)] public float wbTempMax = 8500f;

    // ── Depth of Field (Blur) ────────────────────────────────────────────────
    [Header("Blur (Depth of Field)")]
    public bool  blurEnabled = true;
    [Range(0f, 10f)]  public float blurMin = 0.0f;
    [Range(0f, 10f)]  public float blurMax = 1.8f;

    // ── Internal refs ────────────────────────────────────────────────────────
    private Volume            _volume;
    private ColorAdjustments  _colorAdj;
    private Bloom             _bloom;
    private FilmGrain         _filmGrain;
    private DepthOfField      _dof;
    private WhiteBalance      _wb;

    protected override void OnScenarioStart()
    {
        _volume = GetComponent<Volume>();
        if (_volume == null) return;
        _volume.profile.TryGet(out _colorAdj);
        _volume.profile.TryGet(out _bloom);
        _volume.profile.TryGet(out _filmGrain);
        _volume.profile.TryGet(out _dof);
        _volume.profile.TryGet(out _wb);
    }

    protected override void OnIterationStart()
    {
        // ── Exposure ──────────────────────────────────────────────────────
        if (exposureEnabled && _colorAdj != null)
        {
            _colorAdj.postExposure.Override(
                UnityEngine.Random.Range(exposureMin, exposureMax));
        }

        // ── Bloom ─────────────────────────────────────────────────────────
        if (bloomEnabled && _bloom != null)
        {
            _bloom.active    = true;
            _bloom.intensity.Override(UnityEngine.Random.Range(bloomMin, bloomMax));
            _bloom.threshold.Override(bloomThreshold);
        }
        else if (_bloom != null)
        {
            _bloom.active = false;
        }

        // ── Film Grain / Noise ────────────────────────────────────────────
        if (noiseEnabled && _filmGrain != null)
        {
            _filmGrain.active = true;
            _filmGrain.intensity.Override(UnityEngine.Random.Range(noiseMin, noiseMax));

            FilmGrainLookup lookup;
            switch (noiseMode)
            {
                case NoiseMode.Small:  lookup = FilmGrainLookup.Thin1;   break;
                case NoiseMode.Large:  lookup = FilmGrainLookup.Medium1; break;
                default:               // Random
                    lookup = UnityEngine.Random.value > 0.5f
                        ? FilmGrainLookup.Thin1 : FilmGrainLookup.Medium1;
                    break;
            }
            _filmGrain.type.Override(lookup);
        }
        else if (_filmGrain != null)
        {
            _filmGrain.active = false;
        }

        // ── White Balance ─────────────────────────────────────────────────
        if (wbEnabled && _wb != null)
        {
            _wb.active = true;
            _wb.temperature.Override(UnityEngine.Random.Range(wbTempMin, wbTempMax));
        }
        else if (_wb != null)
        {
            _wb.active = false;
        }

        // ── Blur / DoF ────────────────────────────────────────────────────
        if (blurEnabled && _dof != null)
        {
            float sigma = UnityEngine.Random.Range(blurMin, blurMax);
            _dof.active = sigma > 0.01f;
            if (_dof.active)
                _dof.gaussianMaxRadius.Override(sigma);
        }
        else if (_dof != null)
        {
            _dof.active = false;
        }
    }

    // ── Ambient Occlusion (SSAO via Renderer Feature) ─────────────────────
    // AO is controlled as a Renderer Feature; toggle via its volume override.
    // If using SSAO Volume Override add it here.
}

/// <summary>Noise grain size mode for the PostProcessRandomizer.</summary>
public enum NoiseMode
{
    Small,   // Fine / thin grain
    Large,   // Coarse / medium grain
    Random   // Pick randomly each iteration
}
