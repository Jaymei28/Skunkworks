using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

/// <summary>
/// Depth-aware object scale randomizer.
/// Objects are scaled so they always appear as a consistent fraction of the camera
/// frame, regardless of their distance from the camera.
///
/// Formula (perspective camera):
///   scale = (target_fraction * depth * tan(fov_y / 2)) / bounding_radius
///
/// Python params (read by Skunkworks engine):
///   dist_min              – float, minimum camera distance   (default 400.0)
///   dist_max              – float, maximum camera distance   (default 800.0)
///   target_fraction_min   – float, min apparent height fraction (default 0.10)
///   target_fraction_max   – float, max apparent height fraction (default 0.35)
///   scale_jitter          – float, ±relative jitter on computed scale (default 0.15)
///   translation_max       – float, max lateral XY offset in world units (default 50.0)
///   fov_y                 – float, vertical FOV in degrees   (default 60.0)
/// </summary>
[Serializable]
public class DepthScaleRandomizer : Randomizer
{
    [Header("Camera Distance Range")]
    [Tooltip("Minimum distance from camera to object (world units).")]
    public FloatParameter distMin = new FloatParameter { value = 400f };

    [Tooltip("Maximum distance from camera to object (world units).")]
    public FloatParameter distMax = new FloatParameter { value = 800f };

    [Header("Apparent Size")]
    [Tooltip("Object will fill at least this fraction of the image height.")]
    [Range(0.01f, 1f)]
    public float targetFractionMin = 0.10f;

    [Tooltip("Object will fill at most this fraction of the image height.")]
    [Range(0.01f, 1f)]
    public float targetFractionMax = 0.35f;

    [Header("Jitter & Translation")]
    [Tooltip("±Relative scale jitter applied on top of computed scale. 0.15 = ±15%.")]
    [Range(0f, 0.5f)]
    public float scaleJitter = 0.15f;

    [Tooltip("Maximum lateral XY translation offset in world units.")]
    public float translationMax = 50f;

    [Header("Camera")]
    [Tooltip("Vertical field of view of the render camera (degrees). Must match renderer.")]
    [Range(15f, 120f)]
    public float fovY = 60f;

    protected override void OnIterationStart()
    {
        float depth  = UnityEngine.Random.Range(distMin.value, distMax.value);
        float frac   = UnityEngine.Random.Range(targetFractionMin, targetFractionMax);
        float jitter = UnityEngine.Random.Range(1f - scaleJitter, 1f + scaleJitter);
        float tanHalf = Mathf.Tan(fovY * 0.5f * Mathf.Deg2Rad);

        // Retrieve all tagged objects and apply depth-correct scale
        var tags = GameObject.FindGameObjectsWithTag("RandomizerTag");
        foreach (var go in tags)
        {
            float r = go.GetComponent<Renderer>()?.bounds.extents.magnitude ?? 1f;
            float scale = (frac * depth * tanHalf) / Mathf.Max(r, 1e-4f) * jitter;
            go.transform.localScale = Vector3.one * scale;

            float tx = UnityEngine.Random.Range(-translationMax, translationMax);
            float ty = UnityEngine.Random.Range(-translationMax, translationMax);
            go.transform.position += new Vector3(tx, ty, 0f);
        }
    }
}
