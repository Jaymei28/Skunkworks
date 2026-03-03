using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

[Serializable]
public class LightRandomizer : Randomizer
{
    public FloatParameter intensity = new FloatParameter { range = new FloatRange(0.5f, 2.0f) };
    public ColorParameter color = new ColorParameter();

    protected override void OnIterationStart()
    {
        // Logic to randomize scene lighting
    }
}
