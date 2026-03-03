using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

[Serializable]
public class HueOffsetRandomizer : Randomizer
{
    public FloatParameter hueOffset = new FloatParameter { range = new FloatRange(-0.2f, 0.2f) };

    protected override void OnIterationStart()
    {
        // Logic to apply hue shifts
    }
}
