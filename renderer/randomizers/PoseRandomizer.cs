using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

[Serializable]
public class PoseRandomizer : Randomizer
{
    public FloatParameter radius = new FloatParameter { value = 5f };
    public FloatParameter separationDistance = new FloatParameter { value = 2f };
    public Vector3Parameter rotationRange = new Vector3Parameter 
    { 
        x = new FloatParameter { range = new FloatRange(0, 360) },
        y = new FloatParameter { range = new FloatRange(0, 360) },
        z = new FloatParameter { range = new FloatRange(0, 360) }
    };

    protected override void OnIterationStart()
    {
        // Internal logic for Unity-side execution
    }
}
