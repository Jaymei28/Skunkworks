using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.Randomizers;

[Serializable]
public class TextureRandomizer : Randomizer
{
    public MaterialParameter materialOptions;

    protected override void OnIterationStart()
    {
        // Logic to swap materials on tagged objects
    }
}
