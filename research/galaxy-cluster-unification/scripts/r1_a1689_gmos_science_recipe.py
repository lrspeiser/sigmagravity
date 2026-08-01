"""Frozen A1689 recipe: individually calibrated 2-D frames, before sky fitting."""

recipe_tags = {"GMOS", "SPECT", "LS"}


def reduceFrozenCalibrated2D(p):
    """Follow the DRAGONS 4.2.2 SQ long-slit recipe through distortion only."""
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.attachWavelengthSolution()
    p.flatCorrect()
    p.QECorrect()
    p.flagCosmicRays()
    # distortionCorrect mosaics and rectifies in a single interpolation.
    p.distortionCorrect()
    p.writeOutputs(suffix="_cal2d")
