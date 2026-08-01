"""Frozen A2261 recipe: individually calibrated 2-D frames before sky fitting."""

recipe_tags = {"GMOS", "SPECT", "LS"}


def reduceFrozenCalibrated2D(p):
    """Follow DRAGONS 4.2.2 long-slit processing through distortion only."""
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
    p.distortionCorrect()
    p.writeOutputs(suffix="_cal2d")
