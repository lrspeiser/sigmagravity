import { sha256 } from "./canonical.mjs";

function finiteArray(value, length, label, { positive = false } = {}) {
  if (!Array.isArray(value) || (length !== null && value.length !== length) || value.length === 0) {
    throw new Error(`${label} must contain ${length ?? "one or more"} values`);
  }
  const result = value.map(Number);
  if (result.some((item) => !Number.isFinite(item) || (positive && item <= 0))) {
    throw new Error(`${label} must contain ${positive ? "positive " : ""}finite values`);
  }
  return result;
}

function positiveDefinite(matrix, size) {
  if (!Array.isArray(matrix) || matrix.length !== size) return false;
  const values = matrix.map((row) => finiteArray(row, size, "covariance row"));
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const tolerance = 1e-10 * Math.max(1, Math.abs(values[row][column]), Math.abs(values[column][row]));
      if (Math.abs(values[row][column] - values[column][row]) > tolerance) return false;
    }
  }
  const lower = Array.from({ length: size }, () => Array(size).fill(0));
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = values[row][column];
      for (let inner = 0; inner < column; inner += 1) value -= lower[row][inner] * lower[column][inner];
      if (row === column) {
        if (!(value > 0)) return false;
        lower[row][column] = Math.sqrt(value);
      } else {
        lower[row][column] = value / lower[column][column];
      }
    }
  }
  return true;
}

function license(value) {
  if (!value || typeof value.id !== "string" || !value.id || typeof value.redistributionAllowed !== "boolean") {
    throw new Error("observation target requires an explicit license");
  }
}

function observationArray(inputBundle, key, label, { unit, shape = null } = {}) {
  if (typeof key !== "string" || !key) throw new Error(`${label} requires an array key`);
  const record = inputBundle.arrays.find((item) => item.key === key);
  if (!record) throw new Error(`${label} references missing input array ${key}`);
  if (record.rank !== "scalar" || (unit !== undefined && record.unit !== unit)) {
    throw new Error(`${label} must reference a scalar ${unit ?? ""} array`.trim());
  }
  if (record.shape.length !== 2 || (shape && JSON.stringify(record.shape) !== JSON.stringify(shape))) {
    throw new Error(`${label} must reference a two-dimensional map with the declared observation shape`);
  }
  return record;
}

export function validateObservationTargets({ targets = [], model, inputBundle, requestedObservables }) {
  if (!Array.isArray(targets) || targets.length > 32) throw new Error("observationTargets must contain at most 32 targets");
  const dimensions = model.geometry.dimensions;
  const definitions = new Map(model.observables.map((item) => [item.id, item]));
  const requested = new Set(requestedObservables);
  const seen = new Set();
  return targets.map((target) => {
    if (!target || typeof target !== "object" || Array.isArray(target)) throw new Error("observation target must be an object");
    if (target.schemaVersion !== "sigma-observation-target/1") throw new Error("observation target must use sigma-observation-target/1");
    if (typeof target.id !== "string" || !target.id || seen.has(target.id)) throw new Error(`invalid or duplicate observation target id: ${target.id}`);
    seen.add(target.id);
    if (!["circular_speed_curve", "line_of_sight_velocity_field"].includes(target.kind)) {
      throw new Error(`unsupported observation target kind: ${target.kind}`);
    }
    const definition = definitions.get(target.observable);
    if (!definition) throw new Error(`observation target ${target.id} requires unknown observable ${target.observable}`);
    if (!requested.has(target.observable)) throw new Error(`observation target ${target.id} observable must be requested`);
    if (definition.target !== "massive_tracers" || definition.rank !== "vector" || definition.unit !== "m/s^2") {
      throw new Error(`observation target ${target.id} requires a massive_tracers vector in m/s^2`);
    }
    if (!["cartesian_2d", "cartesian_3d"].includes(model.geometry.coordinateSystem)) {
      throw new Error(`${target.kind} supports Cartesian 2D or 3D models`);
    }
    finiteArray(target.centerM, dimensions, `observation target ${target.id} centerM`);
    if (target.gridOriginM !== undefined) finiteArray(target.gridOriginM, dimensions, `observation target ${target.id} gridOriginM`);
    else if (inputBundle.geometry?.origin !== undefined) finiteArray(inputBundle.geometry.origin, dimensions, "input bundle origin");
    const planeAxes = target.planeAxes ?? [0, 1];
    if (!Array.isArray(planeAxes) || planeAxes.length !== 2 || !planeAxes.every(Number.isInteger)
      || new Set(planeAxes).size !== 2 || planeAxes.some((axis) => axis < 0 || axis >= dimensions)) {
      throw new Error(`observation target ${target.id} planeAxes are invalid`);
    }
    let pointCount;
    let scored;
    if (target.kind === "circular_speed_curve") {
      const radii = finiteArray(target.radiiM, null, `observation target ${target.id} radiiM`, { positive: true });
      if (radii.some((value, index) => index > 0 && value <= radii[index - 1])) throw new Error(`observation target ${target.id} radiiM must be strictly increasing`);
      const sampleCount = target.azimuthalSamples ?? 128;
      if (!Number.isInteger(sampleCount) || sampleCount < 16 || sampleCount > 4096) throw new Error("azimuthalSamples must be an integer from 16 through 4096");
      const coverage = target.minimumAzimuthalCoverage ?? 0.8;
      if (!Number.isFinite(coverage) || coverage <= 0 || coverage > 1) throw new Error("minimumAzimuthalCoverage must lie in (0,1]");
      scored = target.observedSpeedsMPerS !== undefined;
      if (scored) {
        finiteArray(target.observedSpeedsMPerS, radii.length, `observation target ${target.id} observedSpeedsMPerS`, { positive: true });
        const hasUncertainty = target.uncertaintiesMPerS !== undefined;
        const hasCovariance = target.covarianceM2PerS2 !== undefined;
        if (hasUncertainty === hasCovariance) throw new Error("scored observation targets require exactly one uncertainty or covariance input");
        if (hasUncertainty) finiteArray(target.uncertaintiesMPerS, radii.length, `observation target ${target.id} uncertaintiesMPerS`, { positive: true });
        if (hasCovariance && !positiveDefinite(target.covarianceM2PerS2, radii.length)) throw new Error(`observation target ${target.id} covariance must be symmetric positive definite`);
      } else if (target.uncertaintiesMPerS !== undefined || target.covarianceM2PerS2 !== undefined) {
        throw new Error("uncertainty data requires observedSpeedsMPerS");
      }
      pointCount = radii.length;
    } else {
      const inclination = Number(target.inclinationDeg);
      if (!Number.isFinite(inclination) || inclination <= 0 || inclination >= 90) {
        throw new Error("inclinationDeg must lie strictly between 0 and 90 degrees");
      }
      if (![1, -1].includes(target.handedness)) throw new Error("handedness must be -1 or 1");
      const nonpositivePolicy = target.nonPositiveInwardPolicy ?? "exclude";
      if (!["exclude", "zero_speed"].includes(nonpositivePolicy)) throw new Error("nonPositiveInwardPolicy must be exclude or zero_speed");
      const major = observationArray(inputBundle, target.majorCoordinateArrayKey, "majorCoordinateArrayKey", { unit: "m" });
      observationArray(inputBundle, target.minorCoordinateArrayKey, "minorCoordinateArrayKey", { unit: "m", shape: major.shape });
      if (target.maskArrayKey !== undefined && target.scoreMaskArrayKey !== undefined) throw new Error("use either maskArrayKey or scoreMaskArrayKey, not both");
      if (target.maskArrayKey !== undefined) observationArray(inputBundle, target.maskArrayKey, "maskArrayKey", { unit: "1", shape: major.shape });
      if (target.scoreMaskArrayKey !== undefined) observationArray(inputBundle, target.scoreMaskArrayKey, "scoreMaskArrayKey", { unit: "1", shape: major.shape });
      if (target.emissionMaskArrayKey !== undefined) observationArray(inputBundle, target.emissionMaskArrayKey, "emissionMaskArrayKey", { unit: "1", shape: major.shape });
      const hasObserved = target.observedVelocityArrayKey !== undefined;
      const hasUncertainty = target.uncertaintyArrayKey !== undefined;
      if (hasObserved !== hasUncertainty) throw new Error("resolved velocity scoring requires both observedVelocityArrayKey and uncertaintyArrayKey");
      if (hasObserved) {
        observationArray(inputBundle, target.observedVelocityArrayKey, "observedVelocityArrayKey", { unit: "m/s", shape: major.shape });
        observationArray(inputBundle, target.uncertaintyArrayKey, "uncertaintyArrayKey", { unit: "m/s", shape: major.shape });
      }
      const weighting = target.weighting ?? "inverse_variance";
      if (!["inverse_variance", "intensity_inverse_variance"].includes(weighting)) throw new Error("unsupported velocity-field weighting");
      const requiresIntensity = weighting === "intensity_inverse_variance" || target.beamKernelArrayKey !== undefined;
      if (requiresIntensity) observationArray(inputBundle, target.intensityWeightArrayKey, "intensityWeightArrayKey", { shape: major.shape });
      if (target.beamKernelArrayKey !== undefined) {
        const kernel = observationArray(inputBundle, target.beamKernelArrayKey, "beamKernelArrayKey", { unit: "1" });
        if (kernel.shape.some((value) => value % 2 === 0)) throw new Error("beam kernel dimensions must be odd");
      }
      const minimumValid = target.minimumValidPixels ?? 25;
      if (!Number.isInteger(minimumValid) || minimumValid < 1 || minimumValid > major.elementCount) throw new Error("minimumValidPixels must fit within the observation map");
      const zeroPoint = Number(target.observedVelocityZeroPointMPerS ?? 0);
      if (!Number.isFinite(zeroPoint)) throw new Error("observedVelocityZeroPointMPerS must be finite");
      scored = hasObserved;
      pointCount = major.elementCount;
    }
    const fitted = target.fittedNuisanceParameters ?? 0;
    if (!Number.isInteger(fitted) || fitted < 0 || (scored && fitted >= pointCount)) throw new Error("fittedNuisanceParameters must be smaller than the scored point count");
    if (!target.provenance || typeof target.provenance !== "object" || Array.isArray(target.provenance) || Object.keys(target.provenance).length === 0) throw new Error("observation target requires provenance");
    license(target.license);
    return {
      id: target.id,
      kind: target.kind,
      observable: target.observable,
      targetSha256: sha256(target),
      pointCount,
      scored,
      fittedNuisanceParameters: fitted,
    };
  });
}
