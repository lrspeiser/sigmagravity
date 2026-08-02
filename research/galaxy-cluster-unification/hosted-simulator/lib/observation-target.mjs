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
    if (target.kind !== "circular_speed_curve") throw new Error(`unsupported observation target kind: ${target.kind}`);
    const definition = definitions.get(target.observable);
    if (!definition) throw new Error(`observation target ${target.id} requires unknown observable ${target.observable}`);
    if (!requested.has(target.observable)) throw new Error(`observation target ${target.id} observable must be requested`);
    if (definition.target !== "massive_tracers" || definition.rank !== "vector" || definition.unit !== "m/s^2") {
      throw new Error(`observation target ${target.id} requires a massive_tracers vector in m/s^2`);
    }
    if (!["cartesian_2d", "cartesian_3d"].includes(model.geometry.coordinateSystem)) {
      throw new Error("circular_speed_curve supports Cartesian 2D or 3D models");
    }
    finiteArray(target.centerM, dimensions, `observation target ${target.id} centerM`);
    if (target.gridOriginM !== undefined) finiteArray(target.gridOriginM, dimensions, `observation target ${target.id} gridOriginM`);
    else if (inputBundle.geometry?.origin !== undefined) finiteArray(inputBundle.geometry.origin, dimensions, "input bundle origin");
    const planeAxes = target.planeAxes ?? [0, 1];
    if (!Array.isArray(planeAxes) || planeAxes.length !== 2 || !planeAxes.every(Number.isInteger)
      || new Set(planeAxes).size !== 2 || planeAxes.some((axis) => axis < 0 || axis >= dimensions)) {
      throw new Error(`observation target ${target.id} planeAxes are invalid`);
    }
    const radii = finiteArray(target.radiiM, null, `observation target ${target.id} radiiM`, { positive: true });
    if (radii.some((value, index) => index > 0 && value <= radii[index - 1])) throw new Error(`observation target ${target.id} radiiM must be strictly increasing`);
    const sampleCount = target.azimuthalSamples ?? 128;
    if (!Number.isInteger(sampleCount) || sampleCount < 16 || sampleCount > 4096) throw new Error("azimuthalSamples must be an integer from 16 through 4096");
    const coverage = target.minimumAzimuthalCoverage ?? 0.8;
    if (!Number.isFinite(coverage) || coverage <= 0 || coverage > 1) throw new Error("minimumAzimuthalCoverage must lie in (0,1]");
    const scored = target.observedSpeedsMPerS !== undefined;
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
    const fitted = target.fittedNuisanceParameters ?? 0;
    if (!Number.isInteger(fitted) || fitted < 0 || (scored && fitted >= radii.length)) throw new Error("fittedNuisanceParameters must be smaller than the scored point count");
    if (!target.provenance || typeof target.provenance !== "object" || Array.isArray(target.provenance) || Object.keys(target.provenance).length === 0) throw new Error("observation target requires provenance");
    license(target.license);
    return {
      id: target.id,
      kind: target.kind,
      observable: target.observable,
      targetSha256: sha256(target),
      pointCount: radii.length,
      scored,
      fittedNuisanceParameters: fitted,
    };
  });
}
