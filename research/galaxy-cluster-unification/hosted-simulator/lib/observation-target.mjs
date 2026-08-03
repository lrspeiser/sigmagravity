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

export function validateObservationTargets({
  targets = [],
  model,
  inputBundle,
  requestedObservables,
  fieldShape = null,
  fieldGeometry = null,
}) {
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
    if (!["circular_speed_curve", "line_of_sight_velocity_field", "photon_lensing_map", "multiple_image_systems"].includes(target.kind)) {
      throw new Error(`unsupported observation target kind: ${target.kind}`);
    }
    const definition = definitions.get(target.observable);
    if (!definition) throw new Error(`observation target ${target.id} requires unknown observable ${target.observable}`);
    if (!requested.has(target.observable)) throw new Error(`observation target ${target.id} observable must be requested`);
    let pointCount;
    let scored;
    let derivedFitted = null;
    if (target.kind === "multiple_image_systems") {
      if (!["photons", "both"].includes(definition.target) || definition.rank !== "vector" || definition.unit !== "m/s^2") {
        throw new Error(`observation target ${target.id} requires a photons or both vector in m/s^2`);
      }
      if (model.geometry.coordinateSystem !== "cartesian_3d" || dimensions !== 3) {
        throw new Error("multiple_image_systems requires a Cartesian 3D model");
      }
      const axes = [target.northAxis, target.eastAxis, target.lineOfSightAxis];
      if (!axes.every(Number.isInteger) || new Set(axes).size !== 3 || axes.some((axis) => axis < 0 || axis > 2)) {
        throw new Error("northAxis, eastAxis, and lineOfSightAxis must be a permutation of [0,1,2]");
      }
      const lensDistance = Number(target.lensAngularDiameterDistanceM);
      const rootBound = Number(target.rootSearchBoundArcsec);
      if (!Number.isFinite(lensDistance) || lensDistance <= 0) throw new Error("lensAngularDiameterDistanceM must be finite and positive");
      if (!Number.isFinite(rootBound) || rootBound <= 0) throw new Error("rootSearchBoundArcsec must be finite and positive");
      finiteArray(target.skyCenterM, 3, "skyCenterM");
      if (!Array.isArray(fieldShape) || fieldShape.length !== 3 || !fieldShape.every((value) => Number.isInteger(value) && value >= 3)) {
        throw new Error("multiple_image_systems preflight requires the solved 3D field shape");
      }
      const integerControl = (name, fallback, minimum, maximum) => {
        const value = target[name] ?? fallback;
        if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} through ${maximum}`);
      };
      integerControl("rootGridPoints", 161, 21, 401);
      integerControl("maximumResidualMinimumSeeds", 64, 1, 512);
      integerControl("criticalCurveGridPoints", 161, 21, 401);
      for (const [name, fallback] of [["closureToleranceArcsec", 0.002], ["deduplicationToleranceArcsec", 0.2], ["jacobianStepArcsec", 0.08]]) {
        const value = Number(target[name] ?? fallback);
        if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
      }
      for (const name of ["includeResidualMinima", "includeCriticalCurves"]) {
        if (target[name] !== undefined && typeof target[name] !== "boolean") throw new Error(`${name} must be boolean`);
      }
      const supplemental = target.supplementalGridPoints ?? [81, 161, 241];
      if (!Array.isArray(supplemental) || supplemental.length > 4
        || supplemental.some((value) => !Number.isInteger(value) || value < 21 || value > 401)) {
        throw new Error("supplementalGridPoints must contain at most four integers from 21 through 401");
      }
      if (!Array.isArray(target.families) || target.families.length < 1 || target.families.length > 64) {
        throw new Error("families must contain from 1 through 64 image families");
      }
      const familyIds = new Set();
      let imageCount = 0;
      for (const family of target.families) {
        if (!family || typeof family !== "object" || Array.isArray(family)
          || typeof family.id !== "string" || !family.id || familyIds.has(family.id)) {
          throw new Error(`invalid or duplicate image family id: ${family?.id}`);
        }
        familyIds.add(family.id);
        const ratio = Number(family.distanceRatio);
        if (!Number.isFinite(ratio) || ratio <= 0) throw new Error(`family ${family.id} distanceRatio must be finite and positive`);
        if (!Array.isArray(family.observedImagesArcsec) || family.observedImagesArcsec.length < 2) {
          throw new Error(`family ${family.id} observedImagesArcsec must contain at least two positions`);
        }
        family.observedImagesArcsec.forEach((image) => finiteArray(image, 2, `family ${family.id} observed image`));
        finiteArray(family.positionUncertaintiesArcsec, family.observedImagesArcsec.length, `family ${family.id} positionUncertaintiesArcsec`, { positive: true });
        imageCount += family.observedImagesArcsec.length;
      }
      if (imageCount > 512) throw new Error("multiple_image_systems supports at most 512 observed images");
      pointCount = 2 * imageCount;
      scored = true;
      derivedFitted = 2 * target.families.length;
      if (target.fittedNuisanceParameters !== undefined && target.fittedNuisanceParameters !== derivedFitted) {
        throw new Error("multiple_image_systems fittedNuisanceParameters must equal two source coordinates per family");
      }
    } else if (target.kind === "photon_lensing_map") {
      if (!["photons", "both"].includes(definition.target) || definition.rank !== "vector" || definition.unit !== "m/s^2") {
        throw new Error(`observation target ${target.id} requires a photons or both vector in m/s^2`);
      }
      const coordinateSystem = model.geometry.coordinateSystem;
      const axes = [target.northAxis, target.eastAxis, target.lineOfSightAxis];
      const distanceRatio = Number(target.distanceRatio);
      const lensDistance = Number(target.lensAngularDiameterDistanceM);
      if (!Number.isFinite(distanceRatio) || distanceRatio <= 0) throw new Error("distanceRatio must be finite and positive");
      if (!Number.isFinite(lensDistance) || lensDistance <= 0) throw new Error("lensAngularDiameterDistanceM must be finite and positive");
      let projectedShape;
      if (coordinateSystem === "cartesian_3d" && dimensions === 3) {
        if (!axes.every(Number.isInteger) || new Set(axes).size !== 3 || axes.some((axis) => axis < 0 || axis > 2)) {
          throw new Error("northAxis, eastAxis, and lineOfSightAxis must be a permutation of [0,1,2]");
        }
        if (!Array.isArray(fieldShape) || fieldShape.length !== 3 || !fieldShape.every((value) => Number.isInteger(value) && value >= 3)) {
          throw new Error("photon_lensing_map preflight requires the solved 3D field shape");
        }
        if (target.axisymmetricInclinationDeg !== undefined || target.skyShape !== undefined || target.lineOfSightSamples !== undefined) {
          throw new Error("Cartesian photon lensing does not accept axisymmetric projection controls");
        }
        projectedShape = [fieldShape[target.northAxis], fieldShape[target.eastAxis]];
      } else if (coordinateSystem === "axisymmetric_cylindrical" && dimensions === 2) {
        if (axes.some((axis) => axis !== undefined)) {
          throw new Error("axisymmetric photon lensing does not accept Cartesian sky-axis indices");
        }
        if (!Array.isArray(fieldShape) || fieldShape.length !== 2 || !fieldShape.every((value) => Number.isInteger(value) && value >= 3)) {
          throw new Error("axisymmetric photon_lensing_map requires the solved 2D field shape");
        }
        if (JSON.stringify(fieldGeometry?.axisOrder) !== JSON.stringify(["r", "z"])) {
          throw new Error("axisymmetric photon lensing requires field axisOrder=['r','z']");
        }
        const solvedOrigin = finiteArray(fieldGeometry?.origin, 2, "axisymmetric solved-field origin");
        if (solvedOrigin[0] !== 0) throw new Error("axisymmetric photon-lensing radial origin must be exactly r=0");
        let declaredOrigin = null;
        if (target.gridOriginM !== undefined) {
          declaredOrigin = finiteArray(target.gridOriginM, 2, `observation target ${target.id} gridOriginM`);
        } else if (inputBundle.geometry?.coordinateSystem === "axisymmetric_cylindrical") {
          declaredOrigin = finiteArray(inputBundle.geometry.origin, 2, "input bundle origin");
        }
        if (!declaredOrigin) {
          throw new Error("axisymmetric photon lensing requires gridOriginM=[0,z0] when evaluated against a separate observation bundle");
        }
        if (JSON.stringify(declaredOrigin) !== JSON.stringify(solvedOrigin)) {
          throw new Error("axisymmetric photon-lensing target origin must match the solved field");
        }
        const inclination = Number(target.axisymmetricInclinationDeg);
        if (!Number.isFinite(inclination) || inclination < 0 || inclination > 90) {
          throw new Error("axisymmetricInclinationDeg must lie in [0,90]");
        }
        if (!Array.isArray(target.skyShape) || target.skyShape.length !== 2
          || target.skyShape.some((value) => !Number.isInteger(value) || value < 3 || value > 513)) {
          throw new Error("skyShape must contain two integers from 3 through 513");
        }
        if (!Number.isInteger(target.lineOfSightSamples) || target.lineOfSightSamples < 3 || target.lineOfSightSamples > 2049) {
          throw new Error("lineOfSightSamples must be an integer from 3 through 2049");
        }
        if (target.skyShape[0] * target.skyShape[1] * target.lineOfSightSamples > 16_777_216) {
          throw new Error("axisymmetric photon projection exceeds 16,777,216 path samples");
        }
        projectedShape = [...target.skyShape];
      } else {
        throw new Error("photon_lensing_map requires Cartesian 3D or axisymmetric cylindrical geometry");
      }
      const elementCount = projectedShape[0] * projectedShape[1];
      if (target.scoreMaskArrayKey !== undefined) {
        observationArray(inputBundle, target.scoreMaskArrayKey, "scoreMaskArrayKey", { unit: "1", shape: projectedShape });
      }
      const deflectionKeys = [
        target.observedAlphaEastArcsecArrayKey,
        target.observedAlphaNorthArcsecArrayKey,
        target.deflectionUncertaintyArcsecArrayKey,
      ];
      const shearKeys = [
        target.observedReducedShear1ArrayKey,
        target.observedReducedShear2ArrayKey,
        target.reducedShearUncertaintyArrayKey,
      ];
      const completeTriple = (keys, label, unit) => {
        const supplied = keys.filter((key) => key !== undefined).length;
        if (supplied !== 0 && supplied !== 3) throw new Error(`${label} scoring requires both observed component maps and its uncertainty map`);
        if (supplied === 3) keys.forEach((key, index) => observationArray(inputBundle, key, `${label}[${index}]`, { unit, shape: projectedShape }));
        return supplied === 3;
      };
      const deflectionScored = completeTriple(deflectionKeys, "deflection_arcsec", "arcsec");
      const shearScored = completeTriple(shearKeys, "reduced_shear_dimensionless", "1");
      const minimumValid = target.minimumValidPixels ?? 25;
      if (!Number.isInteger(minimumValid) || minimumValid < 1 || minimumValid > elementCount) {
        throw new Error("minimumValidPixels must fit within the photon-lensing map");
      }
      scored = deflectionScored || shearScored;
      pointCount = 2 * elementCount;
    } else {
      if (!["massive_tracers", "both"].includes(definition.target) || definition.rank !== "vector" || definition.unit !== "m/s^2") {
        throw new Error(`observation target ${target.id} requires a massive_tracers or both vector in m/s^2`);
      }
      const coordinateSystem = model.geometry.coordinateSystem;
      if (!["cartesian_2d", "cartesian_3d", "axisymmetric_cylindrical"].includes(coordinateSystem)) {
        throw new Error(`${target.kind} supports Cartesian 2D/3D or axisymmetric cylindrical models`);
      }
      const center = finiteArray(target.centerM, dimensions, `observation target ${target.id} centerM`);
      const axisymmetric = coordinateSystem === "axisymmetric_cylindrical";
      let declaredOrigin = null;
      if (target.gridOriginM !== undefined) {
        declaredOrigin = finiteArray(target.gridOriginM, dimensions, `observation target ${target.id} gridOriginM`);
      } else if (inputBundle.geometry?.origin !== undefined
        && (!axisymmetric || inputBundle.geometry?.coordinateSystem === "axisymmetric_cylindrical")) {
        declaredOrigin = finiteArray(inputBundle.geometry.origin, dimensions, "input bundle origin");
      }
      if (axisymmetric) {
        if (dimensions !== 2) throw new Error("axisymmetric_cylindrical requires dimensions=2");
        if (center[0] !== 0) throw new Error("axisymmetric observation centerM must be [0,z_midplane]");
        if (!declaredOrigin) throw new Error("axisymmetric observation requires gridOriginM=[0,z0] when evaluated against a separate observation bundle");
        if (declaredOrigin[0] !== 0) throw new Error("axisymmetric observation radial origin must be exactly r=0");
        if (target.planeAxes !== undefined) throw new Error("axisymmetric observations do not accept Cartesian planeAxes");
      } else {
        const planeAxes = target.planeAxes ?? [0, 1];
        if (!Array.isArray(planeAxes) || planeAxes.length !== 2 || !planeAxes.every(Number.isInteger)
          || new Set(planeAxes).size !== 2 || planeAxes.some((axis) => axis < 0 || axis >= dimensions)) {
          throw new Error(`observation target ${target.id} planeAxes are invalid`);
        }
      }
      if (target.kind === "circular_speed_curve") {
      const radii = finiteArray(target.radiiM, null, `observation target ${target.id} radiiM`, { positive: true });
      if (radii.some((value, index) => index > 0 && value <= radii[index - 1])) throw new Error(`observation target ${target.id} radiiM must be strictly increasing`);
      if (axisymmetric && target.azimuthalSamples !== undefined) throw new Error("axisymmetric circular_speed_curve does not accept azimuthalSamples");
      const sampleCount = axisymmetric ? null : (target.azimuthalSamples ?? 128);
      if (!axisymmetric && (!Number.isInteger(sampleCount) || sampleCount < 16 || sampleCount > 4096)) throw new Error("azimuthalSamples must be an integer from 16 through 4096");
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
    }
    const fitted = derivedFitted ?? target.fittedNuisanceParameters ?? 0;
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
