import { canonicalize, sha256 } from "./canonical.mjs";

const MAX_NODES = 512;
const MAX_DEPTH = 40;
const FIELD_RANKS = new Set(["scalar", "vector", "tensor2"]);
const FIELD_ROLES = new Set(["source", "solved", "derived"]);
const TARGETS = new Set(["massive_tracers", "photons", "both", "diagnostic"]);
const MODEL_CLASSES = new Set(["stationary_elliptic", "algebraic", "nonlocal_stationary"]);
const COORDINATES = new Map([
  ["cartesian_2d", 2],
  ["cartesian_3d", 3],
  ["axisymmetric_cylindrical", 2],
]);
const BOUNDARIES = new Set(["dirichlet", "neumann", "periodic", "isolated", "mixed"]);
const PARAMETER_MODES = new Set([
  "published_fixed",
  "universal_fixed",
  "universal_fit",
  "train_validation_holdout",
  "hierarchical",
  "per_object",
]);
const SOLVER_FAMILIES = new Set([
  "algebraic",
  "fft_poisson",
  "finite_difference_elliptic",
  "finite_volume_elliptic",
  "nonlinear_finite_volume",
  "coupled_elliptic",
  "nonlocal_elliptic",
]);
const SOURCE_FORMATS = new Set(["latex", "plain_text", "json_ast"]);

const D0 = Object.freeze({ M: 0, L: 0, T: 0 });
const UNIT_DIMENSIONS = Object.freeze({
  "1": D0,
  dimensionless: D0,
  kg: { M: 1, L: 0, T: 0 },
  m: { M: 0, L: 1, T: 0 },
  s: { M: 0, L: 0, T: 1 },
  "1/m": { M: 0, L: -1, T: 0 },
  "1/s^2": { M: 0, L: 0, T: -2 },
  "m/s": { M: 0, L: 1, T: -1 },
  "m/s^2": { M: 0, L: 1, T: -2 },
  "m^2/s^2": { M: 0, L: 2, T: -2 },
  "kg/m^2": { M: 1, L: -2, T: 0 },
  "kg/m^3": { M: 1, L: -3, T: 0 },
  "m^3/(kg*s^2)": { M: -1, L: 3, T: -2 },
});

function dimension(unit) {
  const value = UNIT_DIMENSIONS[unit];
  if (!value) throw new Error(`unsupported unit: ${unit}`);
  return value;
}

function addDimension(a, b, sign = 1) {
  return { M: a.M + sign * b.M, L: a.L + sign * b.L, T: a.T + sign * b.T };
}

function scaleDimension(value, factor) {
  return { M: value.M * factor, L: value.L * factor, T: value.T * factor };
}

function sameDimension(a, b) {
  return a.M === b.M && a.L === b.L && a.T === b.T;
}

function sameType(a, b) {
  return a.rank === b.rank && sameDimension(a.dimension, b.dimension);
}

function describeType(type) {
  const terms = Object.entries(type.dimension)
    .filter(([, exponent]) => exponent !== 0)
    .map(([symbol, exponent]) => `${symbol}^${exponent}`);
  return `${type.rank} ${terms.length ? terms.join(" ") : "dimensionless"}`;
}

function scalar(dimensionValue = D0) {
  return { rank: "scalar", dimension: dimensionValue };
}

function assertArgs(node, count) {
  if (!Array.isArray(node.args) || node.args.length !== count) {
    throw new Error(`${node.op} requires exactly ${count} argument${count === 1 ? "" : "s"}`);
  }
}

function inferExpression(node, context, state, depth = 0) {
  state.nodeCount += 1;
  if (state.nodeCount > MAX_NODES) throw new Error(`model exceeds ${MAX_NODES} expression nodes`);
  if (depth > MAX_DEPTH) throw new Error(`model exceeds expression depth ${MAX_DEPTH}`);
  if (!node || typeof node !== "object" || Array.isArray(node)) throw new Error("expression nodes must be objects");

  if (Object.hasOwn(node, "const")) {
    if (!Number.isFinite(node.const)) throw new Error("const must be finite");
    return scalar(dimension(node.unit ?? "1"));
  }
  if (Object.hasOwn(node, "field")) {
    const field = context.fields.get(node.field);
    if (!field) throw new Error(`unknown field: ${node.field}`);
    state.fields.add(node.field);
    return { rank: field.rank, dimension: field.dimension };
  }
  if (Object.hasOwn(node, "parameter")) {
    const parameter = context.parameters.get(node.parameter);
    if (!parameter) throw new Error(`unknown parameter: ${node.parameter}`);
    state.parameters.add(node.parameter);
    return scalar(parameter.dimension);
  }

  const op = node.op;
  state.operators.add(op);
  if (["add", "subtract", "min", "max"].includes(op)) {
    assertArgs(node, 2);
    const left = inferExpression(node.args[0], context, state, depth + 1);
    const right = inferExpression(node.args[1], context, state, depth + 1);
    if (!sameType(left, right)) throw new Error(`${op} type mismatch: ${describeType(left)} versus ${describeType(right)}`);
    return left;
  }
  if (op === "multiply" || op === "multiply_zero_vector_limit") {
    if (!Array.isArray(node.args) || node.args.length < 2) throw new Error(`${op} requires at least 2 arguments`);
    if (op === "multiply_zero_vector_limit" && node.args.length !== 2) {
      throw new Error("multiply_zero_vector_limit requires exactly 2 arguments");
    }
    const factors = node.args.map((argument) => inferExpression(argument, context, state, depth + 1));
    const nonScalars = factors.filter((factor) => factor.rank !== "scalar");
    if (nonScalars.length > 1) throw new Error(`${op} accepts at most one vector or tensor; use dot or outer for vector/tensor products`);
    if (op === "multiply_zero_vector_limit" && nonScalars[0]?.rank !== "vector") {
      throw new Error("multiply_zero_vector_limit requires exactly one vector and one scalar");
    }
    return {
      rank: nonScalars[0]?.rank ?? "scalar",
      dimension: factors.reduce((total, factor) => addDimension(total, factor.dimension), D0),
    };
  }
  if (op === "divide") {
    assertArgs(node, 2);
    const numerator = inferExpression(node.args[0], context, state, depth + 1);
    const denominator = inferExpression(node.args[1], context, state, depth + 1);
    if (denominator.rank !== "scalar") throw new Error("divide denominator must be scalar");
    return { rank: numerator.rank, dimension: addDimension(numerator.dimension, denominator.dimension, -1) };
  }
  if (op === "negate" || op === "norm") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    return op === "norm" ? scalar(child.dimension) : child;
  }
  if (op === "gradient") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    const nextRank = child.rank === "scalar" ? "vector" : child.rank === "vector" ? "tensor2" : null;
    if (!nextRank) throw new Error("gradient of tensor2 is outside stationary-field v1");
    return { rank: nextRank, dimension: addDimension(child.dimension, { M: 0, L: -1, T: 0 }) };
  }
  if (op === "divergence") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    const nextRank = child.rank === "vector" ? "scalar" : child.rank === "tensor2" ? "vector" : null;
    if (!nextRank) throw new Error("divergence requires a vector or tensor2");
    return { rank: nextRank, dimension: addDimension(child.dimension, { M: 0, L: -1, T: 0 }) };
  }
  if (op === "laplacian") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    return { rank: child.rank, dimension: addDimension(child.dimension, { M: 0, L: -2, T: 0 }) };
  }
  if (op === "curl") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    if (context.dimensions !== 3 || child.rank !== "vector") throw new Error("curl requires a vector in cartesian_3d geometry");
    return { rank: "vector", dimension: addDimension(child.dimension, { M: 0, L: -1, T: 0 }) };
  }
  if (op === "dot" || op === "outer") {
    assertArgs(node, 2);
    const left = inferExpression(node.args[0], context, state, depth + 1);
    const right = inferExpression(node.args[1], context, state, depth + 1);
    if (left.rank !== "vector" || right.rank !== "vector") throw new Error(`${op} requires two vectors`);
    return { rank: op === "dot" ? "scalar" : "tensor2", dimension: addDimension(left.dimension, right.dimension) };
  }
  if (op === "trace") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    if (child.rank !== "tensor2") throw new Error("trace requires tensor2");
    return scalar(child.dimension);
  }
  if (op === "pow") {
    assertArgs(node, 2);
    const base = inferExpression(node.args[0], context, state, depth + 1);
    if (base.rank !== "scalar" || !Object.hasOwn(node.args[1], "const") || node.args[1].unit) {
      throw new Error("pow requires a scalar base and a dimensionless constant exponent");
    }
    inferExpression(node.args[1], context, state, depth + 1);
    return scalar(scaleDimension(base.dimension, node.args[1].const));
  }
  if (op === "sqrt") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    if (child.rank !== "scalar") throw new Error("sqrt requires a scalar");
    return scalar(scaleDimension(child.dimension, 0.5));
  }
  if (["exp", "log", "tanh", "smoothstep"].includes(op)) {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    if (child.rank !== "scalar" || !sameDimension(child.dimension, D0)) throw new Error(`${op} requires a dimensionless scalar`);
    return scalar();
  }
  if (["lt", "lte", "gt", "gte"].includes(op)) {
    assertArgs(node, 2);
    const left = inferExpression(node.args[0], context, state, depth + 1);
    const right = inferExpression(node.args[1], context, state, depth + 1);
    if (!sameType(left, right) || left.rank !== "scalar") throw new Error(`${op} requires matching scalars`);
    return scalar();
  }
  if (op === "piecewise") {
    if (!Array.isArray(node.branches) || node.branches.length === 0 || !node.otherwise) throw new Error("piecewise requires branches and otherwise");
    const output = inferExpression(node.otherwise, context, state, depth + 1);
    for (const branch of node.branches) {
      const condition = inferExpression(branch.when, context, state, depth + 1);
      const value = inferExpression(branch.value, context, state, depth + 1);
      if (condition.rank !== "scalar" || !sameDimension(condition.dimension, D0)) throw new Error("piecewise conditions must be dimensionless scalars");
      if (!sameType(output, value)) throw new Error("piecewise branches must have matching types");
    }
    return output;
  }
  if (op === "line_of_sight_integral") {
    assertArgs(node, 1);
    const child = inferExpression(node.args[0], context, state, depth + 1);
    return { rank: child.rank, dimension: addDimension(child.dimension, { M: 0, L: 1, T: 0 }) };
  }
  if (op === "convolution") {
    assertArgs(node, 2);
    const field = inferExpression(node.args[0], context, state, depth + 1);
    const kernel = inferExpression(node.args[1], context, state, depth + 1);
    if (kernel.rank !== "scalar") throw new Error("convolution kernel must be scalar in v1");
    return { rank: field.rank, dimension: addDimension(addDimension(field.dimension, kernel.dimension), { M: 0, L: context.dimensions, T: 0 }) };
  }
  throw new Error(`unsupported field operator: ${op ?? "missing"}`);
}

function validateName(name, kind) {
  if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name)) throw new Error(`invalid ${kind} name: ${name}`);
}

function computationalManifest(manifest) {
  return canonicalize({
    schemaVersion: manifest.schemaVersion,
    modelClass: manifest.modelClass,
    geometry: manifest.geometry,
    fields: manifest.fields,
    parameters: manifest.parameters,
    equations: manifest.equations,
    observables: manifest.observables,
    dataRequirements: manifest.dataRequirements,
    solver: manifest.solver,
    parameterPolicy: manifest.parameterPolicy,
  });
}

export function validateFieldModel(manifest) {
  const errors = [];
  const warnings = [];
  const fields = new Map();
  const parameters = new Map();
  const state = { nodeCount: 0, fields: new Set(), parameters: new Set(), operators: new Set() };
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return { valid: false, errors: ["model manifest must be an object"], warnings };
  }
  if (manifest.schemaVersion !== "sigma-field-model/1") errors.push("schemaVersion must be sigma-field-model/1");
  if (typeof manifest.name !== "string" || !manifest.name.trim() || manifest.name.length > 160) errors.push("name must contain 1-160 characters");
  if (!MODEL_CLASSES.has(manifest.modelClass)) errors.push(`unsupported modelClass: ${manifest.modelClass}`);
  if (!SOURCE_FORMATS.has(manifest.source?.format) || typeof manifest.source?.text !== "string" || !manifest.source.text.trim()) errors.push("source requires a supported format and non-empty text");
  if (typeof manifest.source?.confirmedCanonical !== "boolean") errors.push("source.confirmedCanonical must be boolean");
  const coordinateSystem = manifest.geometry?.coordinateSystem;
  const dimensions = COORDINATES.get(coordinateSystem);
  if (!dimensions) errors.push(`unsupported coordinate system: ${coordinateSystem}`);
  if (dimensions && manifest.geometry?.dimensions !== dimensions) errors.push(`${coordinateSystem} requires dimensions=${dimensions}`);
  if (!manifest.geometry?.domain || manifest.geometry.domain.lengthUnit !== "m") errors.push("geometry.domain with lengthUnit=m is required");

  if (!manifest.fields || typeof manifest.fields !== "object" || Array.isArray(manifest.fields)) errors.push("fields must be an object");
  else for (const [name, definition] of Object.entries(manifest.fields)) {
    try {
      validateName(name, "field");
      if (!FIELD_RANKS.has(definition.rank)) throw new Error(`field ${name} has unsupported rank`);
      if (!FIELD_ROLES.has(definition.role)) throw new Error(`field ${name} has unsupported role`);
      const fieldDimension = dimension(definition.unit);
      if (definition.role === "source" && !definition.datasetKey) throw new Error(`source field ${name} requires datasetKey`);
      if (definition.role === "solved" && !BOUNDARIES.has(definition.boundary?.type)) throw new Error(`solved field ${name} requires a supported boundary`);
      fields.set(name, { ...definition, dimension: fieldDimension });
    } catch (error) { errors.push(error.message); }
  }

  if (!manifest.parameters || typeof manifest.parameters !== "object" || Array.isArray(manifest.parameters)) errors.push("parameters must be an object");
  else for (const [name, definition] of Object.entries(manifest.parameters)) {
    try {
      validateName(name, "parameter");
      const parameterDimension = dimension(definition.unit);
      if (!['universal', 'per_object'].includes(definition.scope)) throw new Error(`parameter ${name} requires scope universal or per_object`);
      if (!Number.isFinite(definition.value) && !Number.isFinite(definition.initial)) throw new Error(`parameter ${name} requires a finite value or initial`);
      if (definition.bounds && (!Array.isArray(definition.bounds) || definition.bounds.length !== 2 || !definition.bounds.every(Number.isFinite) || definition.bounds[0] >= definition.bounds[1])) throw new Error(`parameter ${name} has invalid bounds`);
      parameters.set(name, { ...definition, dimension: parameterDimension });
    } catch (error) { errors.push(error.message); }
  }

  const context = { fields, parameters, dimensions: dimensions ?? 0 };
  const equationIds = new Set();
  if (!Array.isArray(manifest.equations) || manifest.equations.length === 0) errors.push("at least one equation is required");
  else for (const equation of manifest.equations) {
    try {
      validateName(equation.id, "equation");
      if (equationIds.has(equation.id)) throw new Error(`duplicate equation id: ${equation.id}`);
      equationIds.add(equation.id);
      if (equation.kind !== "equality") throw new Error(`equation ${equation.id} must use kind=equality`);
      const left = inferExpression(equation.lhs, context, state);
      const right = inferExpression(equation.rhs, context, state);
      if (!sameType(left, right)) throw new Error(`equation ${equation.id} mismatch: ${describeType(left)} versus ${describeType(right)}`);
    } catch (error) { errors.push(error.message); }
  }

  const observableIds = new Set();
  if (!Array.isArray(manifest.observables) || manifest.observables.length === 0) errors.push("at least one observable is required");
  else for (const observable of manifest.observables) {
    try {
      validateName(observable.id, "observable");
      if (observableIds.has(observable.id)) throw new Error(`duplicate observable id: ${observable.id}`);
      observableIds.add(observable.id);
      if (!TARGETS.has(observable.target)) throw new Error(`observable ${observable.id} has unsupported target`);
      const output = inferExpression(observable.expression, context, state);
      if (observable.unit && !sameDimension(output.dimension, dimension(observable.unit))) throw new Error(`observable ${observable.id} output is ${describeType(output)}, not ${observable.unit}`);
      if (observable.rank && observable.rank !== output.rank) throw new Error(`observable ${observable.id} rank is ${output.rank}, not ${observable.rank}`);
    } catch (error) { errors.push(error.message); }
  }

  if (!Array.isArray(manifest.dataRequirements)) errors.push("dataRequirements must be an array");
  const requirements = Array.isArray(manifest.dataRequirements) ? manifest.dataRequirements : [];
  const requirementKeys = new Set(requirements.map((item) => item.key));
  for (const [name, field] of fields) {
    if (field.role === "source" && !requirementKeys.has(field.datasetKey)) errors.push(`source field ${name} datasetKey ${field.datasetKey} is absent from dataRequirements`);
  }
  const requirementsByKey = new Map();
  for (const item of requirements) {
    try {
      validateName(item.key, "data requirement");
      if (requirementsByKey.has(item.key)) throw new Error(`duplicate data requirement: ${item.key}`);
      if (!FIELD_RANKS.has(item.rank)) throw new Error(`data requirement ${item.key} has unsupported rank`);
      const itemDimension = dimension(item.unit);
      requirementsByKey.set(item.key, { ...item, dimension: itemDimension });
    } catch (error) { errors.push(error.message); }
  }
  for (const [name, field] of fields) {
    if (field.role !== "source") continue;
    const requirement = requirementsByKey.get(field.datasetKey);
    if (requirement && (requirement.rank !== field.rank || !sameDimension(requirement.dimension, field.dimension))) {
      errors.push(`source field ${name} does not match data requirement ${field.datasetKey} rank and unit`);
    }
  }

  const policy = manifest.parameterPolicy ?? {};
  if (!PARAMETER_MODES.has(policy.mode)) errors.push(`unsupported parameterPolicy.mode: ${policy.mode}`);
  const declaredPerObject = new Set(policy.perObjectParameters ?? []);
  const actualPerObject = new Set([...parameters].filter(([, value]) => value.scope === "per_object").map(([name]) => name));
  if (!Array.isArray(policy.perObjectParameters)) errors.push("parameterPolicy.perObjectParameters must be an array");
  if (Array.isArray(policy.perObjectParameters) && declaredPerObject.size !== policy.perObjectParameters.length) errors.push("parameterPolicy.perObjectParameters must not contain duplicates");
  if ([...declaredPerObject].some((name) => !actualPerObject.has(name)) || [...actualPerObject].some((name) => !declaredPerObject.has(name))) errors.push("parameterPolicy.perObjectParameters must exactly match parameters with scope=per_object");
  if (actualPerObject.size > 0) warnings.push(`${actualPerObject.size} per-object model parameter(s) must be disclosed separately from universal settings`);
  if (!SOLVER_FAMILIES.has(manifest.solver?.family)) errors.push(`unsupported solver.family: ${manifest.solver?.family}`);
  if (!(manifest.solver?.relativeTolerance > 0 && manifest.solver.relativeTolerance < 1)) errors.push("solver.relativeTolerance must lie between 0 and 1");
  if (manifest.solver?.residualTolerance !== undefined && !(manifest.solver.residualTolerance > 0 && manifest.solver.residualTolerance < 1)) errors.push("solver.residualTolerance must lie between 0 and 1");
  if (!Number.isInteger(manifest.solver?.maxIterations) || manifest.solver.maxIterations < 1) errors.push("solver.maxIterations must be a positive integer");
  if (manifest.solver?.damping !== undefined && !(manifest.solver.damping > 0 && manifest.solver.damping <= 1)) errors.push("solver.damping must lie in (0,1]");
  if (manifest.solver?.coefficientFloor !== undefined && !(manifest.solver.coefficientFloor > 0)) errors.push("solver.coefficientFloor must be positive");
  if (manifest.solver?.initialization !== undefined && !["zero", "linearized_unit_coefficient"].includes(manifest.solver.initialization)) errors.push("solver.initialization is unsupported");
  if (manifest.solver?.nonlinearMethod !== undefined && !["picard", "anderson", "newton_krylov"].includes(manifest.solver.nonlinearMethod)) errors.push("solver.nonlinearMethod is unsupported");
  if (manifest.solver?.lineSearch !== undefined && !["armijo", "wolfe", "none"].includes(manifest.solver.lineSearch)) errors.push("solver.lineSearch is unsupported");
  if (manifest.solver?.andersonAlpha !== undefined && !(manifest.solver.andersonAlpha > 0)) errors.push("solver.andersonAlpha must be positive");
  if (manifest.solver?.andersonHistory !== undefined && (!Number.isInteger(manifest.solver.andersonHistory) || manifest.solver.andersonHistory < 1 || manifest.solver.andersonHistory > 20)) errors.push("solver.andersonHistory must be an integer from 1 to 20");
  if (manifest.solver?.andersonRegularization !== undefined && !(manifest.solver.andersonRegularization > 0)) errors.push("solver.andersonRegularization must be positive");
  if (manifest.solver?.krylovMethod !== undefined && !["lgmres", "gmres", "bicgstab", "cgs", "minres", "tfqmr"].includes(manifest.solver.krylovMethod)) errors.push("solver.krylovMethod is unsupported");
  if (manifest.solver?.krylovInnerIterations !== undefined && (!Number.isInteger(manifest.solver.krylovInnerIterations) || manifest.solver.krylovInnerIterations < 1 || manifest.solver.krylovInnerIterations > 200)) errors.push("solver.krylovInnerIterations must be an integer from 1 to 200");
  if (manifest.solver?.picardWarmupIterations !== undefined && (!Number.isInteger(manifest.solver.picardWarmupIterations) || manifest.solver.picardWarmupIterations < 0 || manifest.solver.picardWarmupIterations >= manifest.solver.maxIterations)) errors.push("solver.picardWarmupIterations must be a non-negative integer below maxIterations");
  if (manifest.solver?.picardWarmupDamping !== undefined && !(manifest.solver.picardWarmupDamping > 0 && manifest.solver.picardWarmupDamping <= 1)) errors.push("solver.picardWarmupDamping must lie in (0,1]");
  if (manifest.solver?.maxIterations > 200) warnings.push("the current preview worker executes at most 200 nonlinear iterations and records the requested and effective limits");

  for (const [name, field] of fields) {
    if (field.role === "solved" && !state.fields.has(name)) errors.push(`solved field ${name} is not referenced by an equation or observable`);
  }
  for (const name of parameters.keys()) {
    if (!state.parameters.has(name)) warnings.push(`declared parameter ${name} is unused`);
  }
  if (manifest.source?.format === "latex" && !manifest.source?.confirmedCanonical) warnings.push("LaTeX source is informational until the researcher confirms the canonical manifest");

  const computational = computationalManifest(manifest);
  return {
    valid: errors.length === 0,
    modelSha256: sha256(computational),
    documentSha256: sha256(canonicalize(manifest)),
    canonicalManifest: canonicalize(manifest),
    typeAudit: {
      expressionNodes: state.nodeCount,
      maximumNodes: MAX_NODES,
      maximumDepth: MAX_DEPTH,
      fieldsReferenced: [...state.fields].sort(),
      parametersReferenced: [...state.parameters].sort(),
      operators: [...state.operators].sort(),
    },
    parameterAccounting: {
      universal: [...parameters.values()].filter((item) => item.scope === "universal").length,
      perObject: actualPerObject.size,
      mode: policy.mode ?? null,
    },
    requiredCapabilities: {
      modelClass: manifest.modelClass ?? null,
      coordinateSystem: coordinateSystem ?? null,
      dimensions: dimensions ?? null,
      solverFamily: manifest.solver?.family ?? null,
      operators: [...state.operators].sort(),
      dataKeys: requirements.map((item) => item.key).sort(),
    },
    executionReadiness: {
      state: errors.length ? "invalid" : "worker_not_connected",
      blockers: errors.length ? ["manifest_validation_failed"] : ["generic_scientific_worker_not_connected"],
    },
    errors,
    warnings,
  };
}

export { UNIT_DIMENSIONS };
