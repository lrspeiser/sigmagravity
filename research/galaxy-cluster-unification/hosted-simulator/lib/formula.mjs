import { canonicalize, sha256 } from "./canonical.mjs";

const DIMENSIONLESS = Object.freeze({ L: 0, T: 0, M: 0 });
const INPUT_DIMENSIONS = Object.freeze({
  g_bar: { L: 1, T: -2, M: 0 },
  radius: { L: 1, T: 0, M: 0 },
  surface_density: { L: -2, T: 0, M: 1 },
});
const UNIT_DIMENSIONS = Object.freeze({
  "1": DIMENSIONLESS,
  dimensionless: DIMENSIONLESS,
  "m/s^2": { L: 1, T: -2, M: 0 },
  m: { L: 1, T: 0, M: 0 },
  kg: { L: 0, T: 0, M: 1 },
  "kg/m^2": { L: -2, T: 0, M: 1 },
});
const BINARY = new Set(["add", "sub", "mul", "div", "pow", "min", "max"]);
const UNARY = new Set(["sqrt", "abs", "exp", "log"]);
const MAX_NODES = 128;
const MAX_DEPTH = 24;

function addDimension(a, b, sign = 1) {
  return { L: a.L + sign * b.L, T: a.T + sign * b.T, M: a.M + sign * b.M };
}

function scaleDimension(a, factor) {
  return { L: a.L * factor, T: a.T * factor, M: a.M * factor };
}

function sameDimension(a, b) {
  return a.L === b.L && a.T === b.T && a.M === b.M;
}

function isDimensionless(a) {
  return sameDimension(a, DIMENSIONLESS);
}

function describeDimension(dimension) {
  const terms = Object.entries(dimension)
    .filter(([, exponent]) => exponent !== 0)
    .map(([symbol, exponent]) => `${symbol}^${exponent}`);
  return terms.length ? terms.join(" ") : "dimensionless";
}

function parameterMap(parameters, errors) {
  const mapped = new Map();
  if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) {
    errors.push("parameters must be an object keyed by parameter name");
    return mapped;
  }
  for (const [name, definition] of Object.entries(parameters)) {
    if (!/^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(name)) {
      errors.push(`invalid parameter name: ${name}`);
      continue;
    }
    const unit = definition?.unit;
    const dimension = UNIT_DIMENSIONS[unit];
    if (!dimension) {
      errors.push(`unsupported unit for ${name}: ${unit}`);
      continue;
    }
    if (!Number.isFinite(definition?.value)) {
      errors.push(`parameter ${name} must have a finite numeric value`);
      continue;
    }
    mapped.set(name, { ...definition, dimension });
  }
  return mapped;
}

function inspectNode(node, parameters, state, depth = 0) {
  state.nodes += 1;
  if (state.nodes > MAX_NODES) throw new Error(`formula exceeds ${MAX_NODES} nodes`);
  if (depth > MAX_DEPTH) throw new Error(`formula exceeds depth ${MAX_DEPTH}`);
  if (!node || typeof node !== "object" || Array.isArray(node)) throw new Error("every expression node must be an object");

  if (Object.hasOwn(node, "const")) {
    if (!Number.isFinite(node.const)) throw new Error("const must be finite");
    return { dimension: DIMENSIONLESS, evaluate: () => node.const };
  }
  if (Object.hasOwn(node, "input")) {
    const dimension = INPUT_DIMENSIONS[node.input];
    if (!dimension) throw new Error(`unknown input: ${node.input}`);
    state.inputs.add(node.input);
    return {
      dimension,
      evaluate: (context) => {
        const value = context[node.input];
        if (!Number.isFinite(value)) throw new Error(`input ${node.input} is not finite`);
        return value;
      },
    };
  }
  if (Object.hasOwn(node, "parameter")) {
    const definition = parameters.get(node.parameter);
    if (!definition) throw new Error(`undeclared parameter: ${node.parameter}`);
    state.parameters.add(node.parameter);
    return { dimension: definition.dimension, evaluate: () => definition.value };
  }

  const op = node.op;
  if (BINARY.has(op)) {
    if (!Array.isArray(node.args) || node.args.length !== 2) throw new Error(`${op} requires exactly two args`);
    const left = inspectNode(node.args[0], parameters, state, depth + 1);
    const right = inspectNode(node.args[1], parameters, state, depth + 1);
    if (op === "add" || op === "sub" || op === "min" || op === "max") {
      if (!sameDimension(left.dimension, right.dimension)) {
        throw new Error(`${op} requires matching dimensions, received ${describeDimension(left.dimension)} and ${describeDimension(right.dimension)}`);
      }
      const fn = op === "add" ? (a, b) => a + b : op === "sub" ? (a, b) => a - b : op === "min" ? Math.min : Math.max;
      return { dimension: left.dimension, evaluate: (context) => fn(left.evaluate(context), right.evaluate(context)) };
    }
    if (op === "mul" || op === "div") {
      return {
        dimension: addDimension(left.dimension, right.dimension, op === "mul" ? 1 : -1),
        evaluate: (context) => op === "mul" ? left.evaluate(context) * right.evaluate(context) : left.evaluate(context) / right.evaluate(context),
      };
    }
    if (!isDimensionless(right.dimension) || !Object.hasOwn(node.args[1], "const")) {
      throw new Error("pow exponent must be a dimensionless numeric const");
    }
    const exponent = node.args[1].const;
    return { dimension: scaleDimension(left.dimension, exponent), evaluate: (context) => left.evaluate(context) ** exponent };
  }

  if (UNARY.has(op)) {
    if (!Array.isArray(node.args) || node.args.length !== 1) throw new Error(`${op} requires exactly one arg`);
    const child = inspectNode(node.args[0], parameters, state, depth + 1);
    if (op === "sqrt") {
      return { dimension: scaleDimension(child.dimension, 0.5), evaluate: (context) => Math.sqrt(child.evaluate(context)) };
    }
    if ((op === "exp" || op === "log") && !isDimensionless(child.dimension)) {
      throw new Error(`${op} requires a dimensionless argument`);
    }
    const fn = op === "abs" ? Math.abs : op === "exp" ? Math.exp : Math.log;
    return { dimension: child.dimension, evaluate: (context) => fn(child.evaluate(context)) };
  }
  throw new Error(`unknown or malformed expression node${op ? `: ${op}` : ""}`);
}

export function validateFormula(manifest) {
  const errors = [];
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return { valid: false, errors: ["formula manifest must be an object"] };
  }
  if (typeof manifest.name !== "string" || !manifest.name.trim() || manifest.name.length > 120) {
    errors.push("name must contain 1-120 characters");
  }
  if (manifest.outputUnit !== "m/s^2") errors.push("version 1 formulas must output acceleration in m/s^2");
  const params = parameterMap(manifest.parameters ?? {}, errors);
  const policy = manifest.parameterPolicy ?? {};
  if (policy.universal !== true) errors.push("version 1 requires parameterPolicy.universal=true");
  if ((policy.perObjectParameters ?? 0) !== 0) errors.push("per-object gravity parameters are not accepted in the public benchmark");

  let compiled;
  const state = { nodes: 0, inputs: new Set(), parameters: new Set() };
  if (!manifest.expression) errors.push("expression is required");
  if (!errors.length) {
    try {
      compiled = inspectNode(manifest.expression, params, state);
      if (!sameDimension(compiled.dimension, UNIT_DIMENSIONS[manifest.outputUnit])) {
        errors.push(`expression output is ${describeDimension(compiled.dimension)}, expected acceleration L^1 T^-2`);
      }
    } catch (error) {
      errors.push(error.message);
    }
  }

  const canonicalManifest = canonicalize({
    name: manifest.name?.trim(),
    family: manifest.family ?? "algebraic_acceleration",
    expression: manifest.expression,
    outputUnit: manifest.outputUnit,
    parameters: manifest.parameters ?? {},
    parameterPolicy: manifest.parameterPolicy ?? {},
  });
  return {
    valid: errors.length === 0,
    formulaSha256: sha256(canonicalManifest),
    canonicalManifest,
    dimensionAudit: {
      output: compiled ? describeDimension(compiled.dimension) : null,
      inputs: [...state.inputs].sort(),
    },
    safetyAudit: {
      language: "sigma-ast-v1",
      nodeCount: state.nodes,
      maxNodes: MAX_NODES,
      maxDepth: MAX_DEPTH,
      arbitraryCodeExecuted: false,
    },
    parameterAccounting: {
      universal: params.size,
      perObject: policy.perObjectParameters ?? 0,
      referenced: [...state.parameters].sort(),
    },
    errors,
    evaluate: compiled?.evaluate,
  };
}

export const FIXED_MOND_FORMULA = Object.freeze({
  name: "Fixed simple-MOND comparator",
  family: "algebraic_acceleration",
  outputUnit: "m/s^2",
  parameters: { a0: { value: 1.2e-10, unit: "m/s^2" } },
  parameterPolicy: { universal: true, perObjectParameters: 0 },
  expression: {
    op: "mul",
    args: [
      { const: 0.5 },
      {
        op: "add",
        args: [
          { input: "g_bar" },
          {
            op: "sqrt",
            args: [
              {
                op: "add",
                args: [
                  { op: "pow", args: [{ input: "g_bar" }, { const: 2 }] },
                  {
                    op: "mul",
                    args: [
                      { const: 4 },
                      { op: "mul", args: [{ input: "g_bar" }, { parameter: "a0" }] },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
});

export function compileFormula(manifest) {
  const validation = validateFormula(manifest);
  if (!validation.valid) {
    const error = new Error("formula validation failed");
    error.details = validation.errors;
    throw error;
  }
  return validation;
}
