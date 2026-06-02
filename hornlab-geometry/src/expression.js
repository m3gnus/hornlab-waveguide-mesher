import { isDevRuntime } from './config/runtimeMode.js';

const CONSTANTS = Object.freeze({
  pi: Math.PI,
  e: Math.E,
});

const FUNCTION_TABLE = Object.freeze({
  sin: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.sin(x) },
  cos: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.cos(x) },
  tan: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.tan(x) },
  asin: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.asin(x) },
  acos: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.acos(x) },
  atan: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.atan(x) },
  atan2: { minArgs: 2, maxArgs: 2, call: ([y, x]) => Math.atan2(y, x) },
  sinh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.sinh(x) },
  cosh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.cosh(x) },
  tanh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.tanh(x) },
  asinh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.asinh(x) },
  acosh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.acosh(x) },
  atanh: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.atanh(x) },
  exp: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.exp(x) },
  expm1: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.expm1(x) },
  ln: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log(x) },
  log: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log10(x) },
  log2: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log2(x) },
  log10: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log10(x) },
  log1p: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log1p(x) },
  exp2: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.pow(2, x) },
  pow: { minArgs: 2, maxArgs: 2, call: ([x, y]) => Math.pow(x, y) },
  sqrt: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.sqrt(x) },
  cbrt: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.cbrt(x) },
  hypot: { minArgs: 1, maxArgs: null, call: (args) => Math.hypot(...args) },
  ceil: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.ceil(x) },
  floor: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.floor(x) },
  round: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.round(x) },
  trunc: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.trunc(x) },
  abs: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.abs(x) },
  fabs: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.abs(x) },
  sign: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.sign(x) },
  min: { minArgs: 1, maxArgs: null, call: (args) => Math.min(...args) },
  max: { minArgs: 1, maxArgs: null, call: (args) => Math.max(...args) },
  fmod: { minArgs: 2, maxArgs: 2, call: ([x, y]) => x % y },
  remainder: {
    minArgs: 2,
    maxArgs: 2,
    call: ([x, y]) => x - Math.round(x / y) * y,
  },
  copysign: {
    minArgs: 2,
    maxArgs: 2,
    call: ([x, y]) => Math.sign(y) * Math.abs(x),
  },
  fdim: { minArgs: 2, maxArgs: 2, call: ([x, y]) => Math.max(x - y, 0) },
  fma: { minArgs: 3, maxArgs: 3, call: ([x, y, z]) => x * y + z },
  fmin: { minArgs: 2, maxArgs: 2, call: ([x, y]) => Math.min(x, y) },
  fmax: { minArgs: 2, maxArgs: 2, call: ([x, y]) => Math.max(x, y) },
  nearbyint: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.round(x) },
  deg: { minArgs: 1, maxArgs: 1, call: ([x]) => x * 180 / Math.PI },
  rad: { minArgs: 1, maxArgs: 1, call: ([x]) => x * Math.PI / 180 },
  erf: {
    minArgs: 1,
    maxArgs: 1,
    call: ([x]) => {
      const t = 1 / (1 + 0.3275911 * Math.abs(x));
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
      return x < 0 ? -y : y;
    },
  },
  erfc: { minArgs: 1, maxArgs: 1, call: ([x]) => 1 - FUNCTION_TABLE.erf.call([x]) },
  gamma: {
    minArgs: 1,
    maxArgs: 1,
    call: ([z]) => {
      if (z < 0.5) {
        return Math.PI / (Math.sin(Math.PI * z) * FUNCTION_TABLE.gamma.call([1 - z]));
      }
      z -= 1;
      const g = 7;
      const coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
      ];
      let x = coeffs[0];
      for (let i = 1; i < g + 2; i += 1) {
        x += coeffs[i] / (z + i);
      }
      const t = z + g + 0.5;
      return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
    },
  },
  lgamma: { minArgs: 1, maxArgs: 1, call: ([x]) => Math.log(Math.abs(FUNCTION_TABLE.gamma.call([x]))) },
});

const ZERO_ARG_CONSTANTS = new Set(['pi', 'e']);

function normalizeSource(expr) {
  return expr.toLowerCase().replace(/\bmath\./g, '').trim();
}

function tokenize(source) {
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const ch = source[index];

    if (/\s/.test(ch)) {
      index += 1;
      continue;
    }

    if (/[0-9.]/.test(ch)) {
      const start = index;
      let seenDot = false;
      let seenDigit = false;

      while (index < source.length) {
        const cur = source[index];
        if (/[0-9]/.test(cur)) {
          seenDigit = true;
          index += 1;
          continue;
        }
        if (cur === '.' && !seenDot) {
          seenDot = true;
          index += 1;
          continue;
        }
        break;
      }

      if (!seenDigit) {
        throw new SyntaxError(`Invalid numeric literal near "${source.slice(start)}"`);
      }

      if (source[index] === 'e' || source[index] === 'E') {
        const expStart = index;
        index += 1;
        if (source[index] === '+' || source[index] === '-') {
          index += 1;
        }
        const expDigitsStart = index;
        while (index < source.length && /[0-9]/.test(source[index])) {
          index += 1;
        }
        if (expDigitsStart === index) {
          throw new SyntaxError(`Invalid numeric literal near "${source.slice(expStart)}"`);
        }
      }

      const raw = source.slice(start, index);
      tokens.push({ type: 'number', value: Number(raw) });
      continue;
    }

    if (/[a-z_]/.test(ch)) {
      const start = index;
      index += 1;
      while (index < source.length && /[a-z0-9_]/.test(source[index])) {
        index += 1;
      }
      tokens.push({ type: 'identifier', value: source.slice(start, index) });
      continue;
    }

    if (ch === '+') {
      tokens.push({ type: 'operator', value: '+' });
      index += 1;
      continue;
    }
    if (ch === '-') {
      tokens.push({ type: 'operator', value: '-' });
      index += 1;
      continue;
    }
    if (ch === '*') {
      tokens.push({ type: 'operator', value: '*' });
      index += 1;
      continue;
    }
    if (ch === '/') {
      tokens.push({ type: 'operator', value: '/' });
      index += 1;
      continue;
    }
    if (ch === '^') {
      tokens.push({ type: 'operator', value: '^' });
      index += 1;
      continue;
    }
    if (ch === '(') {
      tokens.push({ type: 'lparen', value: '(' });
      index += 1;
      continue;
    }
    if (ch === ')') {
      tokens.push({ type: 'rparen', value: ')' });
      index += 1;
      continue;
    }
    if (ch === ',') {
      tokens.push({ type: 'comma', value: ',' });
      index += 1;
      continue;
    }

    throw new SyntaxError(`Unexpected token "${ch}"`);
  }

  tokens.push({ type: 'eof', value: null });
  return tokens;
}

// Recursive-descent guard: parseAdditive → parseMultiplicative →
// parseUnary → parsePower → parsePrimary recurse back into parseAdditive
// on each '(' or function call.  A pathological expression like
// (((...((1))...))) could overflow the JS call stack; cap nesting at a
// value that comfortably supports legitimate formulas (our deepest
// built-in reference expressions nest ~6 levels).
const MAX_NESTING_DEPTH = 64;

class ExpressionParser {
  constructor(tokens) {
    this.tokens = tokens;
    this.index = 0;
    this.depth = 0;
  }

  enterNested() {
    this.depth += 1;
    if (this.depth > MAX_NESTING_DEPTH) {
      throw new SyntaxError(
        `Expression nesting depth exceeds limit of ${MAX_NESTING_DEPTH}`,
      );
    }
  }

  exitNested() {
    this.depth -= 1;
  }

  peek(offset = 0) {
    return this.tokens[this.index + offset] || this.tokens[this.tokens.length - 1];
  }

  consume(type = null, value = null) {
    const token = this.peek();
    if (type && token.type !== type) {
      throw new SyntaxError(`Expected ${type} but found ${token.type}`);
    }
    if (value !== null && token.value !== value) {
      throw new SyntaxError(`Expected "${value}" but found "${token.value}"`);
    }
    this.index += 1;
    return token;
  }

  parse() {
    const value = this.parseAdditive();
    if (this.peek().type !== 'eof') {
      throw new SyntaxError(`Unexpected token "${this.peek().value}"`);
    }
    return value;
  }

  parseAdditive() {
    let value = this.parseMultiplicative();
    while (true) {
      const token = this.peek();
      if (token.type !== 'operator' || (token.value !== '+' && token.value !== '-')) {
        break;
      }
      this.consume('operator');
      const right = this.parseMultiplicative();
      value = token.value === '+' ? value + right : value - right;
    }
    return value;
  }

  parseMultiplicative() {
    let value = this.parseUnary();
    while (true) {
      const token = this.peek();
      if (token.type === 'operator' && (token.value === '*' || token.value === '/')) {
        this.consume('operator');
        const right = this.parseUnary();
        value = token.value === '*' ? value * right : value / right;
        continue;
      }
      if (this.canStartPrimary(token)) {
        const right = this.parseUnary();
        value *= right;
        continue;
      }
      break;
    }
    return value;
  }

  parseUnary() {
    const token = this.peek();
    if (token.type === 'operator' && (token.value === '+' || token.value === '-')) {
      this.consume('operator');
      const value = this.parseUnary();
      return token.value === '-' ? -value : value;
    }
    return this.parsePower();
  }

  parsePower() {
    let value = this.parsePrimary();
    if (this.peek().type === 'operator' && this.peek().value === '^') {
      this.consume('operator');
      const exponent = this.parseUnary();
      value = Math.pow(value, exponent);
    }
    return value;
  }

  parsePrimary() {
    const token = this.peek();

    if (token.type === 'number') {
      this.consume('number');
      return token.value;
    }

    if (token.type === 'identifier') {
      const name = token.value;
      const next = this.peek(1);

      if (FUNCTION_TABLE[name] && next.type === 'lparen') {
        return this.parseFunctionCall(name);
      }

      if (ZERO_ARG_CONSTANTS.has(name) && next.type === 'lparen') {
        return this.parseZeroArgConstantCall(name);
      }

      if (name === 'p') {
        this.consume('identifier');
        return this.getVariableValue(name);
      }

      if (Object.prototype.hasOwnProperty.call(CONSTANTS, name)) {
        this.consume('identifier');
        return CONSTANTS[name];
      }

      throw new SyntaxError(`Unknown identifier "${name}"`);
    }

    if (token.type === 'lparen') {
      this.consume('lparen');
      this.enterNested();
      try {
        const value = this.parseAdditive();
        this.consume('rparen');
        return value;
      } finally {
        this.exitNested();
      }
    }

    throw new SyntaxError(`Unexpected token "${token.value}"`);
  }

  parseFunctionCall(name) {
    const fn = FUNCTION_TABLE[name];
    this.consume('identifier');
    this.consume('lparen');
    this.enterNested();

    const args = [];
    try {
      if (this.peek().type !== 'rparen') {
        while (true) {
          args.push(this.parseAdditive());
          if (this.peek().type === 'comma') {
            this.consume('comma');
            continue;
          }
          break;
        }
      }

      this.consume('rparen');
    } finally {
      this.exitNested();
    }
    if (args.length < fn.minArgs) {
      throw new SyntaxError(`Function "${name}" expects at least ${fn.minArgs} argument(s)`);
    }
    if (fn.maxArgs !== null && args.length > fn.maxArgs) {
      throw new SyntaxError(`Function "${name}" expects at most ${fn.maxArgs} argument(s)`);
    }
    return fn.call(args);
  }

  parseZeroArgConstantCall(name) {
    this.consume('identifier');
    this.consume('lparen');
    if (this.peek().type !== 'rparen') {
      throw new SyntaxError(`Function "${name}" does not accept arguments`);
    }
    this.consume('rparen');
    return CONSTANTS[name];
  }

  canStartPrimary(token) {
    return token.type === 'number' || token.type === 'identifier' || token.type === 'lparen';
  }

  getVariableValue(name) {
    if (name === 'p') {
      return this.currentP;
    }
    throw new SyntaxError(`Unknown variable "${name}"`);
  }

  evaluate(currentP) {
    this.currentP = currentP;
    return this.parse();
  }
}

// Tracks expressions that have already emitted an evaluation-error warning.
// `parseExpression` is hot-called inside profile sweeps; without dedup, a
// broken expression at one angle floods the console with hundreds of
// identical lines per build.  WeakSet on functions can't key by the raw
// string, so we use a Set capped at a reasonable size.
const _warnedExpressions = new Set();
const _MAX_WARNED_CACHE = 256;

function _warnOnce(expr, where, error) {
  const key = `${where}::${expr}`;
  if (_warnedExpressions.has(key)) return;
  if (_warnedExpressions.size >= _MAX_WARNED_CACHE) {
    // Drop the oldest entry when the cache fills up — keeps memory bounded
    // without losing the dedup benefit for the active build.
    const first = _warnedExpressions.values().next().value;
    if (first !== undefined) _warnedExpressions.delete(first);
  }
  _warnedExpressions.add(key);
  // eslint-disable-next-line no-console
  console.warn(`[parseExpression] ${where} failed for ${JSON.stringify(expr)}:`, error);
}

export function parseExpression(expr) {
  // Non-string and empty inputs return a constant — return the literal
  // value (or NaN for empty) rather than 0, so callers using
  // `Number.isFinite()` can detect "no expression" the same way they
  // detect "expression failed".
  if (typeof expr !== 'string') {
    if (typeof expr === 'number') return () => expr;
    return () => NaN;
  }
  if (!expr.trim()) return () => NaN;

  try {
    const source = normalizeSource(expr);
    const tokens = tokenize(source);
    const fn = (p) => {
      try {
        const parser = new ExpressionParser(tokens);
        return parser.evaluate(p);
      } catch (e) {
        // Return NaN (not 0) on per-evaluation failure so callers who
        // grep for a sentinel via `Number.isFinite` detect the failure
        // instead of silently treating it as a legitimate zero.  See
        // backlog T2.5.
        _warnOnce(expr, 'evaluate', e);
        return NaN;
      }
    };
    fn._rawExpr = expr;
    return fn;
  } catch (e) {
    // Tokenize/normalize failed.  Emit one warn per expression and
    // return a function that yields NaN so downstream `Number.isFinite`
    // checks fail closed.
    _warnOnce(expr, 'parse', e);
    return () => NaN;
  }
}

if (typeof window !== 'undefined' && isDevRuntime()) {
  window.testExpressionParser = (expr, pVal = 1) => {
    try {
      const fn = parseExpression(expr);
      console.log(`Expr: "${expr}" -> Result(p=${pVal}):`, fn(pVal));
      return fn(pVal);
    } catch (e) {
      console.error(e);
    }
  };
}
