"""
test_param_audit.py — keeps params.yaml and the code that reads it in sync. Run:
`python -m tests.test_param_audit` from the repo root.

Three failure modes, all of which have actually bitten this project:

1. DIVERGENT DEFAULT — `param.get('k', X)` where params.yaml says Y. Harmless
   while the key exists, silently changes flight behaviour the moment the key is
   deleted or a config is written from scratch. The worst instance found:
   `initial_yaw_deg` is 180 in YAML but defaults to 0.0 in code, so dropping the
   key would initialise the EKF facing North on a South-facing track.

2. DEAD KEY — present in params.yaml, read by nothing. Invites tuning a value
   that cannot do anything.

3. SHADOWED KEY — the specific trap that motivated this test. `gate_debounce_sec`
   sat in params.yaml looking live and plausible while mavlink_rx.py hardcoded
   2.0 and never consulted params at all. A dead key at least fails loudly when
   grepped; a shadowed one reads as working.

Receiver names below are the ones actually used for the params dict (surveyed by
AST across the repo). Any other `.get()` receiver — self.data, shared, mav, dbg,
latest, ... — is a different dict entirely and is deliberately not inspected.
"""

import ast
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {'.venv', '__pycache__', '.git', 'YOLO', 'runs', 'logs', '.claude'}

# Expressions that denote the params dict. `raw` is dyn.py's local name for the
# freshly-loaded YAML inside load_params().
PARAM_RECEIVERS = {'param', 'params', '_p', 'self._param', 'self.param',
                   'raw', 'raw_params', '_params_raw'}

# Keys deliberately read with a default but intentionally absent from params.yaml
# (opt-in features, offline-tool-only knobs). Absence is the "off" state.
ALLOWED_MISSING = {
    'accel_bias',          # opt-in; written only by fit_shadow.py --write-bias
    'blip_thrust_frac',    # sysid scripts only
    'blip_dur_sec',        # sysid scripts only
    'rate_bandwidth',      # stored into p[] by dyn.py, never consumed
    'K_rate_roll_override',
    'K_rate_pitch_override',
    'Kp_vel', 'Ki_vel', 'Kp_vz', 'Ki_vz',   # legacy fallbacks for Kp_vN/etc.
}

# Keys in params.yaml with no reader. Should stay EMPTY: the correct response to
# a dead key is to delete it, not to list it here.
ALLOWED_DEAD = set()

# Sites where the in-code default may differ from params.yaml, with a reason.
# Also should stay near-empty — the point of the test is to drive this to zero.
ALLOWED_DIVERGENT = {}


def _iter_py():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith('.py'):
                yield os.path.join(root, fn)


def _literal(node):
    """Python value of an AST literal, or _NOT_LITERAL for anything computed."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return _NOT_LITERAL


class _NotLiteral:
    def __repr__(self):
        return '<non-literal>'


_NOT_LITERAL = _NotLiteral()


def collect_param_gets():
    """[(key, default_or_NOT_LITERAL, relpath, lineno)] for every params lookup."""
    out = []
    for path in _iter_py():
        try:
            tree = ast.parse(open(path, encoding='utf-8').read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = os.path.relpath(path, REPO).replace('\\', '/')
        if rel.startswith('tests/'):
            continue                       # tests use synthetic dicts by design
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'get' and n.args):
                continue
            try:
                recv = ast.unparse(n.func.value)
            except Exception:
                continue
            if recv not in PARAM_RECEIVERS:
                continue
            key_node = n.args[0]
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                continue
            default = _literal(n.args[1]) if len(n.args) > 1 else None
            out.append((key_node.value, default, rel, n.lineno))
    return out


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        if not cond:
            n_fail += 1
        print(f'[{"OK" if cond else "FAIL"}] {name}')

    yml = yaml.safe_load(open(os.path.join(REPO, 'params.yaml'), encoding='utf-8'))
    gets = collect_param_gets()
    read_keys = {k for k, _, _, _ in gets}

    print(f'scanned {len(gets)} params lookups, {len(yml)} keys in params.yaml\n')

    # 1. Divergent in-code defaults.
    divergent = []
    for key, default, rel, line in gets:
        if key not in yml or default is _NOT_LITERAL or default is None:
            continue
        if not _same(default, yml[key]) and f'{rel}:{line}' not in ALLOWED_DIVERGENT:
            divergent.append((key, yml[key], default, rel, line))
    if divergent:
        print(f'  {len(divergent)} divergent default(s) — yaml != code:')
        for key, y, d, rel, line in sorted(divergent):
            print(f'    {key:38s} yaml={y!r:22s} code={d!r:12s}  {rel}:{line}')
    check('every in-code default matches params.yaml', not divergent)

    # 2. Dead keys.
    dead = sorted(k for k in yml if k not in read_keys and k not in ALLOWED_DEAD)
    if dead:
        print(f'  {len(dead)} dead key(s) in params.yaml — read by nothing:')
        for k in dead:
            print(f'    {k}')
    check('every params.yaml key is read somewhere', not dead)

    # 3. Read but absent from YAML — always uses the in-code default.
    missing = sorted({k for k in read_keys if k not in yml and k not in ALLOWED_MISSING})
    if missing:
        print(f'  {len(missing)} key(s) read in code but absent from params.yaml:')
        for k in missing:
            sites = [f'{r}:{l}' for kk, _, r, l in gets if kk == k]
            print(f'    {k:38s} {sites[0]}')
    check('every key read in code exists in params.yaml (or is allow-listed)', not missing)

    print(f'\n{"ALL PASSED" if not n_fail else f"{n_fail} CHECK(S) FAILED"}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
