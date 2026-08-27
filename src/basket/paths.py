"""Where everything lives, discovered rather than assumed.

Every module used to spell its own `<script dir>/../..`, which made the directory depth
part of the contract: moving a file one level broke the data paths silently, and running a
script from the wrong working directory produced `../../out/../../out/<name>`.  The layout
should not be something anyone has to think about.

The root is found by walking up from this file until a directory carries the markers of
this repository.  Set NF_ROOT to override (useful if you keep inputs on another volume).
Set NF_RAW_DIR to point at the raw dunnhumby CSVs wherever they are.
"""
import os

_MARKERS = ("requirements.txt", "src")


def find_root(start=None, use_env=True):
    """First ancestor of `start` that looks like this repository.

    `use_env=False` ignores NF_ROOT, which is how the location of the CODE is found even
    when generated artifacts have been redirected elsewhere.
    """
    env = os.environ.get("NF_ROOT")
    if use_env and env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    d = here
    while True:
        if all(os.path.exists(os.path.join(d, m)) for m in _MARKERS):
            return d
        parent = os.path.dirname(d)
        if parent == d:                       # reached the filesystem root
            # Fall back to two levels up, which is where this file has always sat.
            return os.path.abspath(os.path.join(here, "..", ".."))
        d = parent


# The checkout itself, ignoring NF_ROOT: this is where the CODE is.
REPO = find_root(os.path.dirname(os.path.abspath(__file__)), use_env=False)

# Where generated artifacts go.  NF_ROOT moves these without moving the code, which is what
# you want to run against a scratch tree or keep large inputs on another volume.
ROOT = find_root()
DATA = os.path.join(ROOT, "data")
BI = os.path.join(ROOT, "basket_input")
OUT = os.path.join(ROOT, "out")

# The raw dunnhumby CSVs are an INPUT, so they default relative to the checkout, not to
# NF_ROOT -- otherwise redirecting the outputs would silently move where inputs are looked
# for.  Default: a sibling of the repository, which is how the archive unpacks.
RAW = os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(REPO), "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV"))
if not RAW.endswith(os.sep):
    RAW += os.sep


def resolve_ckpt(p):
    """Accept an absolute path, a path relative to the CWD, or a bare checkpoint name."""
    if os.path.exists(p):
        return os.path.abspath(p)
    cand = os.path.join(OUT, os.path.basename(p))
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"checkpoint not found: {p}\n  also tried: {cand}")


def ensure_dirs():
    for d in (DATA, BI, OUT):
        os.makedirs(d, exist_ok=True)
