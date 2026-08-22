# JIT Arm Isolation — making an A/B actually be two binaries

## When this file applies

The candidate is gated by an env var, a config flag, or a module-level constant, **and** the kernel
is produced by a caching JIT (FlyDSL, Triton, `torch.compile`, or any framework with a compile cache
on disk). If your patch is plain source that is always compiled, this file does not apply — skip it.

This is an **authoring** obligation, not a verification one. Verify will catch a one-binary A/B and
void your direction; catching it there costs the round. Catching it here costs one hash.

## The failure

Two arms print two different markers from Python and then execute **one identical cached binary**,
because the switch never entered the cache key. The measurement reads ~1.000, and every gate passes:
the patch is real, the marker fires, the null arm behaves, the numbers are internally consistent.
There is no symptom. It is indistinguishable from "the idea did not work" — which is precisely why it
is dangerous: it does not produce a wrong number, it produces a **null that closes an axis**.

A retroactive audit on this workflow found an all-ON arm, an all-OFF arm **and** the canonical
unpatched tree all resolving to one `disk_key`. Three arms, two separate checkouts, one binary. The
"well-powered null" it had produced was void and the axis had to be reopened.

## Why separate checkouts do not save you

The compile cache root is typically **per-machine, not per-tree** (FlyDSL: `~/.flydsl/cache`). Two
workspaces with different source therefore hit the *same* cache. Building the arms in separate
checkouts isolates the source and not the artifact — the intuition that "different tree ⇒ different
build" is false here, and it is the intuition most people bring.

## What enters the key and what does not

Worked example, FlyDSL (`jit_function.py`, `_walk` at :247, same at :438/:534): the tracer collects
**only the outer function's `co_names`**. Consequences:

| construct | enters the key? |
|---|---|
| a closure scalar the tracer captures directly (e.g. an `aStages` int) | **yes** |
| a module-level constant read in the **outer** traced function | **yes** |
| an env var / global read inside a **nested or inner-JIT body** | **no** |
| a change that only renames the emitted kernel symbol | **no** |

The middle two are the trap: the switch is visibly in the source, visibly read at run time, and
invisible to the cache. A knob can also survive *by accident* — one on this workflow worked only
because its arm A moved a closure scalar (which does enter the key) while its arm B, which read a
global in an inner body, was never exercised.

Other frameworks have the same shape with different details. Triton keys on the JITFunction source
plus specialization constants — a Python-level global read inside a helper does not key. Check your
framework's key construction rather than assuming; the question to answer is always *"is the value my
switch changes an input to the key, or only to the runtime behaviour of an already-keyed body?"*

## The remedy: anchor the key in the outer function

Read the switch **once, at module level or in the outer traced function**, into a name the tracer
sees, and let the inner body read that:

```python
_key_anchor = int(os.environ.get("MY_SWITCH", "0"))   # module level, read by the OUTER traced fn

def outer(...):
    mode = _key_anchor        # now `mode` is a closure scalar → it enters the key
    ...
    inner(mode)               # inner reads the parameter, never the env var
```

Confirmed working on this workflow: anchoring a three-valued knob this way produced three distinct
keys (`7cf14e9c` / `af58afb4` / `40aee6de`) with the canonical tree proved identical to arm 0 at IR
`sha be49c3b728f4e9e5` — which is the *other* half of the proof and is easy to forget: the null arm
must **match** canonical as surely as the candidate must **differ** from base.

## What you owe in your report

Whenever your candidate is JIT-gated, put in `activation_evidence`:

- the **base** arm's cache key / name-normalised ISA hash / resolved binary path,
- the **candidate** arm's same quantity, shown to differ,
- and where a null arm is supposed to be byte-identical work, its hash shown to match canonical.

Name-normalise the ISA hash (strip symbol names) so a kernel rename cannot fake a difference. The
verify role has three dedicated fields for these and the script compares them itself; a hash written
only into prose does not bind.

**Cheap ways to get the hash:** dump the framework's cache key for the compiled callable; hash the
emitted ISA/IR text after stripping symbol names; or compare the resolved cache directory per arm.
Any one of the three is sufficient. Doing none of them makes your result provisional at best and void
at worst.

## Do not confuse this with activation

They fail at different layers and have different remedies:

| | symptom | remedy |
|---|---|---|
| **inactive** | the host path never ran | fix the predicate; check the marker |
| **one binary** | the host path ran on both arms, and both dispatched the same code | anchor the key |

A marker proves the first and says nothing about the second.
