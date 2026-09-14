#!/usr/bin/env python3
"""validate_skill.py — gate an expert skill before it can land as `validated`.

Two-sided gate (see expert_skills/README.md):
  1. EFFICACY   — on a matching model/shape the skill achieves >= its `expects` (isolated or e2e).
  2. DO-NO-HARM — integrating it does not regress a control scenario / non-trigger run.

This tool does NOT itself spin up a server or a GPU. The actual measurement is produced by the real
harness (kernel_workflow for scope:kernel, e2e_workflow for scope:e2e) so numbers stay honest and
on-box. validate_skill.py has three modes:

  --static            schema + operator alignment + required sections + links. No GPU. (CI default.)
  --emit-plan         print the exact Workflow invocation to measure this skill (by scope).
  --record ...        check supplied measured numbers against `expects`, then stamp the skill's
                      validation block (status: validated|failed) and reindex.

Examples:
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --static
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --emit-plan --model /models/Qwen3.5-27B-FP8
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --record \
      --artifact /path/eval_dir --e2e-pct 2.1 --parity pass --gpu gfx942/MI300X --model Qwen3.5-27B-FP8
"""
import argparse, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(ROOT, "skills")
CAP_INDEX = os.path.normpath(os.path.join(ROOT, "..", "index", "capability_index.yaml"))
GEAK = os.path.normpath(os.path.join(ROOT, "..", ".."))
REQUIRED_SECTIONS = ["When to use", "Mechanism", "Procedure", "Do-no-harm notes", "Sources"]
FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def load(skill_id):
    path = os.path.join(SKILLS_DIR, skill_id, "skill.md")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no such skill: {path}")
    txt = open(path).read()
    m = FM_RE.match(txt)
    if not m:
        sys.exit(f"ERROR: {path}: no YAML frontmatter")
    return path, yaml.safe_load(m.group(1)), m.group(2), txt


def _skill_file(skill_path, fm, key):
    value = str(fm.get(key) or "").strip()
    if not value:
        return None
    if os.path.isabs(value) or ".." in value.split(os.sep):
        raise ValueError(f"{key} must be relative to the skill directory")
    return os.path.join(os.path.dirname(skill_path), value)


def load_validation(skill_path, fm):
    path = _skill_file(skill_path, fm, "validation_file")
    if path is None:
        return None, fm.get("validation") or {}
    if not os.path.isfile(path):
        return path, {}
    data = yaml.safe_load(open(path)) or {}
    return path, data if isinstance(data, dict) else {}


def static_check(skill_path, fm, body):
    errs = []
    for k in ("id", "scope", "match", "expects"):
        if k not in fm:
            errs.append(f"missing frontmatter key: {k}")
    if fm.get("scope") not in ("kernel", "e2e"):
        errs.append(f"scope must be kernel|e2e (got {fm.get('scope')!r})")
    op = (fm.get("match") or {}).get("operator")
    if os.path.exists(CAP_INDEX):
        ops = {c["operator"] for c in (yaml.safe_load(open(CAP_INDEX)).get("candidates") or [])}
        if op not in ops:
            errs.append(f"match.operator '{op}' not in capability_index.yaml")
    exp = fm.get("expects") or {}
    if fm.get("scope") == "kernel" and "isolated_speedup_min" not in exp:
        errs.append("kernel scope needs expects.isolated_speedup_min")
    if fm.get("scope") == "e2e" and "e2e_delta_min_pct" not in exp:
        errs.append("e2e scope needs expects.e2e_delta_min_pct")
    for sec in REQUIRED_SECTIONS:
        if f"## {sec}" not in body:
            errs.append(f"missing body section: ## {sec}")
        elif not _section_filled(body, sec):
            errs.append(f"body section '## {sec}' is empty / placeholder only")
    for key in (
        "playbook_file", "contract_file", "validation_file",
        "runtime_validation_file",
    ):
        if not fm.get(key):
            continue
        try:
            referenced = _skill_file(skill_path, fm, key)
        except ValueError as exc:
            errs.append(str(exc))
            continue
        if not referenced or not os.path.isfile(referenced):
            errs.append(f"{key} does not exist: {fm.get(key)!r}")
    contract_path = None
    try:
        contract_path = _skill_file(skill_path, fm, "contract_file")
    except ValueError:
        pass
    if contract_path and os.path.isfile(contract_path):
        try:
            contract = yaml.safe_load(open(contract_path)) or {}
            if contract.get("schema_version") != "expert-skill-contract-v1":
                errs.append("contract_file must use schema_version expert-skill-contract-v1")
            if contract.get("skill_id") != fm.get("id"):
                errs.append("contract_file skill_id must match skill.md id")
            if str(contract.get("revision") or "") != str(fm.get("revision") or ""):
                errs.append("contract_file revision must match skill.md revision")
            if not contract.get("checks"):
                errs.append("contract_file checks must be non-empty")
        except (OSError, yaml.YAMLError) as exc:
            errs.append(f"cannot parse contract_file: {exc}")
    playbook_path = None
    try:
        playbook_path = _skill_file(skill_path, fm, "playbook_file")
    except ValueError:
        pass
    if playbook_path and os.path.isfile(playbook_path):
        try:
            match = FM_RE.match(open(playbook_path).read())
            if not match:
                errs.append("playbook_file must have YAML frontmatter")
            else:
                playbook = yaml.safe_load(match.group(1)) or {}
                if str(playbook.get("revision") or "") != str(fm.get("revision") or ""):
                    errs.append("playbook_file revision must match skill.md revision")
        except (OSError, yaml.YAMLError) as exc:
            errs.append(f"cannot parse playbook_file: {exc}")
    try:
        validation_path, validation = load_validation(skill_path, fm)
    except ValueError as exc:
        errs.append(str(exc))
        validation_path, validation = None, {}
    if validation_path:
        if validation.get("schema_version") != "expert-skill-validation-v1":
            errs.append("validation_file must use schema_version expert-skill-validation-v1")
        if validation.get("skill_id") != fm.get("id"):
            errs.append("validation_file skill_id must match skill.md id")
        if str(validation.get("revision") or "") != str(fm.get("revision") or ""):
            errs.append("validation_file revision must match skill.md revision")
        if validation.get("status") not in ("draft", "validated", "stale", "failed", "mega_only"):
            errs.append("validation_file status is invalid")
    return errs


def _section_filled(body, sec):
    m = re.search(rf"## {re.escape(sec)}\n(.*?)(?=\n## |\Z)", body, re.S)
    if not m:
        return False
    content = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S).strip()
    return len(content) > 0


def emit_plan(skill_id, fm, args):
    scope = fm.get("scope")
    if scope == "e2e":
        model = args.model or "<MODEL_PATH>"
        print("# EFFICACY (e2e_workflow, Director same-session A/B):")
        print(f"Workflow scriptPath={GEAK}/e2e_workflow/e2e_workflow.js args:")
        print(f"  model_path={model} workflow_dir={GEAK}/e2e_workflow use_expert_skills=true")
        print(f"  task='reproduce expert_skill:{skill_id} on the matching head; gate by e2e A/B'")
        print("# DO-NO-HARM (control model that does NOT match the selector must stay within noise band):")
        print(f"  model_path=<CONTROL_MODEL> use_expert_skills=true  # expect |e2e delta| < noise band")
    else:
        print("# EFFICACY (kernel_workflow, isolated A/B vs the immutable oracle):")
        print(f"Workflow scriptPath={GEAK}/kernel_workflow/kernel_workflow.js args:")
        extras = [f"expert_skill_id={skill_id}"]
        for key in (
            "playbook_file", "contract_file", "validation_file",
            "runtime_validation_file",
        ):
            if fm.get(key):
                arg = key.replace("_file", "")
                extras.append(
                    f"expert_skill_{arg}={os.path.join(SKILLS_DIR, skill_id, str(fm[key]))}"
                )
        print(f"  kernel_path=<OP_TASK_DIR> workflow_dir={GEAK}/kernel_workflow "
              f"use_expert_skills=true {' '.join(extras)}")
        print(f"  target_language={(fm.get('match') or {}).get('to_backend') or 'triton'}")
        print(f"  task='reproduce expert_skill:{skill_id}; beat oracle, hold parity'")
    print("\nThen stamp the result with:  validate_skill.py", skill_id,
          "--record --artifact <eval_dir> ...")


def record(path, fm, body, txt, args):
    exp = fm.get("expects") or {}
    scope = fm.get("scope")
    ok, reasons = True, []
    if args.parity and args.parity != "pass" and exp.get("parity", "required") == "required":
        ok = False; reasons.append(f"parity={args.parity} but required")
    if scope == "kernel":
        need = float(exp.get("isolated_speedup_min", 1.0))
        got = args.isolated
        if got is None:
            sys.exit("ERROR: --isolated required for kernel scope")
        if got < need:
            ok = False; reasons.append(f"isolated {got} < expects {need}")
    else:
        need = float(exp.get("e2e_delta_min_pct", 0.0))
        got = args.e2e_pct
        if got is None:
            sys.exit("ERROR: --e2e-pct required for e2e scope")
        if got < need:
            ok = False; reasons.append(f"e2e +{got}% < expects +{need}%")
    if not args.artifact:
        sys.exit("ERROR: --artifact <eval_dir> required to record a result")

    validation_path, validation = load_validation(path, fm)
    validation["status"] = "validated" if ok else "failed"
    validation["last_verified"] = args.date or ""
    validation["gpu"] = args.gpu or ""
    validation["model"] = args.model or ""
    validation["measured"] = {
        "isolated": args.isolated if args.isolated is not None else "",
        "e2e_pct": args.e2e_pct if args.e2e_pct is not None else "",
        "parity": args.parity or "",
    }
    validation["artifact"] = args.artifact
    if validation_path:
        validation.setdefault("schema_version", "expert-skill-validation-v1")
        validation.setdefault("skill_id", fm.get("id"))
        validation.setdefault("revision", fm.get("revision", "v1"))
        with open(validation_path, "w") as f:
            yaml.safe_dump(validation, f, sort_keys=False, allow_unicode=True, width=100)
    else:
        fm["validation"] = validation
        with open(path, "w") as f:
            f.write("---\n")
            f.write(yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=100))
            f.write("---\n")
            f.write(body)
    print(f"recorded: status={validation['status']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    os.system(f"python3 {os.path.join(HERE, 'scaffold.py')} --reindex >/dev/null 2>&1")
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("skill_id")
    p.add_argument("--static", action="store_true")
    p.add_argument("--emit-plan", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--artifact", default=""); p.add_argument("--gpu", default="")
    p.add_argument("--model", default=""); p.add_argument("--date", default="")
    p.add_argument("--isolated", type=float, default=None)
    p.add_argument("--e2e-pct", dest="e2e_pct", type=float, default=None)
    p.add_argument("--parity", default="")
    a = p.parse_args()
    path, fm, body, txt = load(a.skill_id)

    if a.emit_plan:
        return emit_plan(a.skill_id, fm, a)
    if a.record:
        return record(path, fm, body, txt, a)
    # default / --static
    errs = static_check(path, fm, body)
    if errs:
        print("STATIC FAIL:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"STATIC OK: {a.skill_id} (scope={fm.get('scope')}, operator={fm['match']['operator']}). "
          f"Run with --emit-plan to get the on-box measurement command.")


if __name__ == "__main__":
    main()
