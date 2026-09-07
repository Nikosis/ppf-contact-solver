# File: scenarios/rig_emulated_intersection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The emulator's LIVE intersection scan, and the issue-#138 allowances at that
# second gate.
#
# The emulator runs a real edge-triangle scan at initialize and after every
# step (cpp_emul/intersection.hpp), sharing the pierce predicate and the pair
# filters with the device tester. Without that scan the emulator could only
# report `intersection_free = true` and fabricate records under
# PPF_EMULATED_FAIL_AT_FRAME, so nothing about intersection reporting would be
# exercisable off a GPU. This scenario is what verifies the scan.
#
# Three things are asserted, and the first is the one that decays silently:
#
#   1. a scene that tangles MID-RUN is caught. The scan is the only physics-
#      adjacent thing the emulator computes, so if it regresses to a constant
#      `true` every other emulated scenario still passes.
#   2. the allowance suppresses it, so the same run completes.
#   3. the failing run leaves intersection_records.json behind. The flag alone
#      stops the run; the records are what the addon's violation overlay
#      draws, and an empty file reads to a user as "something went wrong with
#      no geometry to look at".
#
# EMULATED ONLY, and not because the real backend lacks the feature. The scene
# drives a fully pinned blade through a free sheet: on CUDA the contact barrier
# and the CCD line search push the sheet aside and the tangle never forms, so
# the control would not abort and the scenario would be asserting the opposite
# of what it says. The device path's own allowances are covered by
# examples/allow_intersection_smoke.py.
#
# `rig_intersection_allowances` covers the same three rules at the SCENE-BUILD
# gate. Both gates have to grant the same set.
#
# The probe runs in a SUBPROCESS: it imports `frontend`, which loads the
# per-tree cdylib and installs the emulator's debug patches, and the
# orchestrator imports every scenario into one long-lived process that must
# not inherit either.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# The emulator is the point: see the header.
BACKENDS = ("emulated",)
# Runs a solver process per case, so it is slower than a pure-Python check and
# should not share a worker with another solver.
NOT_PARALLELIZABLE = True


_PROBE = r'''
import json
import os
import sys

REPO_ROOT = sys.argv[1]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

import frontend
from frontend import App


def run_case(project, allow_kind):
    """Drive a pinned blade through a free sheet and report what happened.

    The blade must not be PARALLEL to the sheet: two parallel sheets sliding
    through each other produce no edge-triangle pierce at any sampled pose, so
    the scan would be right to stay silent and the case would prove nothing.
    The blade lies in XY and its X-running edges pierce the wall's YZ
    triangles as it advances.
    """
    app = App.create(project)
    Vw, Fw = app.mesh.square(res=6, ex=[0, 1, 0], ey=[0, 0, 1])
    app.asset.add.tri("wall", Vw, Fw)
    Vb, Fb = app.mesh.square(res=6, ex=[1, 0, 0], ey=[0, 1, 0])
    app.asset.add.tri("blade", Vb, Fb)

    scene = app.scene.create()
    scene.add("wall").at(0.0, 0.0, 0.0)
    blade = scene.add("blade").at(-2.0, 0.0, 0.0)
    pin = blade.pin(allow_intersection=(allow_kind == "pin"))
    pin.move_by([3.0, 0.0, 0.0], t_start=0.0, t_end=1.0)
    if allow_kind == "inter":
        blade.param.set("allow-inter-object-intersection", 1.0)

    # Starts apart: the blade spans x in [-3, -1] and the wall sits at x = 0,
    # so initialize() must accept it and the only way to fail is the live scan.
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("dt", 0.02).set("frames", 40)
    session = session.build()
    try:
        session.start(blocking=True)
    except Exception:
        # A detected intersection surfaces as a solver abort, which the
        # frontend raises. The verdict is `finished()`, not the exception.
        pass
    records = os.path.join(session.info.path, "output",
                           "intersection_records.json")
    n_records = 0
    if os.path.exists(records):
        with open(records) as f:
            n_records = len(json.load(f).get("records", []))
    return {"finished": bool(session.finished()), "records": n_records}


cases = {}

control = run_case("rig_emul_isect_control", None)
cases["live_intersection_is_detected"] = {
    "ok": not control["finished"],
    "details": control,
}
cases["failing_run_writes_records"] = {
    "ok": (not control["finished"]) and control["records"] > 0,
    "details": control,
}

inter = run_case("rig_emul_isect_inter", "inter")
cases["inter_object_allowance_suppresses"] = {
    "ok": inter["finished"],
    "details": inter,
}

pinned = run_case("rig_emul_isect_pin", "pin")
cases["pin_allowance_suppresses"] = {
    "ok": pinned["finished"],
    "details": pinned,
}

print("PPFRESULT" + json.dumps(cases))
'''


def run(ctx: r.ScenarioContext) -> dict:
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    # The emulator's per-step sleep exists so other scenarios can observe
    # BUSY/RUNNING transitions. Nothing here reads a transition, and three
    # 40-frame runs at the default 1000 ms would take two minutes.
    env["PPF_EMULATED_STEP_MS"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX],
        capture_output=True,
        text=True,
        env=env,
        timeout=max(ctx.timeout, 300.0),
    )
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("PPFRESULT")
    ]
    if not marker:
        return r.failed([
            "probe produced no result marker; "
            f"rc={proc.returncode} stderr={proc.stderr[-800:]!r}"
        ])
    cases = json.loads(marker[-1][len("PPFRESULT"):])
    return r.report_named_checks(cases, label="emulated intersection cases")
