# File: scenarios/bl_connection_failure_reporting.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What the user is told when a connection step fails.
#
# A failure on this path is only as useful as where it lands. One that ends
# in the Console, or in a status line the next redraw clears, reaches the user
# as "it does not connect" and nothing more. Each check below pins the REPORT,
# not the failure: the failures themselves are covered by the scenarios named
# beside them.
#
#   * A protocol mismatch is the one diagnosis a user cannot guess, and a
#     restart is the fix only when the on-disk binary is already the matching
#     one. A solver that arrived as a Docker image or a Windows bundle carries
#     its version with it, so the message has to name updating a side rather
#     than only restarting, and it has to reach the panel, not just the
#     Console.
#   * Stop Server on the SSH and Docker backends kills by process name. The
#     command runs inside `/bin/sh -c '...'`, whose own command line holds the
#     pattern, so a `pkill -f` match includes the shell issuing it. And a
#     server still answering after the grace period is not a stop that worked.
#   * The cbor2 recovery button runs `ensurepip` first. Every failure worth
#     naming there (a read-only Python install, a stripped ensurepip, a proxy)
#     is described in the child's stderr, so the message has to carry it.
#
# Nothing here contacts a server: the reducer is a pure function, the stop
# path is driven against a recording backend, and the installer is driven
# against a stubbed subprocess.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure reporting logic; no solver is involved.
BACKENDS = ("emulated", "real")


_DRIVER_BODY = r'''
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _RecordingBackend:
    """Backend stand-in that records exec_command and answers queries."""

    backend_type = "docker"
    server_port = 9090

    def __init__(self, alive_forever):
        self._alive_forever = alive_forever
        self.commands = []
        self.queries = 0

    @property
    def current_directory(self):
        return "/root/ppf-contact-solver"

    def exec_command(self, command, **kw):
        self.commands.append(command)
        return {"exit_code": 0, "stdout": [], "stderr": []}

    def query(self, args, project_name, chunk_size):
        self.queries += 1
        return ({}, True) if self._alive_forever else ({}, False)

    def is_alive(self):
        return self._alive_forever


module_mod = None
saved_run = None
try:
    transitions = __import__(pkg + ".core.transitions", fromlist=["transition"])
    protocol = __import__(pkg + ".core.protocol", fromlist=["PROTOCOL_VERSION"])
    events = __import__(pkg + ".core.events", fromlist=["ServerPolled"])
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "runner"])
    module_mod = __import__(pkg + ".core.module", fromlist=["install_module"])

    # ---- a protocol mismatch reaches the panel, and names the real fix ----
    # See bl_rust_binary_protocol for the handshake itself.
    state = facade.engine.state
    mismatched = {
        "protocol_version": "0.0.0-not-the-addons",
        "upload_id": "",
        "status": "",
    }
    new_state, effects = transitions.transition(state, events.ServerPolled(response=mismatched))
    record("mismatch_marks_the_version_not_ok", new_state.version_ok is False,
           {"version_ok": new_state.version_ok})
    record("mismatch_reaches_the_panel_error",
           "0.0.0-not-the-addons" in new_state.error
           and protocol.PROTOCOL_VERSION in new_state.error,
           {"error": new_state.error})
    record("mismatch_names_the_image_and_bundle_case",
           "image" in new_state.error.lower() and "bundle" in new_state.error.lower(),
           {"error": new_state.error})
    record("mismatch_still_stops_the_server",
           any(type(e).__name__ == "DoStopServer" for e in effects),
           {"effects": [type(e).__name__ for e in effects]})

    matching = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "upload_id": "u1",
        "status": "",
    }
    ok_state, _ = transitions.transition(state, events.ServerPolled(response=matching))
    record("matching_version_sets_no_error", not ok_state.error,
           {"error": ok_state.error})

    # ---- Stop Server kills by process name, not by command line ----
    runner = facade.runner
    saved_backend = runner._backend
    saved_project = runner._project_name
    try:
        stopped = _RecordingBackend(alive_forever=False)
        runner._backend = stopped
        runner._project_name = "stop_reporting"
        runner._do_stop_server()
        facade.tick()
        kill = [c for c in stopped.commands if "pkill" in c]
        record("stop_matches_the_process_name",
               kill and "pkill -x ppf-cts-server" in kill[0],
               {"commands": stopped.commands})
        record("stop_does_not_match_its_own_command_line",
               all(" -f " not in c for c in kill), {"commands": kill})
        record("clean_stop_reports_no_error", not runner._engine.state.error,
               {"error": runner._engine.state.error})

        # A server still answering after the grace period is not a stop that
        # worked, and saying it did is the only thing the user could act on.
        still_up = _RecordingBackend(alive_forever=True)
        runner._backend = still_up
        runner._do_stop_server()
        facade.tick()
        err = runner._engine.state.error
        record("stop_reports_a_server_that_is_still_answering",
               "still answering" in err and "9090" in err, {"error": err})
        record("stop_polled_the_full_grace", still_up.queries >= 5,
               {"queries": still_up.queries})
    finally:
        runner._backend = saved_backend
        runner._project_name = saved_project

    # ---- a repeating query failure is written once, not once per poll ----
    # The background poll repeats for as long as the connection is held, so
    # an unreachable server would put one identical line in the console per
    # tick and bury everything else in it.
    backends = __import__(pkg + ".core.backends", fromlist=["_query_via_channel"])
    written = []
    saved_console_write = backends.console.write
    backends.console.write = lambda msg, **kw: written.append(msg)
    try:
        backends._clear_query_failure()

        def _refuse():
            raise ConnectionRefusedError("[Errno 111] Connection refused")

        for _ in range(5):
            backends._query_via_channel(_refuse, {}, "proj", 4096)
        record("repeating_query_failure_is_written_once",
               len(written) == 1 and "Connection refused" in written[0],
               {"written": written})

        def _other():
            raise TimeoutError("timed out")

        backends._query_via_channel(_other, {}, "proj", 4096)
        record("a_different_cause_is_written_too", len(written) == 2,
               {"written": written})
    finally:
        backends.console.write = saved_console_write
        backends._clear_query_failure()

    # ---- the cbor2 recovery button carries the reason it failed ----
    # See bl_cbor2_missing_reported for the missing-wheel reporting itself.
    import time as _time

    class _Result:
        def __init__(self, code, out, err):
            self.returncode, self.stdout, self.stderr = code, out, err

    saved_run = module_mod.subprocess.run
    module_mod.subprocess.run = lambda *a, **kw: _Result(
        1, "", "PermissionError: [Errno 13] Read-only file system: 'site-packages'"
    )
    module_mod.install_module(["cbor2==6.0.1"])
    for _ in range(100):
        if not module_mod.get_installing_status():
            break
        _time.sleep(0.05)
    msg = module_mod.get_install_error_message()
    record("install_failure_reports_the_child_stderr",
           "Read-only file system" in msg, {"message": msg})
    record("install_failure_names_the_step",
           "ensurepip" in msg, {"message": msg})
    record("install_failure_is_recorded_as_a_failure",
           module_mod.get_install_result() is False,
           {"result": module_mod.get_install_result()})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if module_mod is not None and saved_run is not None:
        module_mod.subprocess.run = saved_run
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the reducer is a pure function, the stop path
    is driven against a recording backend the scenario installs and removes,
    and the installer is driven against a stubbed ``subprocess.run``. Nothing
    contacts a server, so it runs on any host.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
