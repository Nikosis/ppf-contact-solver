# File: _solid_weight_transfer_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Carrying a painted per-Blender-vertex weight field onto a tetrahedral mesh.
#
# The artist paints on the mesh they can see; the solver's object is the
# tetrahedralized one, whose vertices are different points. Both stages of the
# transfer are convex combinations of painted values, so the tests here assert
# that property directly: constants survive, the range never widens, and the
# interior stays inside the range of the surface it was extended from.

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.sparse")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend._decoder_ import (
        _build_solid_weight_transfer,
        _harmonic_interior_operator_strict,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


def freudenthal_grid(n: int):
    """A unit cube as an ``n x n x n`` grid of cubes, six tets per cube."""
    axis = np.linspace(0.0, 1.0, n + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    verts = grid.reshape(-1, 3)

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [
                    vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                    vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ]
                for a, b, d, e in (
                    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
                ):
                    tets.append([c[a], c[b], c[d], c[e]])
    return verts, np.asarray(tets, dtype=np.int64)


def boundary_faces(tets):
    """The triangles belonging to exactly one tetrahedron."""
    faces = {}
    for tet in tets:
        for tri in (
            (tet[0], tet[1], tet[2]), (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]), (tet[1], tet[2], tet[3]),
        ):
            faces[tuple(sorted(tri))] = faces.get(tuple(sorted(tri)), 0) + 1
    return np.asarray(
        [list(k) for k, count in faces.items() if count == 1], dtype=np.int64
    )


def make_transfer(n=3, jitter=0.0):
    """A transfer whose Blender surface is the tet mesh's own boundary.

    A real tetrahedralization does not preserve the input vertices, which is
    exactly why the transfer exists. `jitter` displaces the Blender surface so
    the closest-triangle stage does real work rather than matching exactly.
    """
    verts, tets = freudenthal_grid(n)
    faces = boundary_faces(tets)
    surf_ids = np.unique(faces.reshape(-1))
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[surf_ids] = np.arange(surf_ids.size)
    bl_verts = verts[surf_ids].copy()
    if jitter:
        rng = np.random.default_rng(0)
        bl_verts += rng.normal(scale=jitter, size=bl_verts.shape)
    bl_tris = remap[faces]
    tet_mesh = types.SimpleNamespace()
    tet_mesh._pin_blender_surface = (bl_verts, bl_tris)
    mesh = (verts, faces, tets)

    class TetMesh(tuple):
        pass

    holder = TetMesh(mesh)
    holder._pin_blender_surface = (bl_verts, bl_tris)
    return _build_solid_weight_transfer(holder, faces, "Cube"), holder, surf_ids


def test_a_constant_field_survives_the_transfer():
    transfer, _mesh, _surf = make_transfer()
    for value in (0.0, 1.0, 0.375):
        out = transfer.apply(np.full(transfer.n_blender, value))
        assert out.shape == (transfer.n_tet,)
        assert np.allclose(out, value, atol=1e-12)


def test_a_painted_field_stays_inside_the_unit_interval():
    transfer, _mesh, _surf = make_transfer(jitter=0.01)
    rng = np.random.default_rng(7)
    out = transfer.apply(rng.random(transfer.n_blender))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_the_interior_stays_inside_the_range_of_the_surface():
    transfer, _mesh, surf_ids = make_transfer()
    rng = np.random.default_rng(11)
    out = transfer.apply(rng.random(transfer.n_blender))
    interior = np.setdiff1d(np.arange(transfer.n_tet), surf_ids)
    assert interior.size > 0
    surface = out[surf_ids]
    assert out[interior].min() >= surface.min() - 1e-12
    assert out[interior].max() <= surface.max() + 1e-12


def test_a_gradient_reaches_the_interior_rather_than_flattening():
    transfer, mesh, surf_ids = make_transfer()
    bl_verts = mesh._pin_blender_surface[0]
    weights = (bl_verts[:, 0] - bl_verts[:, 0].min())
    weights /= weights.max()
    out = transfer.apply(weights)
    interior = np.setdiff1d(np.arange(transfer.n_tet), surf_ids)
    assert out[interior].std() > 0.05


def test_several_maps_share_one_factorization():
    transfer, _mesh, _surf = make_transfer()
    rng = np.random.default_rng(3)
    a = rng.random(transfer.n_blender)
    b = rng.random(transfer.n_blender)
    stacked = transfer.apply(np.column_stack([a, b]))
    assert stacked.shape == (transfer.n_tet, 2)
    assert np.allclose(stacked[:, 0], transfer.apply(a))
    assert np.allclose(stacked[:, 1], transfer.apply(b))


def test_a_wrong_weight_count_is_refused_by_name():
    transfer, _mesh, _surf = make_transfer()
    with pytest.raises(ValueError, match="Cube"):
        transfer.apply(np.zeros(transfer.n_blender + 1))


def test_an_absent_blender_surface_is_refused_by_name():
    verts, tets = freudenthal_grid(2)
    faces = boundary_faces(tets)
    with pytest.raises(RuntimeError, match="Cube"):
        _build_solid_weight_transfer((verts, faces, tets), faces, "Cube")


def test_the_residual_bounds_the_rounding_the_extension_introduces():
    transfer, _mesh, _surf = make_transfer(n=4)
    # The tolerance is the operator's own measured departure from reproducing
    # a constant, floored at the resolution of the float32 array the weights
    # land in, so it can never mask a representable violation.
    assert transfer.tolerance >= float(np.finfo(np.float32).eps)
    assert transfer.tolerance < 1e-6


def test_the_strict_operator_names_an_empty_boundary():
    _verts, tets = freudenthal_grid(2)
    with pytest.raises(ValueError, match="surface"):
        _harmonic_interior_operator_strict(100, tets, [], [1, 2])


def test_the_strict_operator_names_a_bad_tet_array():
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        _harmonic_interior_operator_strict(10, np.zeros((3, 3)), [0], [1])
