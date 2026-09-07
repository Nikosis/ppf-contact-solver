// File: intersection.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Live edge-triangle intersection detection for the CUDA-free emulator.
//
// Without a detector here the emulator could only report
// `intersection_free = true` and fabricate records under
// PPF_EMULATED_FAIL_AT_FRAME, which would leave nothing about intersection
// reporting exercisable off a GPU. That includes the three allowances of issue
// #138, whose whole point is to change WHICH pairs are reported. This module
// is what gives those rules a CUDA-free acceptance gate
// (`rig_emulated_intersection`).
//
// What it shares with the CUDA path, and what it does not:
//
//   * The PREDICATE is the same one, `ppf_isect::edge_triangle_intersect` from
//     cpp/contact/intersect_core.hpp, which the device kernels and the host
//     build-time check also call. There is no second implementation of the
//     pierce test anywhere.
//   * The PAIR FILTERS are the same rules in the same order as
//     FaceEdgeIntersectTester: shared vertex, either-dynamic, either-nonzero-
//     mass, same-PDRD-body, both-collider, and then the issue-#138
//     allowances. The allowance rule is literally the same function the device
//     testers call, `ppf_isect::intersection_tolerated` from
//     cpp/contact/intersect_policy.hpp, so the two backends cannot come to
//     grant different sets. `rig_emulated_intersection` drives four cases
//     through this side, and `rig_intersection_allowances` walks the same
//     rules at the scene-build gate.
//   * The BROAD PHASE is a uniform grid, not the LBVH the device builds. The
//     grid is rebuilt every call, which is affordable because it is linear and
//     because the emulator only ever runs rig-sized scenes; an LBVH here would
//     be a second traversal implementation to keep correct for no gain.
//   * The ARITHMETIC is double, where the device is float. The emulator is a
//     host backend and is not trying to reproduce the device's round-off; it
//     answers the geometric question, so it takes the accurate path. This is
//     also what the host build-time check does.
//
// Only edge-vs-face is detected. Edge-edge proximity and point-point overlap
// are barrier-offset tests rather than pierce tests, and the emulator assembles
// no contact and carries no offsets it could trust, so reporting them here
// would mean inventing a second, differently-calibrated notion of "too close".

#ifndef PPF_CTS_EMUL_INTERSECTION_HPP
#define PPF_CTS_EMUL_INTERSECTION_HPP

#include "../cpp/contact/intersect_core.hpp"
#include "../cpp/contact/intersect_policy.hpp"
#include "../cpp/data.hpp"

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace emul_isect {

// Cap on how many records are handed back, matching the device's
// MAX_INTERSECTION_RECORDS contract: the count reported to the host is the
// number STORED, and detection itself stops at the first hit per edge, so a
// badly tangled scene stays bounded.
constexpr unsigned MAX_RECORDS = 256;

struct Grid {
    double cell = 1.0;
    double origin[3] = {0.0, 0.0, 0.0};
    int dim[3] = {1, 1, 1};
    std::unordered_map<long long, std::vector<unsigned>> buckets;

    long long key(int x, int y, int z) const {
        // dim is bounded below, so a mixed-radix pack fits comfortably in 64
        // bits; the map means an empty cell costs nothing.
        return (static_cast<long long>(x) * 73856093LL) ^
               (static_cast<long long>(y) * 19349663LL) ^
               (static_cast<long long>(z) * 83492791LL);
    }

    void cell_of(const double *p, int *out) const {
        for (int k = 0; k < 3; ++k) {
            out[k] = static_cast<int>(std::floor((p[k] - origin[k]) / cell));
        }
    }
};

inline void vertex_xyz(const Vec3f &v, const Vec3f &origin, double *out) {
    // Everything the predicate sees is a DIFFERENCE from a single scene-wide
    // origin vertex rather than a world position, exactly as the device
    // wrapper does per pair. The predicate only ever uses coordinate
    // differences (intersect_core.hpp), so the shared origin cancels.
    Vec3f d = (v - origin).cast<float>();
    out[0] = static_cast<double>(d[0]);
    out[1] = static_cast<double>(d[1]);
    out[2] = static_cast<double>(d[2]);
}

// True when the pair must be SKIPPED, for any reason other than geometry.
// Mirrors FaceEdgeIntersectTester's guard, including the order of its clauses.
inline bool pair_filtered(const DataSet &data, unsigned face_index,
                          unsigned edge_index) {
    const Vec3u &f = data.mesh.mesh.face.data[face_index];
    const Vec2u &e = data.mesh.mesh.edge.data[edge_index];
    for (unsigned i = 0; i < 3; ++i) {
        if (f[i] == e[0] || f[i] == e[1]) {
            return true;
        }
    }
    const FaceProp &fp = data.prop.face.data[face_index];
    const EdgeProp &ep = data.prop.edge.data[edge_index];
    bool either_dyn = fp.fixed == false || ep.fixed == false;
    bool either_nonzero = fp.mass > 0.0f || ep.mass > 0.0f;
    if (!either_dyn || !either_nonzero) {
        return true;
    }
    const VertexProp &fv = data.prop.vertex.data[f[0]];
    const VertexProp &ev = data.prop.vertex.data[e[0]];
    if (fv.pdrd_body_index != 0 && fv.pdrd_body_index == ev.pdrd_body_index) {
        return true;
    }
    if (fv.collider && ev.collider) {
        return true;
    }
    return ppf_isect::intersection_tolerated(fv, ev, fp.pin_allow_intersection,
                                             ep.pin_allow_intersection);
}

// Scans the current pose and appends up to MAX_RECORDS face-edge records.
// Returns true when the pose is intersection FREE, matching the sense of
// contact::check_intersection.
inline bool check(const DataSet &data,
                  std::vector<IntersectionRecord> &records) {
    const unsigned n_face = data.mesh.mesh.face.size;
    const unsigned n_edge = data.mesh.mesh.edge.size;
    const unsigned n_vert = data.vertex.curr.size;
    if (n_face == 0 || n_edge == 0 || n_vert == 0) {
        return true;
    }

    const Vec3f origin = data.vertex.curr.data[0];
    std::vector<double> pos(static_cast<size_t>(n_vert) * 3);
    for (unsigned i = 0; i < n_vert; ++i) {
        vertex_xyz(data.vertex.curr.data[i], origin, &pos[3 * size_t(i)]);
    }

    // Cell size: the mean face bounding-box diagonal, so a face lands in a
    // handful of cells whatever the scene's absolute scale. Falls back to a
    // unit cell for a degenerate (all-coincident) pose rather than dividing by
    // zero.
    std::vector<double> fmin(size_t(n_face) * 3), fmax(size_t(n_face) * 3);
    double extent_sum = 0.0;
    for (unsigned t = 0; t < n_face; ++t) {
        const Vec3u &f = data.mesh.mesh.face.data[t];
        for (int k = 0; k < 3; ++k) {
            double lo = pos[3 * size_t(f[0]) + k];
            double hi = lo;
            for (unsigned c = 1; c < 3; ++c) {
                double v = pos[3 * size_t(f[c]) + k];
                lo = v < lo ? v : lo;
                hi = v > hi ? v : hi;
            }
            fmin[3 * size_t(t) + k] = lo;
            fmax[3 * size_t(t) + k] = hi;
            extent_sum += hi - lo;
        }
    }
    Grid grid;
    double mean_extent = extent_sum / (3.0 * double(n_face));
    grid.cell = mean_extent > 0.0 ? mean_extent : 1.0;
    for (unsigned t = 0; t < n_face; ++t) {
        int lo[3], hi[3];
        grid.cell_of(&fmin[3 * size_t(t)], lo);
        grid.cell_of(&fmax[3 * size_t(t)], hi);
        for (int x = lo[0]; x <= hi[0]; ++x) {
            for (int y = lo[1]; y <= hi[1]; ++y) {
                for (int z = lo[2]; z <= hi[2]; ++z) {
                    grid.buckets[grid.key(x, y, z)].push_back(t);
                }
            }
        }
    }

    bool clear = true;
    std::vector<unsigned> seen;
    for (unsigned ei = 0; ei < n_edge && records.size() < MAX_RECORDS; ++ei) {
        const Vec2u &e = data.mesh.mesh.edge.data[ei];
        const double *a0 = &pos[3 * size_t(e[0])];
        const double *a1 = &pos[3 * size_t(e[1])];
        double emin[3], emax[3];
        for (int k = 0; k < 3; ++k) {
            emin[k] = a0[k] < a1[k] ? a0[k] : a1[k];
            emax[k] = a0[k] > a1[k] ? a0[k] : a1[k];
        }
        int lo[3], hi[3];
        grid.cell_of(emin, lo);
        grid.cell_of(emax, hi);
        seen.clear();
        bool hit = false;
        for (int x = lo[0]; x <= hi[0] && !hit; ++x) {
            for (int y = lo[1]; y <= hi[1] && !hit; ++y) {
                for (int z = lo[2]; z <= hi[2] && !hit; ++z) {
                    auto it = grid.buckets.find(grid.key(x, y, z));
                    if (it == grid.buckets.end()) {
                        continue;
                    }
                    for (unsigned t : it->second) {
                        // A face spanning several cells is visited once per
                        // cell; the hash also collides by construction, so the
                        // bucket can hold a face whose box does not overlap.
                        // Both are handled by re-testing the box and by this
                        // small per-edge dedupe.
                        bool dup = false;
                        for (unsigned s : seen) {
                            if (s == t) {
                                dup = true;
                                break;
                            }
                        }
                        if (dup) {
                            continue;
                        }
                        seen.push_back(t);
                        bool overlap = true;
                        for (int k = 0; k < 3; ++k) {
                            if (emax[k] < fmin[3 * size_t(t) + k] ||
                                emin[k] > fmax[3 * size_t(t) + k]) {
                                overlap = false;
                                break;
                            }
                        }
                        if (!overlap || pair_filtered(data, t, ei)) {
                            continue;
                        }
                        const Vec3u &f = data.mesh.mesh.face.data[t];
                        const double *v0 = &pos[3 * size_t(f[0])];
                        const double *v1 = &pos[3 * size_t(f[1])];
                        const double *v2 = &pos[3 * size_t(f[2])];
                        if (!ppf_isect::edge_triangle_intersect<double>(
                                a0, a1, v0, v1, v2)) {
                            continue;
                        }
                        clear = false;
                        if (records.size() < MAX_RECORDS) {
                            IntersectionRecord r{};
                            r.type = 0; // face-edge, matching contact.cu
                            r.elem0 = t;
                            r.elem1 = ei;
                            r.num_verts0 = 3;
                            r.num_verts1 = 2;
                            const double *src[5] = {v0, v1, v2, a0, a1};
                            for (int s = 0; s < 5; ++s) {
                                for (int k = 0; k < 3; ++k) {
                                    r.positions[3 * s + k] =
                                        static_cast<float>(src[s][k]);
                                }
                            }
                            records.push_back(r);
                        }
                        // One record per edge is enough to fail the pose, and
                        // it keeps a tangled scene from filling the buffer
                        // with one edge's hits.
                        hit = true;
                        break;
                    }
                }
            }
        }
    }
    return clear;
}

} // namespace emul_isect

#endif
