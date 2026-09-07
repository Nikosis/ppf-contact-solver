// File: intersect_policy.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Single source of truth for the intersection ALLOWANCE rule of issue #138:
// given the two sides of an intersecting pair, is this a pair the user asked
// the solver to tolerate rather than report?
//
// One definition serves both C++ backends. The device intersect testers in
// cpp/contact/contact.cu call it from a kernel, and the emulator's host
// detector in cpp_emul/intersection.hpp calls it from plain C++. A second copy
// would be a correctness hazard rather than a duplication nuisance: the two
// gates must grant exactly the same set, and a scene that a GPU run reports and
// an emulator run does not (or the reverse) is a defect with no local symptom.
// The rule is mirrored once more in Rust, as `VertexIntersectPolicy::tolerated`
// in ppf-cts-core/src/kernels/intersection.rs, for the build-time check that
// runs before any backend is reached.
//
// The full truth table is gated in Rust, where the unit tests of
// ppf-cts-core/src/kernels/intersection.rs pair every allowed case with a
// control that must still be REPORTED, and end to end by the
// `rig_intersection_allowances` scenario.
//
// It suppresses REPORTING only: contact, CCD and the line search never consult
// it, so the solver still resolves whatever it can and simply stops aborting
// over what it cannot.
//
// This routine is FLOAT-FREE and deliberately NOT a template. It reads only
// unsigned and bool fields, so no floating-point type appears in it at all, and
// there is no scalar to template over. Both properties are worth stating rather
// than leaving to inspection: a `__host__ __device__` TEMPLATE is device code
// for every type it is instantiated with, so a host caller passing `double`
// emits a float64 device instantiation even though no kernel ever calls it,
// which `fp64_guard` in crates/ppf-cts-solver/build.rs fails the release build
// over. A plain non-templated function cannot do that.

#ifndef PPF_CTS_INTERSECT_POLICY_HPP
#define PPF_CTS_INTERSECT_POLICY_HPP

#include "../data.hpp"

// Same idiom as the sibling intersect_core.hpp: the annotation is spelled only
// when nvcc is compiling, so the header is also a valid host-only include. A
// macro of its own rather than intersect_core.hpp's PPF_ISECT_HD, so neither
// header depends on the other having been included first.
#if defined(__CUDACC__)
#define PPF_ISECT_POLICY_HD __host__ __device__
#else
#define PPF_ISECT_POLICY_HD
#endif

namespace ppf_isect {

// Three allowances, and each side is described by the props of the element's
// FIRST vertex (object identity and the material policy are per object, the
// convention `pdrd_body_index` and `collider` already use) plus the element's
// own precomputed "all N of my vertices are pinned by an allowing pin" bit.
//
// EITHER side is enough for the pin and inter-object allowances, so flagging a
// garment covers it against the character it is fitted to without the
// character having to be flagged too. Self-intersection is asked of one object
// only, so there is one flag to read.
PPF_ISECT_POLICY_HD inline bool
intersection_tolerated(const VertexProp &a, const VertexProp &b,
                       bool a_pin_allows, bool b_pin_allows) {
    // A fully pinned element's shape is prescribed. The solver was never going
    // to resolve an intersection it is part of, so reporting one only aborts a
    // run over geometry the user authored. This is the same reasoning that
    // makes `either_dyn` skip a pair whose BOTH sides are fully fix-pinned;
    // the allowance is what lets ONE pinned side be enough, and what extends
    // it to pull pins, whose hold is only as strong as their own force.
    if (a_pin_allows || b_pin_allows) {
        return true;
    }
    // NO_OBJECT_INDEX must not match itself: a pair of vertices whose objects
    // are both unknown is not evidence they share one.
    bool same_object = a.object_index != NO_OBJECT_INDEX &&
                       a.object_index == b.object_index;
    if (same_object) {
        return (a.intersect_policy & INTERSECT_ALLOW_SELF) != 0;
    }
    return ((a.intersect_policy | b.intersect_policy) &
            INTERSECT_ALLOW_INTER_OBJECT) != 0;
}

} // namespace ppf_isect

#endif // PPF_CTS_INTERSECT_POLICY_HPP
