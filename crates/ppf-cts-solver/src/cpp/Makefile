# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

NVCC=/usr/local/cuda/bin/nvcc

OUT_DIR=build
LIBDIR=$(OUT_DIR)/lib

# CUDA architectures, read from the manifest beside this file rather than
# spelled out here. cuda_arch.txt records why it is the single source and which
# consumers derive from it. It is parsed with awk off a ';'-commented,
# space-separated file so build.bat can read the same file with cmd.exe's
# `for /f` and the two builds cannot come to ship different architectures.
CUDA_ARCH_FILE := cuda_arch.txt
CUDA_ARCH_FLOOR := $(strip $(shell awk '$$1=="floor"{print $$2; exit}' $(CUDA_ARCH_FILE) 2>/dev/null))
CUDA_ARCH_CUBINS := $(strip $(shell awk '$$1=="cubin"{print $$2}' $(CUDA_ARCH_FILE) 2>/dev/null))
# A missing or unparsable manifest stops the build here. Continuing on an empty
# list would link a .so carrying no device code, which surfaces much later as
# every GPU being rejected at run time, and that failure names the device rather
# than the file that actually went missing.
ifeq ($(CUDA_ARCH_FLOOR),)
$(error no 'floor' line in $(CURDIR)/$(CUDA_ARCH_FILE))
endif
ifeq ($(CUDA_ARCH_CUBINS),)
$(error no 'cubin' lines in $(CURDIR)/$(CUDA_ARCH_FILE))
endif
# One -gencode per cubin, every one lowered from the single floor IR.
CUDA_GENCODE := $(foreach arch,$(CUDA_ARCH_CUBINS),-gencode arch=compute_$(CUDA_ARCH_FLOOR),code=sm_$(arch))
# Rendered "sm_86, sm_89, ..." for the text main.cu prints when it rejects a
# device, so that message cannot name a different set from the one linked.
CUDA_ARCH_LIST_STR := $(shell awk '$$1=="cubin"{printf "%ssm_%s", (n++?", ":""), $$2}' $(CUDA_ARCH_FILE))

# Warning suppressions
SUPPRESS_WARNINGS=1222 2527 2529 2651 2653 2668 2669 2670 2671 2735 2737 2739 20012 20011 20014 177 940 1394

# nvcc flags for CUDA files. The device solver has no external linear-algebra
# dependency: the in-house crates/.../cpp/linalg headers replaced Eigen, so
# there is no -I<eigen> or -DEIGEN_WARNINGS_DISABLED here (Eigen now lives only
# in the macOS-only cpp_emul emulator, which carries its own include).
NVCC_FLAGS=-std=c++17 -O3 -MMD --device-c -Xcompiler -fPIC -D__NVCC__ --expt-relaxed-constexpr --extended-lambda -Wno-deprecated-gpu-targets
NVCC_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))

# Device link-time optimization (LTO). Compile each TU to an LTO intermediate
# (code=lto_<floor>); the device link below (-dlto) runs whole-program optimization,
# inlining the cross-TU barrier::compute_stiffness into the contact-Hessian embed
# kernels (a 255-register ABI call otherwise) and roughly halving contact
# matrix-assembly cost.
#
# The LTO IR sits at the floor virtual arch (compute_75 as shipped), and every
# device target in the link below emits native SASS from that IR.
# IMPORTANT: device LTO CANNOT embed a JIT-able PTX. The -dlto link lowers its IR
# straight to SASS; a code=compute_XX request is silently dropped (verified with
# cuobjdump: an LTO link yields cubins only, zero PTX). The .so is therefore
# frozen to the exact SASS images we ship, there is no forward-JIT fallback, so
# a cubin per supported arch is enumerated at the link. Adding a GPU generation
# is one line in cuda_arch.txt plus a rebuild. nvcc rejects -dlto beside
# -gencode at compile, hence code=lto_<floor> here and the full -gencode list
# only at the link.
NVCC_FLAGS += -gencode arch=compute_$(CUDA_ARCH_FLOOR),code=lto_$(CUDA_ARCH_FLOOR)
# The architectures this image ships, as text, for the run-time rejection
# message. Single-quoted so the embedded ", " survives the shell and reaches
# nvcc as one argument holding a C string literal.
NVCC_FLAGS += -DPPF_SHIPPED_ARCH_STR='"$(CUDA_ARCH_LIST_STR)"'

# nvcc flags for C++ files (compiled as host code)
NVCC_CPP_FLAGS=-std=c++17 -O3 -MMD -Xcompiler -fPIC,-Wall,-Wno-unused-function

# Source files
BASE_DIRS=buffer main utility csrmat contact energy eigenanalysis barrier strainlimiting solver lbvh plasticity schwarz
KERNEL_FILES=kernels/reduce.cu kernels/exclusive_scan.cu kernels/vec_ops.cu kernels/radix_sort.cu
CUDA_FILES=$(foreach dir,$(BASE_DIRS),$(dir)/$(dir).cu) $(KERNEL_FILES)
SIMPLELOG_FILES=simplelog/SimpleLog.cpp
STUB_FILES=stub.cpp

# Object files
CUDA_OBJS=$(addprefix $(OUT_DIR)/cuda/,$(CUDA_FILES:%.cu=%.o))
SIMPLELOG_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(SIMPLELOG_FILES:%.cpp=%.o))
STUB_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(STUB_FILES:%.cpp=%.o))
DEPS = $(CUDA_OBJS:.o=.d) $(SIMPLELOG_OBJS:.o=.d) $(STUB_OBJS:.o=.d)

all: $(LIBDIR)/libsimbackend_cuda.so $(LIBDIR)/libsimplelog.so

# Objects depend on the Makefile so a flags change forces a recompile (nvcc's
# mtime view only tracks sources; without this a flag-only edit relinks stale .o).
# They depend on the architecture manifest for the same reason and it is not the
# same file: editing it changes both the LTO floor these are compiled to and the
# PPF_SHIPPED_ARCH_STR baked into main.cu, neither of which make can see through
# the flags alone. Without this an arch edit leaves every .o stale, and the
# result is caught only later, by arch_guard, as a build failure telling the
# reader to delete the build directory.
$(OUT_DIR)/cuda/%.o: %.cu Makefile $(CUDA_ARCH_FILE)
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

# Compile C++ files with nvcc (uses host compiler internally)
$(OUT_DIR)/cpp/%.o: %.cpp Makefile
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_CPP_FLAGS) $< -o $@

# Link CUDA library: -dlto runs whole-program optimization on the lto_<floor> objects
# and emits one native SASS cubin per architecture (no PTX, see the note above).
# SASS is binary-compatible forward within a major version, so the sm_86 cubin
# also runs sm_87 (Jetson Orin) and sm_89 (Ada: RTX 40, L40S); each new MAJOR
# needs its own cubin. A brand-new arch not listed here fails to launch ("named
# symbol not found") until its cubin is added and the lib rebuilt.
#   sm_75  Turing     RTX 20, GTX 16, T4     (oldest shipped; requested, and
#                                             the self-hosted CI runner's card)
#   sm_86  Ampere     RTX 30, A10/A40/A6000  (+ sm_87 Orin, sm_89 Ada via compat)
#   sm_89  Ada        RTX 40, L40S, L4       (native; perf on the dev/CI GPU)
#   sm_90  Hopper     H100, H200, GH200
#   sm_100 Blackwell  B200, GB200            (datacenter)
#   sm_120 Blackwell  RTX 50-series          (consumer)
# Keep this list in sync with SUPPORTED_SM in crates/ppf-cts-core/src/utils.rs,
# which gates solver launch against exactly these architectures.
#
# This link, not the 17 compiles, is what a rebuild costs. Each -gencode is an
# independent whole-program optimization of the same LTO IR down to that arch's
# SASS, and nvcc runs them one after another unless told otherwise, so the link
# costs the arch count times a single arch. Measured on an L40S (16 cores, nvcc
# 12.8): 178.9 s for the five against 33.6 s for one alone, while all 17 compiles
# together take ~6 s wall at -j16.
#
# -t overlaps those per-arch optimizations (0 = one thread per CPU), taking the
# link to 43.9 s, near the floor set by the slowest single arch. It is a
# scheduling flag only: it selects no optimization level and changes no codegen
# decision, and the .so it produces is BYTE-IDENTICAL to the serial one (md5
# verified for -t 0, -t 5 and -t 16 against an unflagged link).
#
# Do NOT reach for --split-compile here. It parallelizes too, and slightly
# faster (39.9 s), but by partitioning the optimizer's scope, so it is not free:
# it emitted 1,150,536 SASS instructions against 1,132,816 for the serial link,
# 17,720 more (+1.6%) in the solver's hottest device code. Build time is not
# worth paying for at run time. --split-compile-extended is worse still: it
# fails this link outright ("nvlink error: Size doesn't match for '__unnamed_1'
# in 'lto.cubin'").
#
# -t belongs on this link ONLY, never on the compile rule above. That rule
# builds a single arch, where -t has nothing to overlap and measured no
# gain (solver.cu: 5.3 s with -t 16 vs 5.4 s without), and make already runs the
# compiles -j wide, so each of the 17 would stack its own thread pool on top.
# The manifest is a prerequisite because the -gencode list is derived from it:
# an arch added or removed there changes this link and nothing else, so without
# it make would consider an image built for the previous list up to date. $^
# would then also feed the manifest to nvcc as an input file, hence $(filter-out).
$(LIBDIR)/libsimbackend_cuda.so: $(CUDA_OBJS) $(SIMPLELOG_OBJS) $(STUB_OBJS) $(CUDA_ARCH_FILE)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -dlto -t 0 $(CUDA_GENCODE) -o $@ $(filter-out $(CUDA_ARCH_FILE),$^)

# SimpleLog library (for standalone use)
$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
