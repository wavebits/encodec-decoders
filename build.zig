const std = @import("std");

const Backend = enum { auto, ref, neon, accelerate, blas };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // decoding in Debug is ~10x slower, so default to fast
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;
    // strip debug info, e.g. for release builds on devices
    const strip = b.option(bool, "strip", "Strip debug info (default: false)") orelse false;
    const profile = b.option(bool, "profile", "Print per-stage decode times (C version)") orelse false;
    // keep the weights in f16 in memory (C version): ~23 MB less, not with accelerate
    const f16_weights = b.option(bool, "f16", "Keep weights in f16 in memory (C version; ref and neon backends)") orelse false;
    // the optional Vocos decoder (C version), for Vocos weights files
    const vocos = b.option(bool, "vocos", "Include the Vocos decoder (C version)") orelse false;
    const os = target.result.os.tag;
    const is_apple = os == .macos or os == .ios;

    // C compute backend; auto = accelerate on Apple, neon on other AArch64, ref elsewhere.
    // blas links the system's OpenBLAS (-Dblas-lib to use another CBLAS library).
    // With -Df16, auto skips accelerate, whose BLAS takes f32 weights only.
    var backend = b.option(Backend, "backend", "C compute backend (default: auto)") orelse .auto;
    const is_arm64 = target.result.cpu.arch == .aarch64;
    if (backend == .auto) {
        backend = if (is_apple and !f16_weights) .accelerate else if (is_arm64) .neon else .ref;
    }
    const blas_lib = b.option([]const u8, "blas-lib", "CBLAS library for -Dbackend=blas (default: openblas)") orelse "openblas";
    if (f16_weights and (backend == .accelerate or backend == .blas)) {
        std.debug.print("error: -Df16=true is not supported with the accelerate or blas backends; use neon or ref\n", .{});
        std.process.exit(1);
    }

    var flags: std.ArrayList([]const u8) = .empty;
    flags.appendSlice(b.allocator, &.{ "-std=c17", "-Wall", "-Wextra", "-pedantic" }) catch @panic("OOM");
    switch (backend) {
        .neon => flags.append(b.allocator, "-DZENCODEC_BACKEND_NEON") catch @panic("OOM"),
        .accelerate => flags.appendSlice(b.allocator, &.{ "-DZENCODEC_BACKEND_ACCELERATE", "-DACCELERATE_NEW_LAPACK" }) catch @panic("OOM"),
        .blas => flags.append(b.allocator, "-DZENCODEC_BACKEND_BLAS") catch @panic("OOM"),
        else => {},
    }
    if (profile) flags.append(b.allocator, "-DZENCODEC_PROFILE") catch @panic("OOM");
    if (f16_weights) flags.append(b.allocator, "-DZENCODEC_F16_WEIGHTS") catch @panic("OOM");
    if (vocos) flags.append(b.allocator, "-DZENCODEC_VOCOS") catch @panic("OOM");

    const c_mod = b.createModule(.{ .target = target, .optimize = optimize, .strip = strip, .link_libc = true });
    c_mod.addCSourceFiles(.{
        .root = b.path("c"),
        .files = &.{ "zencodec.c", "vocos.c", "pool.c", "backend_ref.c", "backend_neon.c", "backend_accelerate.c", "backend_blas.c", "play.c", "main.c" },
        .flags = flags.items,
    });
    if (os == .macos) c_mod.linkFramework("AudioToolbox", .{});
    if (backend == .accelerate) c_mod.linkFramework("Accelerate", .{});
    if (backend == .blas) c_mod.linkSystemLibrary(blas_lib, .{});
    b.installArtifact(b.addExecutable(.{ .name = "wb_decode", .root_module = c_mod }));
}
