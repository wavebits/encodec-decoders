const std = @import("std");

const Backend = enum { auto, ref, neon, accelerate };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // decoding in Debug is ~10x slower, so default to fast
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;
    // strip debug info, e.g. for release builds on devices
    const strip = b.option(bool, "strip", "Strip debug info (default: false)") orelse false;
    const profile = b.option(bool, "profile", "Print per-stage decode times (C version)") orelse false;
    const os = target.result.os.tag;
    const is_apple = os == .macos or os == .ios;

    // C compute backend; auto = accelerate on Apple, neon on other AArch64, ref elsewhere
    var backend = b.option(Backend, "backend", "C compute backend (default: auto)") orelse .auto;
    if (backend == .auto) {
        backend = if (is_apple) .accelerate else if (target.result.cpu.arch == .aarch64) .neon else .ref;
    }

    var flags: std.ArrayList([]const u8) = .empty;
    flags.appendSlice(b.allocator, &.{ "-std=c17", "-Wall", "-Wextra", "-pedantic" }) catch @panic("OOM");
    switch (backend) {
        .neon => flags.append(b.allocator, "-DZENCODEC_BACKEND_NEON") catch @panic("OOM"),
        .accelerate => flags.appendSlice(b.allocator, &.{ "-DZENCODEC_BACKEND_ACCELERATE", "-DACCELERATE_NEW_LAPACK" }) catch @panic("OOM"),
        else => {},
    }
    if (profile) flags.append(b.allocator, "-DZENCODEC_PROFILE") catch @panic("OOM");

    // C17 implementation
    const c_mod = b.createModule(.{ .target = target, .optimize = optimize, .strip = strip, .link_libc = true });
    c_mod.addCSourceFiles(.{
        .root = b.path("c"),
        .files = &.{ "zencodec.c", "backend_ref.c", "backend_neon.c", "backend_accelerate.c", "play.c", "main.c" },
        .flags = flags.items,
    });
    if (os == .macos) c_mod.linkFramework("AudioToolbox", .{});
    if (backend == .accelerate) c_mod.linkFramework("Accelerate", .{});
    const c_exe = b.addExecutable(.{ .name = "wb_decode-c", .root_module = c_mod });
    b.installArtifact(c_exe);

    // Zig implementation
    const zig_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = os == .macos, // AudioToolbox
    });
    if (os == .macos) zig_mod.linkFramework("AudioToolbox", .{});
    const zig_exe = b.addExecutable(.{ .name = "wb_decode-zig", .root_module = zig_mod });
    b.installArtifact(zig_exe);
}
