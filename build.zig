const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // decoding in Debug is ~10x slower, so default to fast
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;
    // strip debug info, e.g. for release builds on devices
    const strip = b.option(bool, "strip", "Strip debug info (default: false)") orelse false;
    const is_macos = target.result.os.tag == .macos;

    // C17 implementation
    const c_mod = b.createModule(.{ .target = target, .optimize = optimize, .strip = strip, .link_libc = true });
    c_mod.addCSourceFiles(.{
        .root = b.path("c"),
        .files = &.{ "zencodec.c", "play.c", "main.c" },
        .flags = &.{ "-std=c17", "-Wall", "-Wextra", "-pedantic" },
    });
    if (is_macos) c_mod.linkFramework("AudioToolbox", .{});
    const c_exe = b.addExecutable(.{ .name = "wb_decode-c", .root_module = c_mod });
    b.installArtifact(c_exe);

    // Zig implementation
    const zig_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = is_macos, // AudioToolbox
    });
    if (is_macos) zig_mod.linkFramework("AudioToolbox", .{});
    const zig_exe = b.addExecutable(.{ .name = "wb_decode-zig", .root_module = zig_mod });
    b.installArtifact(zig_exe);
}
