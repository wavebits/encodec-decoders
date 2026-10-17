//! zencodec CLI (Zig).
//!
//!   zencodec decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]
//!   zencodec play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]
//!
//! Codes file (little-endian): int32 n_q, int32 n_frames, int32 codes[n_q][n_frames]
//! (see scripts/encode.py), or an .ecdc file from `encodec` compressed without --lm.

const std = @import("std");
const Io = std.Io;
const zencodec = @import("zencodec.zig");
const play = @import("play.zig");

const usage =
    \\usage: zencodec decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]
    \\       zencodec play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]
    \\
;

/// n_samples is the audio length to keep (an .ecdc file stores the original length).
const Codes = struct { data: []i32, n_q: usize, n_frames: usize, n_samples: usize };

fn readCodes(gpa: std.mem.Allocator, io: Io, path: []const u8) !Codes {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(bytes);
    if (bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "ECDC")) return parseEcdc(gpa, bytes);
    if (bytes.len < 8) return error.InvalidCodes;
    const n_q = std.mem.readInt(i32, bytes[0..4], .little);
    const n_frames = std.mem.readInt(i32, bytes[4..8], .little);
    if (n_q <= 0 or n_q > 1024 or n_frames <= 0 or n_frames > 1 << 24) return error.InvalidCodes;
    const n: usize = @intCast(n_q * n_frames);
    if (bytes.len - 8 != n * 4) return error.InvalidCodes;

    const data = try gpa.alloc(i32, n);
    for (data, 0..) |*v, i| v.* = std.mem.readInt(i32, bytes[8 + 4 * i ..][0..4], .little);
    const frames: usize = @intCast(n_frames);
    return .{ .data = data, .n_q = @intCast(n_q), .n_frames = frames, .n_samples = frames * zencodec.hop_length };
}

/// .ecdc file from `encodec` (Python): "ECDC", u8 version 0, u32 big-endian JSON length,
/// JSON metadata {"m", "al", "nc", "lm"}, then codes bit-packed LSB-first in frame-major
/// order. Only encodec_24khz without the language model (lm = false) is supported.
fn parseEcdc(gpa: std.mem.Allocator, bytes: []const u8) !Codes {
    const bits_per_code = 10; // log2(1024 bins)
    if (bytes.len < 9 or bytes[4] != 0) return error.InvalidCodes;
    const json_len = std.mem.readInt(u32, bytes[5..9], .big);
    if (json_len > bytes.len - 9) return error.InvalidCodes;

    const Meta = struct { m: []const u8, al: u64, nc: u32, lm: bool };
    const meta = std.json.parseFromSlice(Meta, gpa, bytes[9..][0..json_len], .{ .ignore_unknown_fields = true }) catch
        return error.InvalidCodes;
    defer meta.deinit();
    if (!std.mem.eql(u8, meta.value.m, "encodec_24khz")) return error.UnsupportedEcdcModel;
    if (meta.value.lm) return error.UnsupportedEcdcLanguageModel;
    const n_q: usize = meta.value.nc;
    const frames = std.math.divCeil(u64, meta.value.al, zencodec.hop_length) catch unreachable;
    if (n_q == 0 or n_q > 1024 or frames == 0 or frames > 1 << 24) return error.InvalidCodes;
    const n_frames: usize = @intCast(frames);

    const body = bytes[9 + json_len ..];
    const n = n_q * n_frames;
    if (body.len < (n * bits_per_code + 7) / 8) return error.InvalidCodes;
    const data = try gpa.alloc(i32, n);
    var acc: u32 = 0;
    var bits: u5 = 0;
    var i: usize = 0;
    for (body) |byte| {
        acc |= @as(u32, byte) << bits;
        bits += 8;
        while (bits >= bits_per_code and i < n) : (i += 1) {
            data[(i % n_q) * n_frames + i / n_q] = @intCast(acc & ((1 << bits_per_code) - 1)); // frame-major -> [n_q][n_frames]
            acc >>= bits_per_code;
            bits -= bits_per_code;
        }
        if (i == n) break;
    }
    return .{ .data = data, .n_q = n_q, .n_frames = n_frames, .n_samples = @intCast(meta.value.al) };
}

/// Decodes chunk by chunk on its own thread so playback can start after the first chunk.
const DecodeJob = struct {
    io: Io,
    model: *const zencodec.Model,
    codes: Codes,
    n_threads: usize,
    audio: []f32,
    ready: std.atomic.Value(usize) = .init(0), // samples decoded so far
    done: std.atomic.Value(bool) = .init(false),
    err: ?zencodec.Error = null,
    t_done: Io.Timestamp = undefined,

    fn run(j: *DecodeJob) void {
        j.decodeAll() catch |err| {
            j.err = err;
        };
        j.t_done = Io.Timestamp.now(j.io, .awake);
        j.done.store(true, .release);
    }

    fn decodeAll(j: *DecodeJob) zencodec.Error!void {
        const c = j.codes;
        const gpa = j.model.gpa;
        var stream = try zencodec.Stream.init(j.model, c.n_q, j.n_threads);
        defer stream.deinit();
        const chunk = try gpa.alloc(i32, c.n_q * zencodec.chunk_frames);
        defer gpa.free(chunk);
        var f: usize = 0;
        while (f < c.n_frames) : (f += zencodec.chunk_frames) {
            const n = @min(c.n_frames - f, zencodec.chunk_frames);
            for (0..c.n_q) |q| @memcpy(chunk[q * n ..][0..n], c.data[q * c.n_frames + f ..][0..n]);
            try stream.decode(chunk[0 .. c.n_q * n], n, j.audio[f * zencodec.hop_length ..][0 .. n * zencodec.hop_length]);
            j.ready.store(@min((f + n) * zencodec.hop_length, c.n_samples), .release);
        }
    }
};

fn writeWav(gpa: std.mem.Allocator, io: Io, path: []const u8, audio: []const f32, sample_rate: u32) !void {
    const data_size: u32 = @intCast(audio.len * 2);
    const buf = try gpa.alloc(u8, 44 + data_size);
    defer gpa.free(buf);

    const put32 = struct {
        fn f(b: []u8, v: u32) void {
            std.mem.writeInt(u32, b[0..4], v, .little);
        }
    }.f;
    const put16 = struct {
        fn f(b: []u8, v: u16) void {
            std.mem.writeInt(u16, b[0..2], v, .little);
        }
    }.f;
    @memcpy(buf[0..4], "RIFF");
    put32(buf[4..], 36 + data_size);
    @memcpy(buf[8..16], "WAVEfmt ");
    put32(buf[16..], 16);
    put16(buf[20..], 1); // PCM
    put16(buf[22..], 1); // mono
    put32(buf[24..], sample_rate);
    put32(buf[28..], sample_rate * 2);
    put16(buf[32..], 2); // block align
    put16(buf[34..], 16); // bits per sample
    @memcpy(buf[36..40], "data");
    put32(buf[40..], data_size);
    for (audio, 0..) |s, i| {
        const v: i16 = @intFromFloat(@round(std.math.clamp(s, -1.0, 1.0) * 32767.0));
        std.mem.writeInt(i16, buf[44 + 2 * i ..][0..2], v, .little);
    }

    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, buf);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var n_threads: usize = std.Thread.getCpuCount() catch 1;
    var pos: [4][]const u8 = undefined;
    var n_pos: usize = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            i += 1;
            n_threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (n_pos < pos.len) {
            pos[n_pos] = args[i];
            n_pos += 1;
        } else {
            std.debug.print(usage, .{});
            std.process.exit(1);
        }
    }

    const do_play = n_pos == 3 and std.mem.eql(u8, pos[0], "play");
    if (!do_play and !(n_pos == 4 and std.mem.eql(u8, pos[0], "decode"))) {
        std.debug.print(usage, .{});
        std.process.exit(1);
    }

    const t0 = Io.Timestamp.now(io, .awake);
    var model = try zencodec.Model.load(gpa, io, pos[1]);
    defer model.deinit();
    const t1 = Io.Timestamp.now(io, .awake);

    const codes = try readCodes(gpa, io, pos[2]);
    defer gpa.free(codes.data);

    const audio = try gpa.alloc(f32, codes.n_frames * zencodec.hop_length);
    defer gpa.free(audio);
    const sr = model.sample_rate;
    const n_samples = @min(codes.n_samples, audio.len);
    const seconds = @as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(sr));

    if (do_play) {
        var job = DecodeJob{ .io = io, .model = &model, .codes = codes, .n_threads = n_threads, .audio = audio };
        const thread = try std.Thread.spawn(.{}, DecodeJob.run, .{&job});
        while (job.ready.load(.acquire) == 0 and !job.done.load(.acquire)) try io.sleep(.fromMilliseconds(1), .awake);
        std.debug.print("playing {d:.2} s of audio ({d} codebooks, {d} frames)\n", .{ seconds, codes.n_q, codes.n_frames });
        std.debug.print("load {d} ms, first audio after {d} ms\n", .{
            t0.durationTo(t1).toMilliseconds(), t1.durationTo(Io.Timestamp.now(io, .awake)).toMilliseconds(),
        });
        const played = play.playStream(io, audio.ptr, &job.ready, &job.done, sr);
        thread.join();
        if (job.err) |err| {
            std.debug.print("error: decoding failed (n_q = {d}, max {d}): {t}\n", .{ codes.n_q, model.nCodebooks(), err });
            std.process.exit(1);
        }
        try played;
        std.debug.print("decode {d} ms\n", .{t1.durationTo(job.t_done).toMilliseconds()});
    } else {
        model.decode(codes.data, codes.n_q, codes.n_frames, audio, n_threads) catch |err| {
            std.debug.print("error: decoding failed (n_q = {d}, max {d}): {t}\n", .{ codes.n_q, model.nCodebooks(), err });
            std.process.exit(1);
        };
        const t2 = Io.Timestamp.now(io, .awake);
        std.debug.print("decoded {d:.2} s of audio ({d} codebooks, {d} frames)\n", .{ seconds, codes.n_q, codes.n_frames });
        std.debug.print("load {d} ms, decode {d} ms\n", .{ t0.durationTo(t1).toMilliseconds(), t1.durationTo(t2).toMilliseconds() });
        try writeWav(gpa, io, pos[3], audio[0..n_samples], sr);
    }
}
