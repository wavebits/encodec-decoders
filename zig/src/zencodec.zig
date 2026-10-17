//! zencodec: minimal, dependency-free EnCodec (24 kHz) decoder in Zig.
//! Based on encodec.cpp (https://github.com/PABannier/encodec.cpp).
//!
//! Architecture (SEANet decoder, causal, reflect padding), activations are [C][T]:
//!   RVQ decode            sum of n_q codebook embeddings          -> [128][T]
//!   conv k7               128 -> 512
//!   2-layer LSTM + skip   512
//!   4x { ELU, conv-transpose (stride r), residual block }  r = 8, 5, 4, 2
//!        residual block: conv1x1(x) + conv1x1(ELU(conv k3(ELU(x))))
//!   ELU, conv k7          32 -> 1                                  -> [1][T * 320]

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const ggml_magic: i32 = 0x67676d6c;
const ratios = [4]usize{ 8, 5, 4, 2 };
pub const hop_length: usize = ratios[0] * ratios[1] * ratios[2] * ratios[3];
/// 1 s; a good chunk size for Stream.decode
pub const chunk_frames: usize = 75;

const lanes = 8;
const Vec = @Vector(lanes, f32);

const Tensor = struct {
    name: []const u8,
    shape: [4]usize = .{ 1, 1, 1, 1 }, // PyTorch order (outermost first)
    n_dims: usize,
    data: []f32,
};

const Conv = struct { w: *const Tensor, b: *const Tensor };

const LstmLayer = struct { w_ih: *const Tensor, w_hh: *const Tensor, b_ih: *const Tensor, b_hh: *const Tensor };

const DecoderBlock = struct {
    upsample: Conv, // conv-transpose
    conv1: Conv, // k3, C -> C/2
    conv2: Conv, // k1, C/2 -> C
    shortcut: Conv, // k1, C -> C
};

/// Activations laid out as [C][T].
const Act = struct {
    c: usize,
    t: usize,
    d: []f32,

    const empty: Act = .{ .c = 0, .t = 0, .d = &.{} };

    fn init(gpa: Allocator, c: usize, t: usize) !Act {
        const d = try gpa.alloc(f32, c * t);
        @memset(d, 0);
        return .{ .c = c, .t = t, .d = d };
    }

    fn deinit(a: Act, gpa: Allocator) void {
        gpa.free(a.d);
    }

    fn row(a: Act, ch: usize) []f32 {
        return a.d[ch * a.t ..][0..a.t];
    }
};

pub const Error = error{ InvalidWeights, InvalidCodes } || Allocator.Error || std.Thread.SpawnError;

pub const Model = struct {
    gpa: Allocator,
    hidden_dim: usize,
    n_bins: usize,
    sample_rate: u32,

    tensors: []Tensor,
    codebooks: []*const Tensor, // [n_bins][hidden_dim] each
    init_conv: Conv,
    lstm: [2]LstmLayer,
    blocks: [4]DecoderBlock,
    final_conv: Conv,

    /// Load a weights file produced by scripts/convert.py (or a full encodec.cpp ggml-model.bin).
    pub fn load(gpa: Allocator, io: Io, path: []const u8) !Model {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn deinit(m: *Model) void {
        for (m.tensors) |t| {
            m.gpa.free(t.name);
            m.gpa.free(t.data);
        }
        m.gpa.free(m.tensors);
        m.gpa.free(m.codebooks);
    }

    pub fn nCodebooks(m: *const Model) usize {
        return m.codebooks.len;
    }

    /// Decode `codes` laid out as [n_q][n_frames] (codebook-major) into
    /// n_frames * hop_length mono samples. n_q may be any value in
    /// [1, nCodebooks()]: 2 = 1.5 kbps, 4 = 3 kbps, 8 = 6 kbps, 16 = 12 kbps, 32 = 24 kbps.
    /// Decodes in chunks of chunk_frames with a Stream, so memory use does not grow
    /// with the input length.
    pub fn decode(m: *const Model, codes: []const i32, n_q: usize, n_frames: usize, audio: []f32, n_threads: usize) Error!void {
        if (n_q < 1 or n_q > m.codebooks.len or n_frames < 1 or codes.len != n_q * n_frames or audio.len != n_frames * hop_length)
            return error.InvalidCodes;
        for (codes) |code| if (code < 0 or code >= m.n_bins) return error.InvalidCodes;

        var stream = try Stream.init(m, n_q, n_threads);
        defer stream.deinit();
        const chunk = try m.gpa.alloc(i32, n_q * chunk_frames);
        defer m.gpa.free(chunk);
        var f: usize = 0;
        while (f < n_frames) : (f += chunk_frames) {
            const n = @min(n_frames - f, chunk_frames);
            for (0..n_q) |q| @memcpy(chunk[q * n ..][0..n], codes[q * n_frames + f ..][0..n]);
            try stream.decode(chunk[0 .. n_q * n], n, audio[f * hop_length ..][0 .. n * hop_length]);
        }
    }
};

/// Streaming decoder: feed the codes chunk by chunk and get each chunk's audio right
/// away. The concatenated output is the same as Model.decode on the concatenated
/// codes, provided the first chunk has at least 7 frames (or is the whole input).
pub const Stream = struct {
    m: *const Model,
    n_q: usize,
    n_threads: usize,

    // state carried from one chunk to the next (see conv1d, convTranspose1d)
    init_hist: Act = .empty,
    block_hist: [4]Act = @splat(.empty),
    final_hist: Act = .empty,
    block_tail: [4]Act = @splat(.empty),
    lstm_state: []f32 = &.{}, // h, c for each of the 2 layers, [4][512]

    pub fn init(m: *const Model, n_q: usize, n_threads: usize) Error!Stream {
        if (n_q < 1 or n_q > m.codebooks.len) return error.InvalidCodes;
        var s = Stream{ .m = m, .n_q = n_q, .n_threads = @max(n_threads, 1) };
        errdefer s.deinit();
        s.lstm_state = try m.gpa.alloc(f32, 4 * m.init_conv.w.shape[0]);
        @memset(s.lstm_state, 0);
        for (&s.block_tail, m.blocks, ratios) |*tail, b, r| {
            const w = b.upsample.w; // [C_in][C_out][K]
            tail.* = try Act.init(m.gpa, w.shape[1], w.shape[2] - r);
        }
        return s;
    }

    pub fn deinit(s: *Stream) void {
        const gpa = s.m.gpa;
        s.init_hist.deinit(gpa);
        s.final_hist.deinit(gpa);
        for (s.block_hist) |a| a.deinit(gpa);
        for (s.block_tail) |a| a.deinit(gpa);
        gpa.free(s.lstm_state);
    }

    /// `codes` is this chunk's [n_q][n_frames]; writes n_frames * hop_length samples to `audio`.
    pub fn decode(s: *Stream, codes: []const i32, n_frames: usize, audio: []f32) Error!void {
        const m = s.m;
        const gpa = m.gpa;
        const h_dim = m.hidden_dim;
        const T = n_frames;
        const nt = s.n_threads;

        if (T < 1 or codes.len != s.n_q * T or audio.len != T * hop_length) return error.InvalidCodes;
        for (codes) |code| if (code < 0 or code >= m.n_bins) return error.InvalidCodes;

        // residual vector quantizer: sum the selected embeddings
        var x = try Act.init(gpa, h_dim, T);
        for (0..s.n_q) |q| {
            const embed = m.codebooks[q].data;
            for (0..T) |t| {
                const e = embed[@as(usize, @intCast(codes[q * T + t])) * h_dim ..][0..h_dim];
                for (e, 0..) |v, d| x.row(d)[t] += v;
            }
        }

        x = try replace(gpa, x, try conv1d(gpa, x, m.init_conv, &s.init_hist, nt));

        // 2-layer LSTM with skip connection (time-major internally)
        {
            const C = x.c;
            const seq = try gpa.alloc(f32, T * C);
            defer gpa.free(seq);
            const y = try gpa.alloc(f32, T * C);
            defer gpa.free(y);
            for (0..C) |c| for (0..T) |t| {
                seq[t * C + c] = x.row(c)[t];
            };

            try lstmForward(gpa, seq, y, T, C, m.lstm[0], s.lstm_state[0 .. 2 * C], nt);
            try lstmForward(gpa, y, seq, T, C, m.lstm[1], s.lstm_state[2 * C .. 4 * C], nt);

            for (0..C) |c| for (0..T) |t| {
                x.row(c)[t] += seq[t * C + c];
            };
        }

        for (m.blocks, ratios, &s.block_hist, &s.block_tail) |b, r, *hist, *tail| {
            elu(x);
            x = try replace(gpa, x, try convTranspose1d(gpa, x, b.upsample, tail, r, nt));

            const y0 = try Act.init(gpa, x.c, x.t);
            @memcpy(y0.d, x.d);
            elu(y0);
            const y1 = try replace(gpa, y0, try conv1d(gpa, y0, b.conv1, hist, nt));
            elu(y1);
            const y = try replace(gpa, y1, try conv1d(gpa, y1, b.conv2, null, nt));
            defer y.deinit(gpa);

            x = try replace(gpa, x, try conv1d(gpa, x, b.shortcut, null, nt));
            for (x.d, y.d) |*a, v| a.* += v;
        }

        elu(x);
        const out = try replace(gpa, x, try conv1d(gpa, x, m.final_conv, &s.final_hist, nt));
        defer out.deinit(gpa);
        @memcpy(audio, out.d);
    }
};

// ---------------------------------------------------------------------------
// loading

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.bytes.len - r.pos < n) return error.InvalidWeights;
        defer r.pos += n;
        return r.bytes[r.pos..][0..n];
    }

    fn int(r: *Reader) !i32 {
        return std.mem.readInt(i32, (try r.take(4))[0..4], .little);
    }
};

fn parse(gpa: Allocator, bytes: []const u8) !Model {
    var r = Reader{ .bytes = bytes };
    if (try r.int() != ggml_magic) return error.InvalidWeights;

    // in_channels, hidden_dim, n_filters, kernel_size, res_kernel_size, n_bins, bandwidth, sr, ftype
    var hp: [9]i32 = undefined;
    for (&hp) |*v| v.* = try r.int();
    if (hp[0] != 1 or hp[1] <= 0 or hp[2] != 32 or hp[3] != 7 or hp[4] != 3 or hp[5] <= 0 or hp[7] <= 0)
        return error.InvalidWeights;

    var tensors: std.ArrayList(Tensor) = .empty;
    errdefer {
        for (tensors.items) |t| {
            gpa.free(t.name);
            gpa.free(t.data);
        }
        tensors.deinit(gpa);
    }

    while (r.pos < bytes.len) {
        const n_dims = try r.int();
        const name_len = try r.int();
        const ftype = try r.int();
        if (n_dims < 0 or n_dims > 4 or name_len <= 0 or name_len > 256 or (ftype != 0 and ftype != 1))
            return error.InvalidWeights;

        var t = Tensor{ .name = undefined, .n_dims = @intCast(n_dims), .data = undefined };
        var numel: usize = 1;
        var i = t.n_dims;
        while (i > 0) { // stored innermost first
            i -= 1;
            const d = try r.int();
            if (d <= 0) return error.InvalidWeights;
            t.shape[i] = @intCast(d);
            numel *= t.shape[i];
        }
        const name = try r.take(@intCast(name_len));
        const raw = try r.take(numel * @as(usize, if (ftype == 1) 2 else 4));

        const keep = std.mem.startsWith(u8, name, "decoder.") or std.mem.startsWith(u8, name, "quantizer.");
        if (!keep) continue;

        t.data = try gpa.alloc(f32, numel);
        t.name = gpa.dupe(u8, name) catch |err| {
            gpa.free(t.data);
            return err;
        };
        for (t.data, 0..) |*v, j| {
            v.* = if (ftype == 1)
                @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[2 * j ..][0..2], .little))))
            else
                @bitCast(std.mem.readInt(u32, raw[4 * j ..][0..4], .little));
        }
        tensors.append(gpa, t) catch |err| {
            gpa.free(t.name);
            gpa.free(t.data);
            return err;
        };
    }

    var m = Model{
        .gpa = gpa,
        .hidden_dim = @intCast(hp[1]),
        .n_bins = @intCast(hp[5]),
        .sample_rate = @intCast(hp[7]),
        .tensors = try tensors.toOwnedSlice(gpa),
        .codebooks = &.{},
        .init_conv = undefined,
        .lstm = undefined,
        .blocks = undefined,
        .final_conv = undefined,
    };
    errdefer m.deinit();
    try bind(&m);
    return m;
}

fn find(m: *const Model, name: []const u8) ?*const Tensor {
    for (m.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

fn get(m: *const Model, name: []const u8, numel: usize) !*const Tensor {
    const t = find(m, name) orelse {
        std.log.err("zencodec: missing tensor '{s}'", .{name});
        return error.InvalidWeights;
    };
    if (t.data.len != numel) {
        std.log.err("zencodec: tensor '{s}' has unexpected size", .{name});
        return error.InvalidWeights;
    }
    return t;
}

/// weight shape [a][b][k]; bias has n_out entries
fn bindConv(m: *const Model, prefix: []const u8, a: usize, b: usize, k: usize, n_out: usize) !Conv {
    var buf: [128]u8 = undefined;
    const w = try get(m, try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), a * b * k);
    if (w.n_dims != 3) return error.InvalidWeights;
    return .{ .w = w, .b = try get(m, try std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}), n_out) };
}

fn bind(m: *Model) !void {
    const H = m.hidden_dim;
    var buf: [128]u8 = undefined;

    var codebooks: std.ArrayList(*const Tensor) = .empty;
    errdefer codebooks.deinit(m.gpa);
    while (find(m, try std.fmt.bufPrint(&buf, "quantizer.vq.layers.{d}._codebook.embed", .{codebooks.items.len}))) |t| {
        if (t.data.len != m.n_bins * H) return error.InvalidWeights;
        try codebooks.append(m.gpa, t);
    }
    if (codebooks.items.len == 0) {
        std.log.err("zencodec: no quantizer codebooks found", .{});
        return error.InvalidWeights;
    }
    m.codebooks = try codebooks.toOwnedSlice(m.gpa);

    var ch: usize = 512; // n_filters * 2^4
    m.init_conv = try bindConv(m, "decoder.model.0.conv.conv", ch, H, 7, ch);

    for (&m.lstm, 0..) |*l, i| {
        l.* = .{
            .w_ih = try get(m, try std.fmt.bufPrint(&buf, "decoder.model.1.lstm.weight_ih_l{d}", .{i}), 4 * ch * ch),
            .w_hh = try get(m, try std.fmt.bufPrint(&buf, "decoder.model.1.lstm.weight_hh_l{d}", .{i}), 4 * ch * ch),
            .b_ih = try get(m, try std.fmt.bufPrint(&buf, "decoder.model.1.lstm.bias_ih_l{d}", .{i}), 4 * ch),
            .b_hh = try get(m, try std.fmt.bufPrint(&buf, "decoder.model.1.lstm.bias_hh_l{d}", .{i}), 4 * ch),
        };
    }

    for (&m.blocks, ratios, 0..) |*b, r, i| {
        const up = 3 * (i + 1);
        const res = up + 1;
        const out = ch / 2;
        b.* = .{
            .upsample = try bindConv(m, try std.fmt.bufPrint(&buf, "decoder.model.{d}.convtr.convtr", .{up}), ch, out, 2 * r, out),
            .conv1 = try bindConv(m, try std.fmt.bufPrint(&buf, "decoder.model.{d}.block.1.conv.conv", .{res}), out / 2, out, 3, out / 2),
            .conv2 = try bindConv(m, try std.fmt.bufPrint(&buf, "decoder.model.{d}.block.3.conv.conv", .{res}), out, out / 2, 1, out),
            .shortcut = try bindConv(m, try std.fmt.bufPrint(&buf, "decoder.model.{d}.shortcut.conv.conv", .{res}), out, out, 1, out),
        };
        ch = out;
    }

    m.final_conv = try bindConv(m, "decoder.model.15.conv.conv", 1, ch, 7, 1);
}

// ---------------------------------------------------------------------------
// math helpers

fn dot(a: []const f32, b: []const f32) f32 {
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        acc += @as(Vec, a[i..][0..lanes].*) * @as(Vec, b[i..][0..lanes].*);
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// dst += w * src
fn axpy(dst: []f32, w: f32, src: []const f32) void {
    const wv: Vec = @splat(w);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        dst[i..][0..lanes].* = @as(Vec, dst[i..][0..lanes].*) + wv * @as(Vec, src[i..][0..lanes].*);
    }
    while (i < dst.len) : (i += 1) dst[i] += w * src[i];
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn elu(x: Act) void {
    for (x.d) |*v| v.* = if (v.* > 0) v.* else std.math.expm1(v.*);
}

/// Frees `old` and returns `new`, so stages can be chained.
fn replace(gpa: Allocator, old: Act, new: Act) !Act {
    old.deinit(gpa);
    return new;
}

/// Run `func(ctx, i)` for i in [0, n) across up to `n_threads` threads.
fn parallelFor(n: usize, n_threads: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize) void) !void {
    const nt = @min(n_threads, n, 64);
    if (nt <= 1) {
        for (0..n) |i| func(ctx, i);
        return;
    }
    const Worker = struct {
        fn run(next: *std.atomic.Value(usize), total: usize, c: @TypeOf(ctx)) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= total) return;
                func(c, i);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    var threads: [64]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (threads[0..spawned]) |t| t.join();
    for (1..nt) |_| {
        threads[spawned] = std.Thread.spawn(.{}, Worker.run, .{ &next, n, ctx }) catch break;
        spawned += 1;
    }
    Worker.run(&next, n, ctx);
}

// ---------------------------------------------------------------------------
// layers

/// encodec's pad1d in reflect mode, including its workaround for inputs
/// shorter than the padding.
fn padReflect(gpa: Allocator, x: Act, left: usize, right: usize) !Act {
    const max_pad = @max(left, right);
    const extra = if (x.t <= max_pad) max_pad - x.t + 1 else 0;
    const L: isize = @intCast(x.t + extra); // zero-extended length

    const out = try Act.init(gpa, x.c, x.t + left + right);
    for (0..x.c) |c| {
        const src = x.row(c);
        for (out.row(c), 0..) |*dst, i| {
            var j: isize = @as(isize, @intCast(i)) - @as(isize, @intCast(left));
            if (j < 0) j = -j;
            if (j >= L) j = 2 * (L - 1) - j;
            const ju: usize = @intCast(j);
            dst.* = if (ju < x.t) src[ju] else 0;
        }
    }
    return out;
}

/// Causal stride-1 conv1d. w: [C_out][C_in][K].
/// `hist` carries the last K - 1 input samples from one chunk to the next. On the
/// first chunk (hist still empty) the input is reflect-padded instead, as in encodec.
/// Unused (may be null) when K == 1.
fn conv1d(gpa: Allocator, x: Act, cv: Conv, hist: ?*Act, n_threads: usize) !Act {
    const c_out = cv.w.shape[0];
    const K = cv.w.shape[2];

    var padded = x;
    if (K > 1) {
        const h = hist.?;
        if (h.d.len == 0) {
            padded = try padReflect(gpa, x, K - 1, 0);
            h.* = Act.init(gpa, x.c, K - 1) catch |err| {
                padded.deinit(gpa);
                return err;
            };
        } else {
            padded = try Act.init(gpa, x.c, x.t + K - 1);
            for (0..x.c) |c| {
                @memcpy(padded.row(c)[0 .. K - 1], h.row(c));
                @memcpy(padded.row(c)[K - 1 ..], x.row(c));
            }
        }
        for (0..x.c) |c| @memcpy(h.row(c), padded.row(c)[x.t..]);
    }
    defer if (K > 1) padded.deinit(gpa);

    const out = try Act.init(gpa, c_out, x.t);
    const Ctx = struct {
        in: Act,
        out: Act,
        cv: Conv,

        fn run(c: *const @This(), co: usize) void {
            const c_in = c.cv.w.shape[1];
            const k_size = c.cv.w.shape[2];
            const dst = c.out.row(co);
            @memset(dst, c.cv.b.data[co]);
            for (0..c_in) |ci| {
                const wk = c.cv.w.data[(co * c_in + ci) * k_size ..][0..k_size];
                const src = c.in.row(ci);
                for (wk, 0..) |wv, k| axpy(dst, wv, src[k..][0..dst.len]);
            }
        }
    };
    try parallelFor(c_out, n_threads, &Ctx{ .in = padded, .out = out, .cv = cv }, Ctx.run);
    return out;
}

/// Causal conv-transpose1d. w: [C_in][C_out][K].
/// Each chunk's last K - stride output samples also depend on the next chunk, so
/// they are kept in `tail` ([C_out][K - stride], zeros at the start) and added to
/// the next chunk's output; at the end of the stream they are dropped, which is
/// encodec's causal trim.
/// Computed in polyphase form: output sample t * stride + r only receives taps
/// k = r + j * stride from input t - j, so each tap is a contiguous axpy into
/// phase r's buffer.
fn convTranspose1d(gpa: Allocator, x: Act, cv: Conv, tail: *Act, stride: usize, n_threads: usize) !Act {
    const c_out = cv.w.shape[1];
    const K = cv.w.shape[2];
    const L = x.t + (K + stride - 1) / stride; // per-phase buffer length

    // per-output-channel scratch, allocated up front so workers cannot fail
    const scratch = try gpa.alloc(f32, c_out * stride * L);
    defer gpa.free(scratch);

    const out = try Act.init(gpa, c_out, x.t * stride);
    const Ctx = struct {
        x: Act,
        out: Act,
        tail: Act,
        cv: Conv,
        stride: usize,
        L: usize,
        scratch: []f32,

        fn run(c: *const @This(), co: usize) void {
            const c_in = c.cv.w.shape[0];
            const n_out = c.cv.w.shape[1];
            const k_size = c.cv.w.shape[2];
            const s = c.stride;
            const acc = c.scratch[co * s * c.L ..][0 .. s * c.L];
            @memset(acc, 0);
            for (0..c_in) |ci| {
                const wk = c.cv.w.data[(ci * n_out + co) * k_size ..][0..k_size];
                const src = c.x.row(ci);
                for (wk, 0..) |wv, k| axpy(acc[(k % s) * c.L + k / s ..][0..c.x.t], wv, src);
            }
            const dst = c.out.row(co);
            const b = c.cv.b.data[co];
            for (0..s) |r| {
                const a = acc[r * c.L ..];
                for (0..c.x.t) |t| dst[t * s + r] = a[t] + b;
            }
            // add the previous chunk's spill-over, then keep this chunk's
            const spill = c.tail.row(co);
            for (dst[0..spill.len], spill) |*d, v| d.* += v;
            for (spill, 0..) |*v, i| {
                const pos = c.x.t * s + i;
                v.* = acc[(pos % s) * c.L + pos / s];
            }
        }
    };
    try parallelFor(c_out, n_threads, &Ctx{ .x = x, .out = out, .tail = tail.*, .cv = cv, .stride = stride, .L = L, .scratch = scratch }, Ctx.run);
    return out;
}

/// Single LSTM layer over x: [T][H] (time-major), writes y: [T][H].
/// `state` holds h and c ([H] each) across chunks.
fn lstmForward(gpa: Allocator, x: []const f32, y: []f32, T: usize, H: usize, l: LstmLayer, state: []f32, n_threads: usize) !void {
    const G = 4 * H; // gates i, f, g, o

    // input projections for all timesteps at once
    const xg = try gpa.alloc(f32, T * G);
    defer gpa.free(xg);
    const Ctx = struct {
        x: []const f32,
        xg: []f32,
        l: LstmLayer,
        H: usize,

        fn run(c: *const @This(), t: usize) void {
            const xt = c.x[t * c.H ..][0..c.H];
            for (c.xg[t * 4 * c.H ..][0 .. 4 * c.H], 0..) |*v, g| {
                v.* = c.l.b_ih.data[g] + c.l.b_hh.data[g] + dot(c.l.w_ih.data[g * c.H ..][0..c.H], xt);
            }
        }
    };
    try parallelFor(T, n_threads, &Ctx{ .x = x, .xg = xg, .l = l, .H = H }, Ctx.run);

    const gates = try gpa.alloc(f32, G);
    defer gpa.free(gates);
    const h = state[0..H];
    const c = state[H .. 2 * H];

    for (0..T) |t| {
        for (gates, 0..) |*v, g| v.* = xg[t * G + g] + dot(l.w_hh.data[g * H ..][0..H], h);
        for (0..H) |i| {
            const ig = sigmoid(gates[i]);
            const fg = sigmoid(gates[H + i]);
            const gg = std.math.tanh(gates[2 * H + i]);
            const og = sigmoid(gates[3 * H + i]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * std.math.tanh(c[i]);
        }
        @memcpy(y[t * H ..][0..H], h);
    }
}
