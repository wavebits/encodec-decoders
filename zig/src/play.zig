//! Blocking mono float playback.
//! macOS: AudioQueue (AudioToolbox). Linux: pipes raw PCM into `aplay`.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// Play `samples` in [-1, 1] at `sample_rate`; returns once playback ends.
pub fn play(io: Io, samples: []const f32, sample_rate: u32) !void {
    var ready = std.atomic.Value(usize).init(samples.len);
    var done = std.atomic.Value(bool).init(true);
    try playStream(io, samples.ptr, &ready, &done, sample_rate);
}

/// Like play, but another thread is still writing the samples: only
/// samples[0..ready) are available so far, and the writer raises `ready` as it goes
/// (release order) and sets `done` after its last sample. Playback starts right away,
/// plays silence if it catches up with the writer, and ends once `done` is set and
/// every ready sample has been played.
pub fn playStream(io: Io, samples: [*]const f32, ready: *const std.atomic.Value(usize), done: *const std.atomic.Value(bool), sample_rate: u32) !void {
    switch (builtin.os.tag) {
        .macos => try macos.play(io, samples, ready, done, sample_rate),
        .linux => try playAplay(io, samples, ready, done, sample_rate),
        else => return error.Unsupported,
    }
}

fn playAplay(io: Io, samples: [*]const f32, ready: *const std.atomic.Value(usize), done: *const std.atomic.Value(bool), sample_rate: u32) !void {
    var rate_buf: [16]u8 = undefined;
    const rate = try std.fmt.bufPrint(&rate_buf, "{d}", .{sample_rate});
    var child = try std.process.spawn(io, .{
        .argv = &.{ "aplay", "-q", "-t", "raw", "-f", "FLOAT_LE", "-c", "1", "-r", rate },
        .stdin = .pipe,
    });
    const stdin = child.stdin.?;
    // aplay inserts the silence itself if we fall behind
    var written: usize = 0;
    while (true) {
        const fin = done.load(.acquire); // before reading ready
        const avail = ready.load(.acquire);
        if (avail == written) {
            if (fin) break;
            try io.sleep(.fromMilliseconds(5), .awake);
            continue;
        }
        stdin.writeStreamingAll(io, std.mem.sliceAsBytes(samples[written..avail])) catch |err| {
            stdin.close(io);
            child.stdin = null;
            _ = child.wait(io) catch {};
            return err;
        };
        written = avail;
    }
    stdin.close(io);
    child.stdin = null;
    switch (try child.wait(io)) {
        .exited => |code| if (code != 0) return error.PlaybackFailed,
        else => return error.PlaybackFailed,
    }
}

const macos = struct {
    // Minimal AudioToolbox bindings.
    const OSStatus = i32;
    const AudioQueueRef = *opaque {};

    const AudioStreamBasicDescription = extern struct {
        mSampleRate: f64,
        mFormatID: u32,
        mFormatFlags: u32,
        mBytesPerPacket: u32,
        mFramesPerPacket: u32,
        mBytesPerFrame: u32,
        mChannelsPerFrame: u32,
        mBitsPerChannel: u32,
        mReserved: u32 = 0,
    };

    const AudioQueueBuffer = extern struct {
        mAudioDataBytesCapacity: u32,
        mAudioData: [*]u8,
        mAudioDataByteSize: u32,
        mUserData: ?*anyopaque,
        mPacketDescriptionCapacity: u32,
        mPacketDescriptions: ?*anyopaque,
        mPacketDescriptionCount: u32,
    };

    const OutputCallback = *const fn (?*anyopaque, AudioQueueRef, *AudioQueueBuffer) callconv(.c) void;

    extern "c" fn AudioQueueNewOutput(*const AudioStreamBasicDescription, OutputCallback, ?*anyopaque, ?*anyopaque, ?*anyopaque, u32, *AudioQueueRef) OSStatus;
    extern "c" fn AudioQueueAllocateBuffer(AudioQueueRef, u32, **AudioQueueBuffer) OSStatus;
    extern "c" fn AudioQueueEnqueueBuffer(AudioQueueRef, *AudioQueueBuffer, u32, ?*const anyopaque) OSStatus;
    extern "c" fn AudioQueueStart(AudioQueueRef, ?*const anyopaque) OSStatus;
    extern "c" fn AudioQueueStop(AudioQueueRef, u8) OSStatus;
    extern "c" fn AudioQueueDispose(AudioQueueRef, u8) OSStatus;
    extern "c" fn AudioQueueGetProperty(AudioQueueRef, u32, *anyopaque, *u32) OSStatus;

    const kAudioFormatLinearPCM: u32 = 0x6c70636d; // 'lpcm'
    const kLinearPCMFormatFlagIsFloat: u32 = 1 << 0;
    const kLinearPCMFormatFlagIsPacked: u32 = 1 << 3;
    const kAudioQueueProperty_IsRunning: u32 = 0x6171726e; // 'aqrn'

    const n_buffers = 3;
    const buffer_frames = 4096;
    const silence_frames = 480; // 20 ms, played when the decoder has not caught up yet

    const Player = struct {
        samples: [*]const f32,
        pos: usize = 0,
        ready: *const std.atomic.Value(usize),
        done: *const std.atomic.Value(bool),
        n_idle: std.atomic.Value(usize) = .init(0), // buffers returned with nothing left to play
    };

    fn fill(user: ?*anyopaque, q: AudioQueueRef, buf: *AudioQueueBuffer) callconv(.c) void {
        const p: *Player = @ptrCast(@alignCast(user.?));
        const done = p.done.load(.acquire); // before reading ready
        const n = @min(p.ready.load(.acquire) - p.pos, buffer_frames);
        if (n == 0 and done) {
            _ = p.n_idle.fetchAdd(1, .release);
            return;
        }
        if (n == 0) { // underrun: keep the buffer cycling with a little silence
            @memset(buf.mAudioData[0 .. silence_frames * @sizeOf(f32)], 0);
            buf.mAudioDataByteSize = silence_frames * @sizeOf(f32);
            _ = AudioQueueEnqueueBuffer(q, buf, 0, null);
            return;
        }
        const bytes = std.mem.sliceAsBytes(p.samples[p.pos..][0..n]);
        @memcpy(buf.mAudioData[0..bytes.len], bytes);
        buf.mAudioDataByteSize = @intCast(bytes.len);
        p.pos += n;
        _ = AudioQueueEnqueueBuffer(q, buf, 0, null);
    }

    fn play(io: Io, samples: [*]const f32, ready: *const std.atomic.Value(usize), done: *const std.atomic.Value(bool), sample_rate: u32) !void {
        const fmt = AudioStreamBasicDescription{
            .mSampleRate = @floatFromInt(sample_rate),
            .mFormatID = kAudioFormatLinearPCM,
            .mFormatFlags = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
            .mBytesPerPacket = @sizeOf(f32),
            .mFramesPerPacket = 1,
            .mBytesPerFrame = @sizeOf(f32),
            .mChannelsPerFrame = 1,
            .mBitsPerChannel = 32,
        };
        var p = Player{ .samples = samples, .ready = ready, .done = done };
        var q: AudioQueueRef = undefined;
        // null run loop: callbacks run on the queue's own thread
        if (AudioQueueNewOutput(&fmt, fill, &p, null, null, 0, &q) != 0) return error.AudioOutputUnavailable;
        defer _ = AudioQueueDispose(q, 1);

        for (0..n_buffers) |_| {
            var buf: *AudioQueueBuffer = undefined;
            if (AudioQueueAllocateBuffer(q, buffer_frames * @sizeOf(f32), &buf) != 0) break;
            fill(&p, q, buf);
        }
        if (AudioQueueStart(q, null) != 0) return error.AudioOutputUnavailable;

        // every buffer has come back empty -> all audio was handed to the device;
        // then stop without discarding and wait for the queue to wind down
        while (p.n_idle.load(.acquire) < n_buffers) try io.sleep(.fromMilliseconds(20), .awake);
        _ = AudioQueueStop(q, 0);
        while (true) {
            var running: u32 = 0;
            var size: u32 = @sizeOf(u32);
            _ = AudioQueueGetProperty(q, kAudioQueueProperty_IsRunning, &running, &size);
            if (running == 0) break;
            try io.sleep(.fromMilliseconds(20), .awake);
        }
    }
};
