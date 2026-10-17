# Threading: the thread pool and the parallel LSTM recurrence

This note explains how the C decoder uses threads: why it needs a persistent thread pool,
how the pool works, and how the LSTM recurrence, the one part of EnCodec that has to run
one frame at a time, is split across cores. The code is in `c/pool.{h,c}` and
`lstm_forward` / `lstm_step` in `c/zencodec.c`.

## The problem

Every layer of the decoder is a matrix product (see [architecture.md](architecture.md)
and `c/backend.h`), and the convolutions parallelize easily: each output channel, or
each tile of output, is independent. The LSTM is different. Its hidden state at frame
*t* depends on the state at frame *t* − 1:

```
gates(t) = W_ih · x(t) + b  +  W_hh · h(t-1)        4H = 2048 gate values, H = 512
c(t)     = σ(f) ⊙ c(t-1) + σ(i) ⊙ tanh(g)
h(t)     = σ(o) ⊙ tanh(c(t))
```

The input half, `W_ih · x(t)`, doesn't depend on the previous frame, so it is computed
for a whole chunk at once as one GEMM. The recurrent half, `W_hh · h(t-1)`, can't be:
it is a matrix-vector product (GEMV) with a 2048 × 512 matrix, once per frame and per
layer. That is 150 GEMVs per second of audio, each reading the whole 4 MB matrix
(f32) to do only 1 M multiply-adds. It is limited by memory bandwidth, not arithmetic.

Before this change, the recurrence ran on one thread and took about a third of the
decode time (30 s clip, Apple M1, 4 threads):

| Backend | Decode | LSTM recurrence |
| --- | --- | --- |
| neon | 760 ms | 266 ms (35%) |
| accelerate | 479 ms | 151 ms (32%) |

Splitting each GEMV across threads looks obvious, but it means synchronizing the threads
150 times per second of audio, about every 30–60 µs. The old `zc_parallel_for` created
and joined fresh threads for every call. That costs tens of µs per thread, as much as
the GEMV itself, so it could never have worked here. It was already costing time
elsewhere too: a 1 s chunk runs up to 37 parallel layers, so with 4 threads a 30 s clip
created and joined about 3,300 threads.

So the plan has two parts:

1. **A persistent thread pool** whose fork-join costs about a microsecond.
2. **A recurrence split by hidden unit**, so a step needs no synchronization beyond
   that fork-join.

## Part 1: the thread pool

`c/pool.h` is a small fork-join pool:

```c
zc_pool * zc_pool_new(int n_threads);        // starts n_threads - 1 workers
void      zc_pool_run(zc_pool * pool, int n, zc_task_fn fn, void * ctx); // fn(ctx, i), i in [0, n)
void      zc_pool_free(zc_pool * pool);
```

A `zencodec_stream` owns one pool for its lifetime. Every layer, and every backend
GEMM, runs its tasks on it. `zc_pool_run` publishes a job, runs tasks on the calling
thread too, and returns once all `n` tasks have finished.

### How a job is published

The whole job state is one 64-bit atomic, the *ticket*:

```
 63            40 39           20 19            0
┌────────────────┬───────────────┬───────────────┐
│ generation     │ task count n  │ next task     │
└────────────────┴───────────────┴───────────────┘
```

- **Publishing:** the caller writes `fn` and `ctx`, then stores a ticket with the next
  generation, `n`, and `next = 0`.
- **Claiming:** a thread claims task `next` by compare-and-swap of the ticket to
  `next + 1`. The swap only succeeds if the ticket still has the generation the thread
  is working on.
- **Finishing:** each finished task increments a `finished` counter, and the caller
  waits until `finished == n`.

Because the generation is part of what is compared and swapped, a worker can never run
a task of a job it didn't see published. This matters for the next point.

### A job waits for tasks, not for workers

The first version of the pool ended a job when **every worker had checked in**, even
workers that found no task left. That is simpler, and on a quiet machine it is just as
fast. But when the OS deschedules one worker, for example because another process wants
the core, every job has to wait until that worker runs again, which can take a
scheduler time slice. And the recurrence runs 150 jobs per second of audio.

With the ticket, a job only waits for tasks that have already started. A worker that
wakes up late finds a newer generation, or no tasks left, and goes back to waiting.
The LSTM also uses **two tasks per thread**. If one thread starts late, the others take
over most of its share, and the step waits only for the one task it holds.

Measured with four busy-looping processes competing for the cores (neon, 4 threads,
30 s clip, median of 7 interleaved runs):

| Pool | Decode | LSTM recurrence |
| --- | --- | --- |
| no pool: threads created per layer (before) | 1878 ms | 592 ms |
| job ends when every worker has checked in | 1624 ms | 448 ms |
| job ends when every task is done (ticket) | **1574 ms** | **399 ms** |

On a quiet machine the two pool designs measure the same (84 and 87 ms).

### Spin, then sleep

Between jobs, a worker spins on the ticket, using the CPU's `yield` / `pause` hint, for
`ZENCODEC_SPIN_US` (default 100 µs). After that it sleeps on a condition variable.

- **While spinning,** a new job reaches the worker in well under a microsecond. Back-to-back
  jobs, like the LSTM's steps and the layers within a chunk, never pay for a wake-up.
- **Once asleep,** the worker costs nothing. A streaming decoder that waits for the
  network, or for the audio device, doesn't burn a core. The first job after a pause
  pays one wake-up, about 5–10 µs.

The caller only signals the condition variable when some worker is asleep, so the
common case makes no system call. The worker's "increment `sleeping`, then re-check the
ticket" and the caller's "store the ticket, then check `sleeping`" are sequentially
consistent atomics. So either the worker sees the new job, or the caller sees the
sleeper and wakes it, and a wake-up can't be lost.

Measured cost per job with empty tasks, against creating and joining threads as before
(Apple M1, which has 4 performance and 4 efficiency cores):

| Threads | Pool, workers spinning | Pool, workers asleep | Create + join threads (before) |
| --- | --- | --- | --- |
| 2 | 0.1–0.2 µs | 5–9 µs | 36 µs |
| 4 | 0.3 µs | 5–6 µs | 50 µs |
| 8 | 1.7 µs | 10–11 µs | 79 µs |

### Details

- **QoS on Apple platforms:** workers are created at the creating thread's QoS class.
  The CLI runs at `QOS_CLASS_USER_INITIATED`, so the workers stay on the performance
  cores with it.
- **Not reentrant:** a task must not call `zc_pool_run` on the same pool. The layers
  never nest; the LSTM step tasks call the backend's single-threaded `zc_gemv`.
- **Threads:** a stream with `n_threads = 1` has no pool and creates no threads.
  `zencodec_decode` makes a stream, and so a pool, for each call.

## Part 2: the parallel recurrence

### Split by hidden unit

Each hidden unit *j* needs four gate values (*i*, *f*, *g*, *o*) to update its `c[j]`
and `h[j]`. Those come from four rows of `W_hh`: rows *j*, *H*+*j*, 2*H*+*j* and 3*H*+*j*.
If each thread owns a set of hidden units, it can:

1. compute its units' gate rows of `W_hh · h(t-1)`, reading all of `h(t-1)`;
2. update `c` and `h` for its units only;
3. write its part of `h(t)`.

No thread needs another thread's gates. The only data shared between threads is
`h(t-1)`, which they all read and none write: `h(t)` goes to a second buffer, and the
two buffers swap every step. So each step is exactly one `zc_pool_run`, with no
locks and no reductions.

### Reordered gate rows

PyTorch stores the gate rows gate-major, `[i₀…i₅₁₁; f₀…f₅₁₁; g…; o…]`. A thread's units
would then be four separate row ranges. At load time, `bind` (`lstm_reorder`) reorders
the rows of `W_ih`, `W_hh` and the bias into blocks of `LSTM_U = 8` units:

```
row (b·4 + gate)·8 + u   =   gate `gate` of hidden unit b·8 + u
[ i₀…i₇ f₀…f₇ g₀…g₇ o₀…o₇ | i₈…i₁₅ f₈…f₁₅ … | … ]   64 blocks of 32 rows
```

Any run of whole blocks is then a contiguous row range that holds all four gates of its
units. That is what the backend interface's GEMV takes:

```c
void zc_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y);  // y += A[row0 : row0+rows] x
```

A block is 32 rows, a multiple of the NEON kernel's 8-row panels. The reordering is a
permutation applied once at load, so the output doesn't change. `W_ih` and the bias get
the same order, so `W_ih · x + b` comes out already in block order. It is also
transposed to time-major after its GEMM, so each step reads its 2048 inputs from one
contiguous run.

### One step

```c
static void lstm_step(void * p, int i) {          // task i of n_tasks, for frame t
    // blocks [b0, b1) → rows [r0, r0 + rows)
    memcpy(g, xg_t + r0, rows * sizeof(float));   // W_ih x(t) + b, precomputed
    zc_gemv(&w_hh, r0, rows, h_in, g);            // += W_hh h(t-1), this task's rows
    for (b = b0; b < b1; b++)                     // cell update, 8 units at a time:
        lstm_cell(g + ..., c + 8 * b, h_out + 8 * b);  // c = σ(f)·c + σ(i)·tanh(g), h = σ(o)·tanh(c)
    for (j = 8 * b0; j < 8 * b1; j++) y[j][t] = h_out[j];
}
```

`lstm_forward` runs it once per frame: `zc_pool_run(pool, n_tasks, lstm_step, &ctx)`,
then swaps `h_in` and `h_out`. With one thread, `n_tasks = 1` and it is one full GEMV,
as before.

Two details kept the 1-thread path from getting slower. The first versions cost 15 ms
and 40 ms extra on the 30 s clip:

- **The cell update is its own function with `restrict` pointers** (`lstm_cell`). Written
  inline, with the strided store to `y` in the same loop, the compiler's cost model
  refused to vectorize it. That made the gate math take 3 µs more per step.
- **The time-major copy of `W_ih · x` is done in 16 × 16 tiles.** A plain double loop
  writes with a stride of 8 KB and took 40 ms over the clip. In tiles it takes 8 ms.

### The accelerate backend: don't share the matrix unit

On the `accelerate` backend, splitting the GEMV made the recurrence *slower*. Every
thread called `cblas_sgemv` on its rows, and Accelerate runs those on the CPU's matrix
unit (AMX), which is shared by the whole CPU cluster. The threads just queued for it:

| accelerate, 4 threads | Recurrence |
| --- | --- |
| one `cblas_sgemv` per step (before) | 151 ms |
| `cblas_sgemv` split 8 ways | 143 ms |
| split 8 ways, NEON for partial row ranges (now) | **90 ms** |

So the accelerate backend's `zc_gemv` sends a partial row range, which means other
threads are running the rest, to the core's own NEON units. It uses a row-major
4-row kernel that reads the same weights as cblas, so it needs no second copy. A full
GEMV, as on one thread, still goes to `cblas_sgemv`. For one GEMV on one thread, AMX
is about twice as fast as NEON on one core (133 GB/s against 65–79 GB/s). Four cores'
worth of NEON beats one shared AMX.

### Results

Per stage, from profile builds (`-Dprofile=true`), 30 s clip, Apple M1, median of 5
interleaved runs:

| Backend | Threads | Decode, before → after | LSTM recurrence, before → after |
| --- | --- | --- | --- |
| neon | 1 | 1654 → 1654 ms | 266 → 261 ms |
| neon | 4 | 760 → **531 ms** (−30%) | 266 → **76 ms** (3.5×) |
| accelerate | 1 | 541 → 541 ms | 151 → 143 ms |
| accelerate | 4 | 479 → **404 ms** (−16%) | 151 → **90 ms** (1.7×) |

The rest of the gain on 4 threads is the pool. ELU, for instance, went from 47 to 31 ms
on neon, because it no longer creates threads 13 times per chunk.

End to end, with `bench/bench.py` (stripped builds, whole process, 5 runs; full table in
[bench/results/2026-10-17T1817_5c5dec9-dirty.md](../bench/results/2026-10-17T1817_5c5dec9-dirty.md)):

| 30 s clip, 4 threads | Wall, before → after | CPU time (user), before → after |
| --- | --- | --- |
| ref | 2.21 → 1.86 s | 6.43 → 7.12 s |
| neon | 0.87 → 0.68 s | 1.96 → 2.39 s |
| accelerate | 0.52 → 0.45 s | 0.60 → 0.96 s |

With 1 thread, all three backends are unchanged. With 4 threads, accelerate is now
faster than with 1 thread (0.45 against 0.58 s). Before, it was the other way round.

The CPU time goes up because workers spin between jobs instead of sleeping. That is the
price of 0.3 µs fork-joins. The spin only lasts while jobs keep coming, which is during
a decode. An idle stream costs no CPU.

Two caveats:

- **Under heavy load, accelerate's recurrence is slower when split.** With four
  busy-looping processes on the cores, the split recurrence took 376 ms against 258 ms
  for one `cblas_sgemv`. The other processes slow down the NEON cores but not the matrix
  unit. The whole decode was still faster (935 against 1074 ms), because the pool speeds
  up everything else.
- **Memory is unchanged.** Peak memory is 77–81 MB for all backends, before and after.
  The pool adds the workers' stacks, and the LSTM adds a 600 KB transposed copy of the
  input projections while it runs.

### Output

The work split doesn't change any sum. Each gate is still one dot product, over the
same `K = 512` inputs in the same order.

- **ref and neon:** output is bit-identical for every thread count, and identical to
  before this change.
- **accelerate:** with 1 thread, output is identical to before. With more threads, the
  NEON GEMV sums in a different order than cblas, so samples differ by up to 4 LSB of
  16-bit audio. That is well below the f16 weight rounding (see the README's Accuracy
  section).

The pool was checked with ThreadSanitizer, with no reports, and with a 40-run stress
test at 2–8 threads, in which every output was identical.

## What wasn't done, and what's next

- **Overlapping the two LSTM layers.** Layer 2 at frame *t* only needs layer 1 at frame
  *t*, so both could run in one step: a wavefront that halves the number of fork-joins.
  But layer 2's input projection would then become a GEMV per frame, instead of one
  GEMM over the chunk, which raises GEMV traffic from 2 to 3 matrices per frame. The
  fork-join now costs about 0.3 µs of a ~17 µs step (neon, 4 threads), so this would
  lose.
- **Splitting the GEMV over K** (the inputs) instead of over rows. Each thread would
  produce partial sums for every gate, which then need a reduction and a second
  synchronization. Splitting over rows needs neither.
- **Pinning tasks to threads.** Tasks are claimed dynamically, so a thread may read
  different rows of `W_hh` from step to step. On Apple silicon the cluster shares its L2
  cache, so this costs nothing. On CPUs with a private L2 per core (many Cortex-A cores,
  x86), giving each thread fixed rows could keep its slice of `W_hh` in its own cache.
  That needs measuring on such a device first.
- **f16 `W_hh`.** The recurrence is bandwidth-bound, so storing `W_hh` in f16 halves the
  bytes per step. This is now available as a build option for all weights
  (`-Df16=true`, see the README). It makes neon's GEMV 20% faster (43 against 54 µs) with
  bit-identical output, and saves about 23 MB of memory.
- **The Zig decoder** still runs the recurrence on one thread. It gets the same design
  when it gets backends.

## Tuning

- **`n_threads`:** use the number of performance cores. The CLI's default on Apple
  platforms is `hw.perflevel0.physicalcpu`. More threads than cores makes every step
  wait for preempted threads.
- **`-DZENCODEC_SPIN_US=…`:** how long idle workers spin before sleeping. Raise it if
  your app calls `zencodec_stream_decode` in quick succession and you can spare the
  power. Lower it, or set it to 0, on battery-critical devices that decode in short
  bursts.
