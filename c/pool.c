// Persistent fork-join thread pool. See pool.h and docs/threading.md.
//
// The current job is published through one 64-bit atomic `ticket` holding
// (generation, task count, next task index). Workers claim tasks with a compare-and-swap
// on the ticket, which only succeeds while the ticket still carries the generation the
// worker woke up for. A job is over when all its tasks have finished, whether or not
// every worker took part: a worker that was descheduled and wakes up late finds a newer
// generation, or no tasks left, and goes back to waiting. So one slow or preempted
// thread never holds up a job it hasn't started a task of.

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE // pthread QoS
#elif !defined(_GNU_SOURCE)
#define _GNU_SOURCE // clock_gettime, sched_yield
#endif

#include "pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#ifndef ZENCODEC_SPIN_US
#define ZENCODEC_SPIN_US 100 // how long idle workers poll before sleeping
#endif
#define MAX_THREADS 64

// ticket = gen << 40 | n << 20 | next
#define TASK_BITS 20 // up to 2^20 - 1 tasks per job
#define TASK_MASK ((1u << TASK_BITS) - 1)
#define GEN_SHIFT (2 * TASK_BITS)
static inline uint32_t t_gen (uint64_t t) { return (uint32_t) (t >> GEN_SHIFT); }
static inline int      t_n   (uint64_t t) { return (int) ((t >> TASK_BITS) & TASK_MASK); }
static inline int      t_next(uint64_t t) { return (int) (t & TASK_MASK); }

struct zc_pool {
    int       n_workers;
    pthread_t threads[MAX_THREADS];

    // the current job, written by the caller before it publishes the ticket and
    // unchanged until all its tasks are finished
    zc_task_fn fn;
    void     * ctx;

    _Atomic uint64_t ticket;
    atomic_int       finished; // tasks of the current job that have returned
    atomic_bool      stop;
    atomic_int       sleeping; // workers waiting on `wake`
    pthread_mutex_t  lock;
    pthread_cond_t   wake;
};

static inline void cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Runs tasks of generation `gen` until there are none left to claim.
static void run_tasks(zc_pool * p, uint32_t gen) {
    uint64_t t = atomic_load_explicit(&p->ticket, memory_order_acquire);
    for (;;) {
        if (t_gen(t) != gen || t_next(t) >= t_n(t)) return;
        if (!atomic_compare_exchange_weak_explicit(&p->ticket, &t, t + 1, memory_order_acquire,
                                                   memory_order_acquire))
            continue; // t now holds the current ticket
        p->fn(p->ctx, t_next(t)); // fn and ctx are stable: this job can't end before we do
        atomic_fetch_add_explicit(&p->finished, 1, memory_order_release);
        t = atomic_load_explicit(&p->ticket, memory_order_acquire);
    }
}

// Waits for a generation other than `seen`: spins first, then sleeps.
static uint32_t wait_for_job(zc_pool * p, uint32_t seen) {
    uint32_t g;
    const double deadline = now_us() + ZENCODEC_SPIN_US;
    for (int spins = 0;; spins++) {
        if ((g = t_gen(atomic_load(&p->ticket))) != seen) return g;
        cpu_relax();
        if ((spins & 255) == 255 && now_us() > deadline) break;
    }
    pthread_mutex_lock(&p->lock);
    atomic_fetch_add(&p->sleeping, 1);
    while ((g = t_gen(atomic_load(&p->ticket))) == seen) pthread_cond_wait(&p->wake, &p->lock);
    atomic_fetch_sub(&p->sleeping, 1);
    pthread_mutex_unlock(&p->lock);
    return g;
}

static void * worker(void * arg) {
    zc_pool * p = arg;
    uint32_t seen = 0;
    for (;;) {
        seen = wait_for_job(p, seen);
        if (atomic_load(&p->stop)) return NULL;
        run_tasks(p, seen);
    }
}

// Publishes a new generation with n tasks and wakes sleeping workers. The seq_cst
// store of the ticket and load of `sleeping` pair with the worker's increment of
// `sleeping` and load of the ticket: either the worker sees the new job, or we see
// the sleeper and signal it.
static uint32_t publish(zc_pool * p, int n) {
    const uint32_t gen = (t_gen(atomic_load_explicit(&p->ticket, memory_order_relaxed)) + 1) & 0xffffff;
    atomic_store(&p->ticket, (uint64_t) gen << GEN_SHIFT | (uint64_t) n << TASK_BITS);
    if (atomic_load(&p->sleeping) > 0) {
        pthread_mutex_lock(&p->lock);
        pthread_cond_broadcast(&p->wake);
        pthread_mutex_unlock(&p->lock);
    }
    return gen;
}

zc_pool * zc_pool_new(int n_threads) {
    if (n_threads <= 1) return NULL;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    zc_pool * p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->wake, NULL);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef __APPLE__
    // workers run at the creating thread's QoS, so they get the same (performance or
    // efficiency) cores as the thread that decodes
    pthread_attr_set_qos_class_np(&attr, qos_class_self(), 0);
#endif
    for (int t = 1; t < n_threads; t++) {
        if (pthread_create(&p->threads[p->n_workers], &attr, worker, p) == 0) p->n_workers++;
    }
    pthread_attr_destroy(&attr);
    if (p->n_workers == 0) {
        zc_pool_free(p);
        return NULL;
    }
    return p;
}

void zc_pool_free(zc_pool * p) {
    if (!p) return;
    atomic_store(&p->stop, true);
    publish(p, 0);
    for (int t = 0; t < p->n_workers; t++) pthread_join(p->threads[t], NULL);
    pthread_cond_destroy(&p->wake);
    pthread_mutex_destroy(&p->lock);
    free(p);
}

int zc_pool_threads(const zc_pool * p) { return p ? p->n_workers + 1 : 1; }

void zc_pool_run(zc_pool * p, int n, zc_task_fn fn, void * ctx) {
    if (!p || n <= 1 || n > (int) TASK_MASK) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    p->fn  = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->finished, 0, memory_order_relaxed);
    run_tasks(p, publish(p, n));
    // the remaining tasks are running on workers: spin, and yield if one was preempted
    for (int spins = 0; atomic_load_explicit(&p->finished, memory_order_acquire) < n; spins++) {
        if (spins < 4096) cpu_relax();
        else              sched_yield();
    }
}
