// zencodec thread pool (internal). See docs/threading.md.
//
// A pool owns n_threads - 1 worker threads, created once and reused for every
// parallel loop. zc_pool_run is a fork-join: the calling thread runs tasks too, and
// returns when all n tasks are done. Between jobs the workers spin for a short while
// (ZENCODEC_SPIN_US, default 100 us), so back-to-back jobs such as the LSTM's
// per-frame steps start in about a microsecond; after that they sleep on a condition
// variable and cost nothing.
//
// A pool runs one job at a time and is not reentrant: tasks must not call
// zc_pool_run. A NULL pool, or a pool of 1 thread, runs tasks on the caller in order.
#ifndef ZENCODEC_POOL_H
#define ZENCODEC_POOL_H

typedef struct zc_pool zc_pool;
typedef void (*zc_task_fn)(void * ctx, int i);

zc_pool * zc_pool_new(int n_threads); // NULL on failure or when n_threads <= 1
void      zc_pool_free(zc_pool * pool);
int       zc_pool_threads(const zc_pool * pool); // including the caller; 1 for NULL

// Runs fn(ctx, i) for every i in [0, n), each exactly once, in any order and on any
// of the pool's threads.
void zc_pool_run(zc_pool * pool, int n, zc_task_fn fn, void * ctx);

#endif // ZENCODEC_POOL_H
