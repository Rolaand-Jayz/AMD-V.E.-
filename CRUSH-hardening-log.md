# CRUSH Hardening Log

Purpose: Record hardening and optimization reviews for the AMD Video Enhancer project.

---

## 2026-03-16: SwsContext Caching Optimization

### Risks Reviewed

| Risk Area | Assessment |
|-----------|------------|
| Thread safety | **Mitigated** - Uses `std::mutex` to protect cache access |
| Memory leak | **Mitigated** - Static `SwsCacheCleaner` frees all contexts on program exit |
| Memory growth | **Low** - Cache is bounded by unique (srcW, srcH, srcFmt, dstW, dstH, dstFmt) combinations; typical video processing uses 1-2 combinations |
| Quality regression | **None** - Same conversion logic, just cached context |

### Changes Made

**File**: `src/frame_io.cpp`

1. Added `SwsKey` struct to uniquely identify conversion parameters
2. Added `SwsKeyHash` for `unordered_map` key hashing
3. Added static `SwsCache` (`unordered_map<SwsKey, SwsContext*>`) with mutex protection
4. Added `getOrCreateSwsContext()` to get cached or create new context
5. Added `clearSwsCache()` for explicit cleanup
6. Added static `SwsCacheCleaner` for automatic cleanup on program exit
7. Modified `avFrameToRgb24()` to use cached context instead of per-frame allocation
8. Modified `rgb24ToAvFrame()` to use cached context instead of per-frame allocation

### Performance Reasoning

- `sws_getContext()` performs non-trivial allocation and setup work (lookup tables, filter coefficients, scaler context)
- For a 60-second video at 30fps, this was called ~1800 times per conversion function
- Caching eliminates redundant allocation/deallocation for identical conversion parameters
- Expected impact: measurable reduction in CPU overhead during frame conversion phase

### Verification

- Build: Successful (`cmake --build build-main-merge -j$(nproc)`)
- Tests: 2/2 passed (`ctest --test-dir build-main-merge --output-on-failure`)
  - `planner_tests`: Passed (0.01s)
  - `tensor_contract_tests`: Passed (0.00s)

### Residual Risks

- None identified. The cache is self-managing and thread-safe.

---

## Previous Optimizations (Pre-existing)

The following optimizations were already in place before this review:

| Optimization | Location | Description |
|--------------|----------|-------------|
| Tile batching | `migraphx_backend.cpp:698-732` | Adaptive batch sizing based on frame/tile dimensions |
| Async encode thread | `migraphx_backend.cpp:2114-2210` | Double-buffered output queue for parallel encode |
| Parallel pixel ops | `frame_io.cpp:35-66` | `parallelForPixels()` helper for multi-threaded processing |
| Pipe I/O tuning | `migraphx_backend.cpp:603-616` | `setvbuf` and `F_SETPIPE_SZ` for pipe performance |
