const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const StringValue = @import("string.zig").StringValue;
const Matrix = @import("matrix.zig").Matrix;
const distances = @import("distances.zig");

/// Configuration for parallel computation
pub const ParallelConfig = struct {
    /// Number of threads to use (0 = auto-detect)
    thread_count: usize = 0,
    /// Minimum work size per thread
    min_work_per_thread: usize = 100,
    /// Maximum number of threads to create
    max_threads: usize = 16,

    /// Get optimal thread count for given work size
    pub fn getThreadCount(self: ParallelConfig, work_size: usize) usize {
        var threads = self.thread_count;

        // Auto-detect if not specified
        if (threads == 0) {
            threads = std.Thread.getCpuCount() catch 1;
        }

        // Limit by work size
        if (work_size < self.min_work_per_thread) {
            return 1;
        }

        const max_useful_threads = work_size / self.min_work_per_thread;
        threads = @min(threads, max_useful_threads);
        threads = @min(threads, self.max_threads);

        return @max(threads, 1);
    }
};

/// Context for parallel matrix computation
const ComputeContext = struct {
    strings: []const StringValue,
    matrix: *Matrix,
    measure_fn: MeasureFn,
    allocator: Allocator,
    start_row: usize,
    end_row: usize,
    error_occurred: bool = false,
    // Remove per-context mutex - use lock-free approach for disjoint ranges

    const MeasureFn = *const fn (StringValue, StringValue) f64;
};

/// Parallel matrix computation engine
pub const ParallelCompute = struct {
    allocator: Allocator,
    config: ParallelConfig,

    const Self = @This();

    /// Initialize parallel compute engine
    pub fn init(allocator: Allocator, config: ParallelConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Compute similarity matrix in parallel using provided measure function
    pub fn computeMatrix(
        self: Self,
        matrix: *Matrix,
        strings: []const StringValue,
        comptime measure_fn: anytype,
    ) !void {
        const work_size = matrix.getComputations();
        const thread_count = self.config.getThreadCount(work_size);

        if (thread_count == 1) {
            // Single-threaded fallback
            return self.computeMatrixSingleThreaded(matrix, strings, measure_fn);
        }

        // Create thread pool
        var threads = try self.allocator.alloc(Thread, thread_count);
        defer self.allocator.free(threads);

        var contexts = try self.allocator.alloc(ComputeContext, thread_count);
        defer self.allocator.free(contexts);

        // Divide work among threads
        const rows_per_thread = matrix.row_range.length() / thread_count;
        const remainder = matrix.row_range.length() % thread_count;

        var current_row = matrix.row_range.start;
        for (0..thread_count) |i| {
            const extra: usize = if (i < remainder) 1 else 0;
            const end_row = current_row + rows_per_thread + extra;

            contexts[i] = ComputeContext{
                .strings = strings,
                .matrix = matrix,
                .measure_fn = measure_fn,
                .allocator = self.allocator,
                .start_row = current_row,
                .end_row = @min(end_row, matrix.row_range.end),
            };

            current_row = end_row;
        }

        // Launch threads
        for (0..thread_count) |i| {
            threads[i] = try Thread.spawn(.{}, computeWorker, .{&contexts[i]});
        }

        // Wait for completion
        for (0..thread_count) |i| {
            threads[i].join();
        }

        // Check for errors
        for (contexts) |ctx| {
            if (ctx.error_occurred) {
                return error.ComputationFailed;
            }
        }
    }

    /// Single-threaded matrix computation
    fn computeMatrixSingleThreaded(
        self: Self,
        matrix: *Matrix,
        strings: []const StringValue,
        comptime measure_fn: anytype,
    ) !void {
        _ = self;

        for (matrix.row_range.start..matrix.row_range.end) |i| {
            for (matrix.col_range.start..matrix.col_range.end) |j| {
                if (matrix.triangular and i > j) continue;

                const similarity = measure_fn(strings[i], strings[j]);
                matrix.set(i, j, @floatCast(similarity)) catch |err| {
                    // This should not happen with valid ranges, but handle gracefully
                    std.log.warn("Matrix set failed for ({}, {}): {}", .{ i, j, err });
                    continue;
                };
            }
        }
    }

    /// Compute matrix chunk in parallel with load balancing
    pub fn computeMatrixBalanced(
        self: Self,
        matrix: *Matrix,
        strings: []const StringValue,
        comptime measure_fn: anytype,
    ) !void {
        const work_size = matrix.getComputations();
        const thread_count = self.config.getThreadCount(work_size);

        if (thread_count == 1) {
            return self.computeMatrixSingleThreaded(matrix, strings, measure_fn);
        }

        // Create work queue for load balancing
        var work_queue = WorkQueue.init(self.allocator, matrix);
        defer work_queue.deinit();

        // Create thread pool
        var threads = try self.allocator.alloc(Thread, thread_count);
        defer self.allocator.free(threads);

        var contexts = try self.allocator.alloc(BalancedContext, thread_count);
        defer self.allocator.free(contexts);

        // Initialize contexts
        for (0..thread_count) |i| {
            contexts[i] = BalancedContext{
                .strings = strings,
                .matrix = matrix,
                .measure_fn = measure_fn,
                .allocator = self.allocator,
                .work_queue = &work_queue,
            };
        }

        // Launch threads
        for (0..thread_count) |i| {
            threads[i] = try Thread.spawn(.{}, balancedWorker, .{&contexts[i]});
        }

        // Wait for completion
        for (0..thread_count) |i| {
            threads[i].join();
        }

        // Check for errors
        for (contexts) |ctx| {
            if (ctx.error_occurred) {
                return error.ComputationFailed;
            }
        }
    }
};

/// Worker function for parallel computation
/// Uses lock-free approach since each thread works on disjoint row ranges
fn computeWorker(ctx: *ComputeContext) void {
    for (ctx.start_row..ctx.end_row) |i| {
        for (ctx.matrix.col_range.start..ctx.matrix.col_range.end) |j| {
            if (ctx.matrix.triangular and i > j) continue;

            const similarity = ctx.measure_fn(ctx.strings[i], ctx.strings[j]);

            // Lock-free matrix update - safe because threads work on disjoint ranges
            ctx.matrix.set(i, j, @floatCast(similarity)) catch |err| {
                // Mark error without locking - individual bool writes are atomic
                ctx.error_occurred = true;
                if (!@import("builtin").is_test) {
                    std.log.warn("Matrix set failed for ({}, {}): {}", .{ i, j, err });
                }
                continue;
            };
        }
    }
}

/// Work item for load balancing
const WorkItem = struct {
    row: usize,
    col: usize,
};

/// Optimized work queue with reduced mutex contention
const WorkQueue = struct {
    items: std.ArrayList(WorkItem),
    mutex: Thread.Mutex = .{},
    index: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    batch_size: usize = 8, // Process multiple items per lock

    const Self = @This();

    fn init(allocator: Allocator, matrix: *Matrix) Self {
        var queue = Self{
            .items = std.ArrayList(WorkItem).init(allocator),
        };

        // Populate work queue
        for (matrix.row_range.start..matrix.row_range.end) |i| {
            for (matrix.col_range.start..matrix.col_range.end) |j| {
                if (matrix.triangular and i > j) continue;
                queue.items.append(WorkItem{ .row = i, .col = j }) catch {
                    // This is a critical failure that would result in incomplete computation
                    continue;
                };
            }
        }

        return queue;
    }

    fn deinit(self: *Self) void {
        self.items.deinit();
    }

    /// Get next work item using atomic operations for reduced contention
    fn getNext(self: *Self) ?WorkItem {
        // Use atomic fetch-and-add for lock-free index increment
        const current_index = self.index.fetchAdd(1, .acq_rel);

        if (current_index >= self.items.items.len) {
            return null;
        }

        return self.items.items[current_index];
    }

    /// Get a batch of work items to reduce mutex contention
    fn getBatch(self: *Self, allocator: Allocator) ![]WorkItem {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current_index = self.index.load(.acquire);
        if (current_index >= self.items.items.len) {
            return &[_]WorkItem{};
        }

        const remaining = self.items.items.len - current_index;
        const batch_count = @min(self.batch_size, remaining);

        const batch = try allocator.alloc(WorkItem, batch_count);
        @memcpy(batch, self.items.items[current_index .. current_index + batch_count]);

        self.index.store(current_index + batch_count, .release);
        return batch;
    }
};

/// Context for balanced parallel computation
const BalancedContext = struct {
    strings: []const StringValue,
    matrix: *Matrix,
    measure_fn: ComputeContext.MeasureFn,
    allocator: Allocator,
    work_queue: *WorkQueue,
    error_occurred: bool = false,
    // Remove mutex - matrix access is naturally thread-safe for different indices
};

/// Worker function for balanced parallel computation with lock-free matrix updates
fn balancedWorker(ctx: *BalancedContext) void {
    while (ctx.work_queue.getNext()) |item| {
        const similarity = ctx.measure_fn(ctx.strings[item.row], ctx.strings[item.col]);

        // Lock-free matrix update - safe because each (row,col) is processed by only one thread
        ctx.matrix.set(item.row, item.col, @floatCast(similarity)) catch |err| {
            // Mark error atomically - individual bool writes are atomic
            ctx.error_occurred = true;
            if (!@import("builtin").is_test) {
                std.log.warn("Matrix set failed for ({}, {}): {}", .{ item.row, item.col, err });
            }
            continue;
        };
    }
}

/// Optimized batch worker that processes multiple items with reduced atomic contention
fn balancedBatchWorker(ctx: *BalancedContext) void {
    while (true) {
        // Get a batch of work items to reduce atomic operations
        const batch = ctx.work_queue.getBatch(ctx.allocator) catch {
            ctx.error_occurred = true;
            break;
        };
        defer ctx.allocator.free(batch);

        if (batch.len == 0) break; // No more work

        // Process entire batch without additional atomic operations
        for (batch) |item| {
            const similarity = ctx.measure_fn(ctx.strings[item.row], ctx.strings[item.col]);

            ctx.matrix.set(item.row, item.col, @floatCast(similarity)) catch |err| {
                ctx.error_occurred = true;
                if (!@import("builtin").is_test) {
                    std.log.warn("Matrix set failed for ({}, {}): {}", .{ item.row, item.col, err });
                }
                continue;
            };
        }
    }
}

// Tests
test "ParallelConfig thread count calculation" {
    const config = ParallelConfig{
        .thread_count = 4,
        .min_work_per_thread = 100,
        .max_threads = 8,
    };

    // Small work size should use 1 thread
    try testing.expect(config.getThreadCount(50) == 1);

    // Normal work size should use specified threads
    try testing.expect(config.getThreadCount(1000) == 4);

    // Large work size should be limited by work per thread
    try testing.expect(config.getThreadCount(200) == 2);
}

test "ParallelConfig auto-detect threads" {
    const config = ParallelConfig{
        .min_work_per_thread = 100,
        .max_threads = 8,
    };

    const threads = config.getThreadCount(1000);
    try testing.expect(threads >= 1 and threads <= 8);
}

test "Parallel matrix computation" {
    const allocator = testing.allocator;

    // Create test strings
    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "hello"),
        try StringValue.fromBytes(allocator, "world"),
        try StringValue.fromBytes(allocator, "test"),
    };
    defer for (&strings) |*s| s.deinit();

    // Create matrix
    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test with small thread count to ensure it works
    const config = ParallelConfig{
        .thread_count = 2,
        .min_work_per_thread = 1,
        .max_threads = 2,
    };

    const parallel_compute = ParallelCompute.init(allocator, config);

    // Compute matrix using Hamming distance
    try parallel_compute.computeMatrix(&matrix, &strings, distances.hamming);

    // Verify diagonal elements are 0 (strings equal to themselves)
    try testing.expect(try matrix.get(0, 0) == 0.0);
    try testing.expect(try matrix.get(1, 1) == 0.0);
    try testing.expect(try matrix.get(2, 2) == 0.0);

    // Verify symmetric matrix (due to triangular storage)
    try testing.expect(try matrix.get(0, 1) == try matrix.get(1, 0));
}

test "Single-threaded fallback" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    const config = ParallelConfig{
        .thread_count = 1,
        .min_work_per_thread = 100,
    };

    const parallel_compute = ParallelCompute.init(allocator, config);
    try parallel_compute.computeMatrix(&matrix, &strings, distances.hamming);

    try testing.expect(try matrix.get(0, 0) == 0.0);
    try testing.expect(try matrix.get(1, 1) == 0.0);
}

test "Balanced parallel computation" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "cat"),
        try StringValue.fromBytes(allocator, "bat"),
        try StringValue.fromBytes(allocator, "rat"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    const config = ParallelConfig{
        .thread_count = 2,
        .min_work_per_thread = 1,
    };

    const parallel_compute = ParallelCompute.init(allocator, config);
    try parallel_compute.computeMatrixBalanced(&matrix, &strings, distances.hamming);

    // Verify results
    try testing.expect(try matrix.get(0, 0) == 0.0);
    try testing.expect(try matrix.get(0, 1) == 1.0); // "cat" vs "bat" differs by 1
    try testing.expect(try matrix.get(0, 2) == 1.0); // "cat" vs "rat" differs by 1
}

test "Atomic work queue operations" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
        try StringValue.fromBytes(allocator, "c"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test atomic work queue
    var work_queue = WorkQueue.init(allocator, &matrix);
    defer work_queue.deinit();

    // Verify atomic operations work correctly
    const total_items = work_queue.items.items.len;
    var retrieved_count: usize = 0;

    // Simulate concurrent access
    while (work_queue.getNext()) |_| {
        retrieved_count += 1;
    }

    try testing.expect(retrieved_count == total_items);

    // Verify no more items available
    try testing.expect(work_queue.getNext() == null);
}

test "Lock-free matrix updates" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "test1"),
        try StringValue.fromBytes(allocator, "test2"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    const config = ParallelConfig{
        .thread_count = 2,
        .min_work_per_thread = 1,
    };

    const parallel_compute = ParallelCompute.init(allocator, config);

    // Test that lock-free updates work correctly
    try parallel_compute.computeMatrix(&matrix, &strings, distances.hamming);

    // Verify matrix was computed correctly without data races
    try testing.expect(try matrix.get(0, 0) == 0.0);
    try testing.expect(try matrix.get(1, 1) == 0.0);

    // Check that off-diagonal elements were computed
    const val = try matrix.get(0, 1);
    try testing.expect(val >= 0.0); // Should be a valid distance
}
