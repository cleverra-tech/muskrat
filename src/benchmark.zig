const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;
const Matrix = @import("matrix.zig").Matrix;
const distances = @import("distances.zig");
const ParallelCompute = @import("parallel.zig").ParallelCompute;
const memory = @import("memory.zig");

/// Benchmark result structure
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_ns: u64,
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    ops_per_sec: f64,
    memory_used: usize,

    /// Format benchmark result for display
    pub fn format(self: BenchmarkResult, writer: anytype) !void {
        try writer.print("{s}: {d} iterations, avg {d}ns ({d:.2} ops/sec), mem {d}B", .{
            self.name,
            self.iterations,
            self.avg_time_ns,
            self.ops_per_sec,
            self.memory_used,
        });
    }
};

/// Benchmark runner
pub const Benchmark = struct {
    const Self = @This();

    allocator: Allocator,
    results: std.ArrayList(BenchmarkResult),

    /// Initialize benchmark runner
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
        };
    }

    /// Cleanup benchmark runner
    pub fn deinit(self: *Self) void {
        self.results.deinit();
    }

    /// Run a benchmark function
    pub fn run(self: *Self, comptime name: []const u8, iterations: usize, benchmark_fn: anytype, context: anytype) !void {
        const start_memory = self.getCurrentMemoryUsage();
        const times = try self.allocator.alloc(u64, iterations);
        defer self.allocator.free(times);

        // Warmup
        _ = try benchmark_fn(context);

        // Run benchmark
        for (times) |*time| {
            const start = std.time.nanoTimestamp();
            _ = try benchmark_fn(context);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
        }

        const end_memory = self.getCurrentMemoryUsage();
        const memory_used = if (end_memory >= start_memory) end_memory - start_memory else 0;

        // Calculate statistics
        var total_time: u64 = 0;
        var min_time: u64 = std.math.maxInt(u64);
        var max_time: u64 = 0;

        for (times) |time| {
            total_time += time;
            min_time = @min(min_time, time);
            max_time = @max(max_time, time);
        }

        const avg_time = total_time / iterations;
        const ops_per_sec = if (avg_time > 0) 1_000_000_000.0 / @as(f64, @floatFromInt(avg_time)) else 0.0;

        const result = BenchmarkResult{
            .name = name,
            .iterations = iterations,
            .total_time_ns = total_time,
            .avg_time_ns = avg_time,
            .min_time_ns = min_time,
            .max_time_ns = max_time,
            .ops_per_sec = ops_per_sec,
            .memory_used = memory_used,
        };

        try self.results.append(result);
    }

    /// Get current memory usage (approximation)
    fn getCurrentMemoryUsage(self: Self) usize {
        _ = self;
        // This is a simplified memory tracking
        // In a real implementation, you might use a tracking allocator
        return 0;
    }

    /// Print all benchmark results
    pub fn printResults(self: *const Self) void {
        std.debug.print("\nBenchmark Results:\n");
        std.debug.print("==================\n");
        for (self.results.items) |result| {
            std.debug.print("{f}\n", .{result});
        }
    }

    /// Get results for programmatic use
    pub fn getResults(self: *const Self) []const BenchmarkResult {
        return self.results.items;
    }
};

/// String distance benchmark context
const DistanceBenchmarkContext = struct {
    str1: StringValue,
    str2: StringValue,
    allocator: Allocator,
};

/// Matrix computation benchmark context
const MatrixBenchmarkContext = struct {
    strings: []StringValue,
    matrix: Matrix,
    allocator: Allocator,
};

/// Memory pool benchmark context
const MemoryPoolBenchmarkContext = struct {
    pool: memory.ObjectPool(TestObject),
    allocator: Allocator,

    const TestObject = struct {
        data: [64]u8,
        value: i32,
    };
};

// Benchmark functions
fn benchmarkLevenshtein(context: DistanceBenchmarkContext) !f32 {
    return distances.levenshtein(context.allocator, context.str1, context.str2);
}

fn benchmarkHamming(context: DistanceBenchmarkContext) !f64 {
    return distances.hamming(context.str1, context.str2);
}

fn benchmarkJaroWinkler(context: DistanceBenchmarkContext) !f32 {
    return distances.jaroWinkler(context.allocator, context.str1, context.str2);
}

fn benchmarkMatrixCreation(context: MatrixBenchmarkContext) !Matrix {
    var matrix = try Matrix.init(context.allocator, context.strings);
    defer matrix.deinit();
    return matrix;
}

fn benchmarkParallelComputation(context: MatrixBenchmarkContext) !void {
    const config = ParallelCompute.ParallelConfig{
        .thread_count = 4,
        .min_work_per_thread = 25,
    };

    var parallel = ParallelCompute.init(context.allocator, config);
    defer parallel.deinit();

    try parallel.computeMatrix(&context.matrix, context.strings, distances.levenshtein);
}

fn benchmarkMemoryPool(context: MemoryPoolBenchmarkContext) !*MemoryPoolBenchmarkContext.TestObject {
    const obj = try context.pool.acquire();
    obj.value = 42;
    context.pool.release(obj);
    return obj;
}

/// Run comprehensive benchmarks
pub fn runComprehensiveBenchmarks(allocator: Allocator) !void {
    var benchmark = Benchmark.init(allocator);
    defer benchmark.deinit();

    // Test strings for benchmarking
    var test_str1 = try StringValue.fromBytes(allocator, "kitten");
    defer test_str1.deinit();
    var test_str2 = try StringValue.fromBytes(allocator, "sitting");
    defer test_str2.deinit();

    var long_str1 = try StringValue.fromBytes(allocator, "The quick brown fox jumps over the lazy dog");
    defer long_str1.deinit();
    var long_str2 = try StringValue.fromBytes(allocator, "The quick brown fox jumped over the lazy dog");
    defer long_str2.deinit();

    // Distance benchmarks
    {
        const short_context = DistanceBenchmarkContext{
            .str1 = test_str1,
            .str2 = test_str2,
            .allocator = allocator,
        };

        const long_context = DistanceBenchmarkContext{
            .str1 = long_str1,
            .str2 = long_str2,
            .allocator = allocator,
        };

        try benchmark.run("Levenshtein (short)", 10000, benchmarkLevenshtein, short_context);
        try benchmark.run("Levenshtein (long)", 1000, benchmarkLevenshtein, long_context);
        try benchmark.run("Hamming (short)", 50000, benchmarkHamming, short_context);
        try benchmark.run("Hamming (long)", 10000, benchmarkHamming, long_context);
        try benchmark.run("Jaro-Winkler (short)", 10000, benchmarkJaroWinkler, short_context);
        try benchmark.run("Jaro-Winkler (long)", 1000, benchmarkJaroWinkler, long_context);
    }

    // Matrix benchmarks
    {
        var test_strings = [_]StringValue{
            try StringValue.fromBytes(allocator, "apple"),
            try StringValue.fromBytes(allocator, "banana"),
            try StringValue.fromBytes(allocator, "cherry"),
            try StringValue.fromBytes(allocator, "date"),
            try StringValue.fromBytes(allocator, "elderberry"),
        };
        defer for (&test_strings) |*s| s.deinit();

        var matrix = try Matrix.init(allocator, &test_strings);
        defer matrix.deinit();

        const matrix_context = MatrixBenchmarkContext{
            .strings = &test_strings,
            .matrix = matrix,
            .allocator = allocator,
        };

        try benchmark.run("Matrix creation", 1000, benchmarkMatrixCreation, matrix_context);
        try benchmark.run("Parallel computation", 100, benchmarkParallelComputation, matrix_context);
    }

    // Memory pool benchmarks
    {
        var pool = memory.ObjectPool(MemoryPoolBenchmarkContext.TestObject).init(allocator);
        defer pool.deinit();

        const pool_context = MemoryPoolBenchmarkContext{
            .pool = pool,
            .allocator = allocator,
        };

        try benchmark.run("Memory pool acquire/release", 100000, benchmarkMemoryPool, pool_context);
    }

    benchmark.printResults();
}

/// Integration test runner
pub const IntegrationTest = struct {
    const Self = @This();

    allocator: Allocator,
    passed: usize,
    failed: usize,
    test_results: std.ArrayList(TestResult),

    const TestResult = struct {
        name: []const u8,
        passed: bool,
        error_msg: ?[]const u8,
        duration_ns: u64,
    };

    /// Initialize integration test runner
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .passed = 0,
            .failed = 0,
            .test_results = std.ArrayList(TestResult).init(allocator),
        };
    }

    /// Cleanup test runner
    pub fn deinit(self: *Self) void {
        for (self.test_results.items) |result| {
            if (result.error_msg) |msg| {
                self.allocator.free(msg);
            }
        }
        self.test_results.deinit();
    }

    /// Run a test function
    pub fn runTest(self: *Self, comptime name: []const u8, test_fn: anytype, context: anytype) void {
        const start = std.time.nanoTimestamp();

        if (test_fn(context)) |_| {
            const end = std.time.nanoTimestamp();
            self.passed += 1;
            self.test_results.append(TestResult{
                .name = name,
                .passed = true,
                .error_msg = null,
                .duration_ns = @intCast(end - start),
            }) catch {};
        } else |err| {
            const end = std.time.nanoTimestamp();
            self.failed += 1;
            const error_msg = std.fmt.allocPrint(self.allocator, "{}", .{err}) catch "Unknown error";
            self.test_results.append(TestResult{
                .name = name,
                .passed = false,
                .error_msg = error_msg,
                .duration_ns = @intCast(end - start),
            }) catch {};
        }
    }

    /// Print test summary
    pub fn printSummary(self: *const Self) void {
        std.debug.print("\nIntegration Test Results:\n");
        std.debug.print("========================\n");
        std.debug.print("Passed: {d}\n", .{self.passed});
        std.debug.print("Failed: {d}\n", .{self.failed});
        std.debug.print("Total: {d}\n", .{self.passed + self.failed});

        if (self.failed > 0) {
            std.debug.print("\nFailed tests:\n");
            for (self.test_results.items) |result| {
                if (!result.passed) {
                    std.debug.print("  {s}: {s}\n", .{ result.name, result.error_msg orelse "Unknown error" });
                }
            }
        }
    }
};

// Integration test functions
fn testStringProcessingPipeline(allocator: Allocator) !void {
    var str1 = try StringValue.fromBytes(allocator, "hello world");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "hello earth");
    defer str2.deinit();

    // Test distance calculations
    const lev = try distances.levenshtein(allocator, str1, str2);
    const ham = distances.hamming(str1, str2);
    const jw = try distances.jaroWinkler(allocator, str1, str2);

    try testing.expect(lev > 0.0);
    try testing.expect(ham > 0.0);
    try testing.expect(jw > 0.0 and jw < 1.0);
}

fn testMatrixOperations(allocator: Allocator) !void {
    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "test1"),
        try StringValue.fromBytes(allocator, "test2"),
        try StringValue.fromBytes(allocator, "test3"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test matrix operations
    matrix.set(0, 1, 0.5);
    matrix.set(1, 2, 0.8);

    try testing.expect(matrix.get(0, 1) == 0.5);
    try testing.expect(matrix.get(1, 2) == 0.8);
    try testing.expect(matrix.get(0, 0) == 0.0); // Diagonal should be 0
}

fn testParallelProcessing(allocator: Allocator) !void {
    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "parallel1"),
        try StringValue.fromBytes(allocator, "parallel2"),
        try StringValue.fromBytes(allocator, "parallel3"),
        try StringValue.fromBytes(allocator, "parallel4"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    const config = ParallelCompute.ParallelConfig{
        .thread_count = 2,
        .min_work_per_thread = 50,
    };

    var parallel = ParallelCompute.init(allocator, config);
    defer parallel.deinit();

    try parallel.computeMatrix(&strings, distances.hamming, &matrix, config);

    // Verify some computations were done
    var computed_count: usize = 0;
    for (matrix.row_range.start..matrix.row_range.end) |i| {
        for (matrix.col_range.start..matrix.col_range.end) |j| {
            if (i != j and matrix.get(i, j) >= 0.0) {
                computed_count += 1;
            }
        }
    }
    try testing.expect(computed_count > 0);
}

fn testMemoryManagement(allocator: Allocator) !void {
    // Test object pool
    var pool = memory.ObjectPool(i32).init(allocator);
    defer pool.deinit();

    const obj1 = try pool.acquire();
    obj1.* = 42;
    pool.release(obj1);

    const obj2 = try pool.acquire(); // Should reuse obj1
    try testing.expect(obj2.* == 42);
    pool.release(obj2);

    // Test string cache
    var cache = memory.StringCache.init(allocator);
    defer cache.deinit();

    const str1 = try cache.intern("cached_string");
    const str2 = try cache.intern("cached_string");
    try testing.expect(str1.ptr == str2.ptr); // Should be same pointer

    cache.release(str1);
    cache.release(str2);
}

/// Run all integration tests
pub fn runIntegrationTests(allocator: Allocator) !void {
    var test_runner = IntegrationTest.init(allocator);
    defer test_runner.deinit();

    test_runner.runTest("String processing pipeline", testStringProcessingPipeline, allocator);
    test_runner.runTest("Matrix operations", testMatrixOperations, allocator);
    test_runner.runTest("Parallel processing", testParallelProcessing, allocator);
    test_runner.runTest("Memory management", testMemoryManagement, allocator);

    test_runner.printSummary();

    if (test_runner.failed > 0) {
        return error.IntegrationTestsFailed;
    }
}

// Tests
test "Benchmark basic functionality" {
    const allocator = testing.allocator;

    var benchmark = Benchmark.init(allocator);
    defer benchmark.deinit();

    var str1 = try StringValue.fromBytes(allocator, "test");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "best");
    defer str2.deinit();

    const context = DistanceBenchmarkContext{
        .str1 = str1,
        .str2 = str2,
        .allocator = allocator,
    };

    try benchmark.run("Test benchmark", 10, benchmarkHamming, context);

    const results = benchmark.getResults();
    try testing.expect(results.len == 1);
    try testing.expect(results[0].iterations == 10);
    try testing.expect(results[0].avg_time_ns > 0);
}

test "Integration test runner" {
    const allocator = testing.allocator;

    var test_runner = IntegrationTest.init(allocator);
    defer test_runner.deinit();

    test_runner.runTest("Passing test", testStringProcessingPipeline, allocator);

    try testing.expect(test_runner.passed == 1);
    try testing.expect(test_runner.failed == 0);
}

test "BenchmarkResult formatting" {
    const result = BenchmarkResult{
        .name = "Test",
        .iterations = 100,
        .total_time_ns = 1000000,
        .avg_time_ns = 10000,
        .min_time_ns = 8000,
        .max_time_ns = 15000,
        .ops_per_sec = 100000.0,
        .memory_used = 1024,
    };

    var buffer: [256]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&buffer, "{f}", .{result});
    try testing.expect(std.mem.indexOf(u8, formatted, "Test") != null);
    try testing.expect(std.mem.indexOf(u8, formatted, "100") != null);
}
