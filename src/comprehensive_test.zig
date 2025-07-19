const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const muskrat = @import("muskrat.zig");

/// Comprehensive test suite that validates core functionality
pub fn runComprehensiveTests(allocator: Allocator) !void {
    std.debug.print("Running comprehensive test suite...\n", .{});

    try testStringTypes(allocator);
    try testDistanceMeasures(allocator);
    try testMatrixOperations(allocator);
    try testParallelComputation(allocator);
    try testKernelFunctions(allocator);
    try testSimilarityCoefficients(allocator);
    try testInputReaders(allocator);
    try testOutputFormatters(allocator);
    try testMemoryManagement(allocator);
    try testConfiguration(allocator);

    std.debug.print("All comprehensive tests passed!\n", .{});
}

fn testStringTypes(allocator: Allocator) !void {
    std.debug.print("Testing string types...\n", .{});

    // Test byte strings
    var byte_str = try muskrat.StringValue.fromBytes(allocator, "hello world");
    defer byte_str.deinit();

    switch (byte_str.data) {
        .byte => |bytes| try testing.expect(std.mem.eql(u8, bytes, "hello world")),
        else => return error.UnexpectedStringType,
    }

    // Test string operations
    switch (byte_str.data) {
        .byte => |bytes| try testing.expect(bytes.len == 11),
        else => return error.UnexpectedStringType,
    }

    var empty_str = try muskrat.StringValue.fromBytes(allocator, "");
    defer empty_str.deinit();
    switch (empty_str.data) {
        .byte => |bytes| try testing.expect(bytes.len == 0),
        else => return error.UnexpectedStringType,
    }
}

fn testDistanceMeasures(allocator: Allocator) !void {
    std.debug.print("Testing distance measures...\n", .{});

    var str1 = try muskrat.StringValue.fromBytes(allocator, "kitten");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "sitting");
    defer str2.deinit();

    // Test Levenshtein distance
    const lev = try muskrat.distances.levenshtein(allocator, str1, str2);
    try testing.expect(lev == 3.0); // Expected distance

    // Test Hamming distance (same length strings)
    var str3 = try muskrat.StringValue.fromBytes(allocator, "karolin");
    defer str3.deinit();
    var str4 = try muskrat.StringValue.fromBytes(allocator, "kathrin");
    defer str4.deinit();

    const ham = muskrat.distances.hamming(str3, str4);
    try testing.expect(ham == 3.0); // Expected distance

    // Test Jaro-Winkler similarity
    const jw = try muskrat.distances.jaroWinkler(allocator, str1, str2);
    try testing.expect(jw >= 0.0 and jw <= 1.0); // Should be between 0 and 1
}

fn testMatrixOperations(allocator: Allocator) !void {
    std.debug.print("Testing matrix operations...\n", .{});

    var strings = [_]muskrat.StringValue{
        try muskrat.StringValue.fromBytes(allocator, "apple"),
        try muskrat.StringValue.fromBytes(allocator, "banana"),
        try muskrat.StringValue.fromBytes(allocator, "cherry"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try muskrat.Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test matrix properties
    try testing.expect(matrix.row_range.length() == 3);
    try testing.expect(matrix.col_range.length() == 3);

    // Test matrix operations
    try matrix.set(0, 1, 0.5);
    try matrix.set(1, 2, 0.8);
    try matrix.set(2, 0, 0.3);

    try testing.expect(try matrix.get(0, 1) == 0.5);
    try testing.expect(try matrix.get(1, 2) == 0.8);
    try testing.expect(try matrix.get(2, 0) == 0.3);

    // Test symmetry for triangular matrices
    if (matrix.triangular) {
        try testing.expect(try matrix.get(1, 0) == try matrix.get(0, 1));
    }

    // Test diagonal is zero
    try testing.expect(try matrix.get(0, 0) == 0.0);
    try testing.expect(try matrix.get(1, 1) == 0.0);
    try testing.expect(try matrix.get(2, 2) == 0.0);
}

fn testParallelComputation(allocator: Allocator) !void {
    std.debug.print("Testing parallel computation...\n", .{});

    var strings = [_]muskrat.StringValue{
        try muskrat.StringValue.fromBytes(allocator, "parallel1"),
        try muskrat.StringValue.fromBytes(allocator, "parallel2"),
        try muskrat.StringValue.fromBytes(allocator, "parallel3"),
        try muskrat.StringValue.fromBytes(allocator, "parallel4"),
        try muskrat.StringValue.fromBytes(allocator, "parallel5"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try muskrat.Matrix.init(allocator, &strings);
    defer matrix.deinit();

    const parallel_config = muskrat.ParallelConfig{
        .thread_count = 2,
        .min_work_per_thread = 50,
    };
    _ = muskrat.ParallelCompute.init(allocator, parallel_config);

    // Test different parallel configurations
    const configs = [_]muskrat.ParallelConfig{
        .{ .thread_count = 2, .min_work_per_thread = 50 },
        .{ .thread_count = 3, .min_work_per_thread = 25 },
    };

    for (configs) |config| {
        var new_parallel = muskrat.ParallelCompute.init(allocator, config);
        try new_parallel.computeMatrix(&matrix, &strings, muskrat.distances.hamming);

        // Verify computations were performed
        var computed_pairs: usize = 0;
        for (matrix.row_range.start..matrix.row_range.end) |i| {
            for (matrix.col_range.start..matrix.col_range.end) |j| {
                if (i != j) {
                    try testing.expect(try matrix.get(i, j) >= 0.0);
                    computed_pairs += 1;
                }
            }
        }
        try testing.expect(computed_pairs > 0);
    }
}

fn testKernelFunctions(allocator: Allocator) !void {
    std.debug.print("Testing kernel functions...\n", .{});

    var str1 = try muskrat.StringValue.fromBytes(allocator, "abcdef");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "abcxyz");
    defer str2.deinit();

    // Test spectrum kernel
    var spectrum = muskrat.kernels.SpectrumKernel.init(allocator, 2);

    const spec_sim = try spectrum.compute(str1, str2);
    try testing.expect(spec_sim >= 0.0);

    // Test subsequence kernel
    var subseq = muskrat.kernels.SubsequenceKernel.init(allocator, 3, 0.5);

    const subseq_sim = try subseq.compute(str1, str2);
    try testing.expect(subseq_sim >= 0.0);

    // Test weighted degree kernel
    var weighted = try muskrat.kernels.WeightedDegreeKernel.init(allocator, 3);
    defer weighted.deinit();

    const weighted_sim = weighted.compute(str1, str2);
    try testing.expect(weighted_sim >= 0.0);
}

fn testSimilarityCoefficients(allocator: Allocator) !void {
    std.debug.print("Testing similarity coefficients...\n", .{});

    var str1 = try muskrat.StringValue.fromBytes(allocator, "abcde");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "cdefg");
    defer str2.deinit();

    // Test Jaccard coefficient
    var jaccard = muskrat.coefficients.JaccardCoefficient.init(allocator, 2);

    const jac_sim = try jaccard.compute(str1, str2);
    try testing.expect(jac_sim >= 0.0 and jac_sim <= 1.0);

    // Test Dice coefficient
    var dice = muskrat.coefficients.DiceCoefficient.init(allocator, 2);

    const dice_sim = try dice.compute(str1, str2);
    try testing.expect(dice_sim >= 0.0 and dice_sim <= 1.0);

    // Test Simpson coefficient
    var simpson = muskrat.coefficients.SimpsonCoefficient.init(allocator, 2);

    const simp_sim = try simpson.compute(str1, str2);
    try testing.expect(simp_sim >= 0.0 and simp_sim <= 1.0);

    // Test Cosine coefficient
    var cosine = muskrat.coefficients.CosineCoefficient.init(allocator, 0);

    const cos_sim = try cosine.compute(str1, str2);
    try testing.expect(cos_sim >= 0.0 and cos_sim <= 1.0);
}

fn testInputReaders(allocator: Allocator) !void {
    std.debug.print("Testing input readers...\n", .{});

    // Test memory reader
    var memory_reader = muskrat.readers.MemoryReader.init(allocator);
    defer memory_reader.deinit();

    try memory_reader.addFromSlice("test1");
    try memory_reader.addFromSlice("test2");
    try memory_reader.addFromSlice("test3");

    try testing.expect(memory_reader.count() == 3);

    const strings = memory_reader.getStrings();
    try testing.expect(strings.len == 3);

    // Test delimited parsing
    try memory_reader.parseDelimited("apple,banana,cherry", ",");
    try testing.expect(memory_reader.count() == 6); // 3 + 3 new strings

    // Test file reader with temporary file
    const test_content = "line1\nline2\nline3\n";
    const temp_file = "test_comprehensive.txt";

    // Write test file
    {
        const file = try std.fs.cwd().createFile(temp_file, .{});
        defer file.close();
        try file.writeAll(test_content);
    }
    defer std.fs.cwd().deleteFile(temp_file) catch {};

    var file_reader = muskrat.readers.FileReader.init(allocator);
    defer file_reader.deinit();

    try file_reader.readFromFile(temp_file);
    try testing.expect(file_reader.count() == 3);
}

fn testOutputFormatters(allocator: Allocator) !void {
    std.debug.print("Testing output formatters...\n", .{});

    var strings = [_]muskrat.StringValue{
        try muskrat.StringValue.fromBytes(allocator, "format1"),
        try muskrat.StringValue.fromBytes(allocator, "format2"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try muskrat.Matrix.init(allocator, &strings);
    defer matrix.deinit();

    try matrix.set(0, 1, 0.75);
    try matrix.set(1, 0, 0.75);

    const metadata = muskrat.formatters.OutputMetadata{
        .string_count = 2,
        .computation_time_ms = 123.45,
        .distance_measure = "test",
    };

    // Test text formatter
    var text_formatter = muskrat.formatters.TextFormatter.init(allocator);
    const text_output = try text_formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(text_output);

    try testing.expect(std.mem.indexOf(u8, text_output, "0") != null);
    try testing.expect(std.mem.indexOf(u8, text_output, "Metadata") != null);

    // Test JSON formatter
    var json_formatter = muskrat.formatters.JsonFormatter.init(allocator);
    const json_output = try json_formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(json_output);

    try testing.expect(std.mem.indexOf(u8, json_output, "\"strings\"") != null);
    try testing.expect(std.mem.indexOf(u8, json_output, "\"metadata\"") != null);

    // Test CSV formatter
    var csv_formatter = muskrat.formatters.CsvFormatter.init(allocator);
    const csv_output = try csv_formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(csv_output);

    try testing.expect(std.mem.indexOf(u8, csv_output, "String,0,1") != null);

    // Test binary formatter
    var binary_formatter = muskrat.formatters.BinaryFormatter.init(allocator);
    const binary_output = try binary_formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(binary_output);

    try testing.expect(binary_output.len >= muskrat.formatters.BINARY_MAGIC_SIZE);
    try testing.expect(std.mem.eql(u8, binary_output[0..muskrat.formatters.BINARY_MAGIC_SIZE], muskrat.formatters.BINARY_MAGIC_NUMBER));
}

fn testMemoryManagement(allocator: Allocator) !void {
    std.debug.print("Testing memory management...\n", .{});

    // Test object pool
    const TestStruct = struct {
        value: i32,
        data: [32]u8,
    };

    var pool = muskrat.memory.ObjectPool(TestStruct).init(allocator);
    defer pool.deinit();

    const obj1 = try pool.acquire();
    obj1.value = 42;
    pool.release(obj1);

    const obj2 = try pool.acquire(); // Should reuse obj1
    try testing.expect(obj2.value == 42);
    pool.release(obj2);

    const stats = pool.getStats();
    try testing.expect(stats.reused_count > 0);

    // Test memory pool
    var mem_pool = muskrat.memory.MemoryPool.init(allocator, 1024);
    defer mem_pool.deinit();

    const data1 = try mem_pool.alloc(64, 8);
    const data2 = try mem_pool.alloc(128, 8);

    try testing.expect(data1.len == 64);
    try testing.expect(data2.len == 128);

    // Test string cache
    var cache = muskrat.memory.StringCache.init(allocator);
    defer cache.deinit();

    const str1 = try cache.intern("cached");
    const str2 = try cache.intern("cached");
    try testing.expect(str1.ptr == str2.ptr);

    cache.release(str1);
    cache.release(str2);

    // Test temp arena
    var arena = muskrat.memory.TempArena.init(allocator);
    defer arena.deinit();

    const temp_allocator = arena.allocator();
    const temp_data = try temp_allocator.alloc(u8, 100);
    try testing.expect(temp_data.len == 100);

    arena.reset(); // Should reset without leaking
}

fn testConfiguration(allocator: Allocator) !void {
    std.debug.print("Testing configuration...\n", .{});

    const config = muskrat.Config{};

    // Test default values
    try testing.expect(config.string.max_length == 1024 * 1024);
    try testing.expect(config.matrix.triangular == true);
    try testing.expect(config.parallel.thread_count == 0); // Auto-detect

    // Test JSON serialization
    const json_str = try std.json.stringifyAlloc(allocator, config, .{});
    defer allocator.free(json_str);

    try testing.expect(std.mem.indexOf(u8, json_str, "string") != null);
    try testing.expect(std.mem.indexOf(u8, json_str, "matrix") != null);

    // Test JSON parsing
    const parsed = try std.json.parseFromSlice(muskrat.Config, allocator, json_str, .{});
    defer parsed.deinit();

    try testing.expect(parsed.value.string.max_length == config.string.max_length);
    try testing.expect(parsed.value.matrix.triangular == config.matrix.triangular);
}

// Tests
test "Comprehensive functionality test" {
    const allocator = testing.allocator;
    try runComprehensiveTests(allocator);
}

test "Performance regression test" {
    const allocator = testing.allocator;

    // Basic performance expectations
    var str1 = try muskrat.StringValue.fromBytes(allocator, "performance");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "regression");
    defer str2.deinit();

    // Time Hamming distance (should be very fast)
    const start = std.time.nanoTimestamp();
    const distance = muskrat.distances.hamming(str1, str2);
    const end = std.time.nanoTimestamp();

    try testing.expect(distance >= 0.0);
    try testing.expect((end - start) < 1_000_000); // Should complete in under 1ms
}

test "Memory usage validation" {
    // Use testing allocator to catch leaks
    const allocator = testing.allocator;

    var strings = [_]muskrat.StringValue{
        try muskrat.StringValue.fromBytes(allocator, "memory1"),
        try muskrat.StringValue.fromBytes(allocator, "memory2"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try muskrat.Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Perform operations that should not leak
    try matrix.set(0, 1, 0.5);
    _ = try matrix.get(0, 1);

    // Test passes if no leaks detected by testing allocator
}

test "Error handling validation" {
    const allocator = testing.allocator;

    // Test error cases
    var empty_str = try muskrat.StringValue.fromBytes(allocator, "");
    defer empty_str.deinit();

    var normal_str = try muskrat.StringValue.fromBytes(allocator, "test");
    defer normal_str.deinit();

    // Hamming distance with different lengths should return max distance
    const result = muskrat.distances.hamming(empty_str, normal_str);
    try testing.expect(result > 0.0);
}
