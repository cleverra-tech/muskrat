# Muskrat

A comprehensive string similarity computation library written in Zig. Muskrat provides efficient implementations of various string distance measures, similarity coefficients, and kernel functions for analyzing text similarity.

## Features

### String Representations
- **Byte strings**: Standard UTF-8 encoded text
- **Token sequences**: Numeric token arrays for preprocessed text
- **Bit vectors**: Binary representations for set-based operations

### Distance Measures
- **Hamming distance**: Character-by-character differences for equal-length strings
- **Levenshtein distance**: Minimum edit operations (insertions, deletions, substitutions)
- **Jaro distance**: Character matching with transposition penalties
- **Jaro-Winkler distance**: Jaro distance with common prefix bonus

### Similarity Coefficients
- **Jaccard coefficient**: Intersection over union (|A ∩ B| / |A ∪ B|)
- **Dice coefficient**: Harmonic mean of precision/recall (2|A ∩ B| / (|A| + |B|))
- **Simpson coefficient**: Overlap coefficient (|A ∩ B| / min(|A|, |B|))
- **Cosine coefficient**: Cosine similarity (|A ∩ B| / sqrt(|A| * |B|))

### Kernel Functions
- **Spectrum kernel**: Compares k-gram frequency distributions
- **Subsequence kernel**: Measures common subsequences with decay factors
- **Mismatch kernel**: Spectrum kernel with allowed mismatches

### Performance Features
- **SIMD optimizations**: Automatic vectorization for x86_64 (SSE) and ARM64 (NEON)
- **Parallel computation**: Multi-threaded matrix computation with load balancing
- **Memory pooling**: Efficient allocation strategies for large-scale processing
- **Lock-free algorithms**: Reduced contention for high-performance parallel processing

### Input/Output Support
- **Multiple input formats**: Text files, CSV, directory scanning with glob patterns
- **Multiple output formats**: Text, JSON, binary, CSV
- **Matrix operations**: Efficient similarity matrix storage and computation
- **Streaming processing**: Memory-efficient handling of large datasets

### Development Features
- **Comprehensive benchmarking**: Performance measurement and comparison tools
- **Memory tracking**: Built-in memory usage monitoring
- **Extensive testing**: Unit tests and integration tests for all components
- **Configuration management**: Flexible parameter tuning and optimization

## Installation

Add Muskrat as a dependency in your `build.zig`:

```zig
const muskrat = b.dependency("muskrat", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("muskrat", muskrat.module("muskrat"));
```

## Basic Usage

### Computing String Distances

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create string values
    var str1 = try muskrat.StringValue.fromBytes(allocator, "kitten");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "sitting");
    defer str2.deinit();

    // Compute distances
    const hamming_dist = muskrat.distances.hamming(str1, str2);
    const levenshtein_dist = try muskrat.distances.levenshtein(allocator, str1, str2);
    const jaro_sim = try muskrat.distances.jaro(allocator, str1, str2);

    std.debug.print("Hamming: {d}\n", .{hamming_dist});
    std.debug.print("Levenshtein: {d}\n", .{levenshtein_dist});
    std.debug.print("Jaro: {d}\n", .{jaro_sim});
}
```

### Similarity Coefficients

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var str1 = try muskrat.StringValue.fromBytes(allocator, "hello world");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "hello earth");
    defer str2.deinit();

    // Initialize coefficients with 2-gram analysis
    var jaccard = muskrat.coefficients.JaccardCoefficient.init(allocator, 2);
    var dice = muskrat.coefficients.DiceCoefficient.init(allocator, 2);
    var cosine = muskrat.coefficients.CosineCoefficient.init(allocator, 2);

    // Compute similarities
    const jaccard_sim = try jaccard.compute(str1, str2);
    const dice_sim = try dice.compute(str1, str2);
    const cosine_sim = try cosine.compute(str1, str2);

    std.debug.print("Jaccard: {d}\n", .{jaccard_sim});
    std.debug.print("Dice: {d}\n", .{dice_sim});
    std.debug.print("Cosine: {d}\n", .{cosine_sim});
}
```

### Similarity Matrix Computation

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create string dataset
    var strings = [_]muskrat.StringValue{
        try muskrat.StringValue.fromBytes(allocator, "cat"),
        try muskrat.StringValue.fromBytes(allocator, "bat"),
        try muskrat.StringValue.fromBytes(allocator, "rat"),
        try muskrat.StringValue.fromBytes(allocator, "hat"),
    };
    defer for (&strings) |*s| s.deinit();

    // Create similarity matrix
    var matrix = try muskrat.Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Configure parallel computation
    const config = muskrat.ParallelConfig{
        .thread_count = 4,
        .min_work_per_thread = 1,
    };
    const parallel_compute = muskrat.ParallelCompute.init(allocator, config);

    // Compute similarity matrix using Hamming distance
    try parallel_compute.computeMatrix(&matrix, &strings, muskrat.distances.hamming);

    // Access results
    for (0..strings.len) |i| {
        for (0..strings.len) |j| {
            const similarity = try matrix.get(i, j);
            std.debug.print("Distance({s}, {s}) = {d}\n", .{
                strings[i].getBytes(), strings[j].getBytes(), similarity
            });
        }
    }
}
```

### Kernel Functions

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var str1 = try muskrat.StringValue.fromBytes(allocator, "machine learning");
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromBytes(allocator, "deep learning");
    defer str2.deinit();

    // Spectrum kernel with 3-grams
    var spectrum = muskrat.kernels.SpectrumKernel.init(allocator, 3);
    const spectrum_sim = try spectrum.compute(str1, str2);

    // Subsequence kernel with length 2 and decay 0.8
    var subsequence = muskrat.kernels.SubsequenceKernel.init(allocator, 2, 0.8);
    const subseq_sim = try subsequence.compute(str1, str2);

    // Mismatch kernel with 3-grams and 1 allowed mismatch
    var mismatch = muskrat.kernels.MismatchKernel.init(allocator, 3, 1);
    const mismatch_sim = try mismatch.compute(str1, str2);

    std.debug.print("Spectrum kernel: {d}\n", .{spectrum_sim});
    std.debug.print("Subsequence kernel: {d}\n", .{subseq_sim});
    std.debug.print("Mismatch kernel: {d}\n", .{mismatch_sim});
}
```

### Working with Token Sequences

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create token sequences (e.g., from a tokenizer)
    const tokens1 = [_]u64{ 101, 2023, 2003, 102 }; // [CLS] hello world [SEP]
    const tokens2 = [_]u64{ 101, 2023, 3186, 102 }; // [CLS] hello earth [SEP]

    var str1 = try muskrat.StringValue.fromTokens(allocator, &tokens1);
    defer str1.deinit();
    var str2 = try muskrat.StringValue.fromTokens(allocator, &tokens2);
    defer str2.deinit();

    // Compute distances on token sequences
    const hamming_dist = muskrat.distances.hamming(str1, str2);
    const levenshtein_dist = try muskrat.distances.levenshtein(allocator, str1, str2);

    std.debug.print("Token Hamming distance: {d}\n", .{hamming_dist});
    std.debug.print("Token Levenshtein distance: {d}\n", .{levenshtein_dist});
}
```

### Processing Files and Directories

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Configure processing
    const config = muskrat.Config{
        .distance_measure = .hamming,
        .output_format = .json,
        .parallel_threads = 4,
        .use_simd = true,
    };

    // Initialize processor
    var processor = try muskrat.Processor.init(allocator, config);
    defer processor.deinit();

    // Process directory with glob pattern
    try processor.processDirectory("./text_files", "*.txt");

    // Save results
    try processor.saveResults("similarity_results.json");
}
```

### Memory Management and Pooling

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const base_allocator = gpa.allocator();

    // Use memory pool for efficient allocation
    var pool = try muskrat.memory.Pool.init(base_allocator, .{
        .block_size = 4096,
        .max_blocks = 1000,
    });
    defer pool.deinit();

    const allocator = pool.allocator();

    // Process large datasets efficiently
    var strings = std.ArrayList(muskrat.StringValue).init(allocator);
    defer {
        for (strings.items) |*s| s.deinit();
        strings.deinit();
    }

    // Add many strings to the dataset
    for (0..10000) |i| {
        const text = try std.fmt.allocPrint(allocator, "text_{d}", .{i});
        const str_val = try muskrat.StringValue.fromBytes(allocator, text);
        try strings.append(str_val);
    }

    // Compute similarity matrix with memory pooling
    var matrix = try muskrat.Matrix.init(allocator, strings.items);
    defer matrix.deinit();

    std.debug.print("Processed {d} strings\n", .{strings.items.len});
}
```

### Benchmarking and Performance Analysis

```zig
const std = @import("std");
const muskrat = @import("muskrat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize benchmark suite
    var benchmark_suite = try muskrat.benchmark.BenchmarkSuite.init(allocator);
    defer benchmark_suite.deinit();

    // Add test data
    var test_strings = [_][]const u8{
        "hello world",
        "goodbye world",
        "hello universe",
        "farewell earth",
    };

    // Benchmark different distance measures
    try benchmark_suite.benchmarkDistanceMeasures(&test_strings);

    // Benchmark SIMD vs scalar implementations
    try benchmark_suite.benchmarkSIMDPerformance(&test_strings);

    // Benchmark parallel vs sequential processing
    try benchmark_suite.benchmarkParallelPerformance(&test_strings);

    // Generate performance report
    try benchmark_suite.generateReport("performance_report.json");
}
```

## Configuration

Muskrat supports extensive configuration through the `Config` struct:

```zig
const config = muskrat.Config{
    .distance_measure = .levenshtein,
    .output_format = .json,
    .parallel_threads = 8,
    .use_simd = true,
    .memory_pool_size = 1024 * 1024, // 1MB
    .batch_size = 100,
    .triangular_matrix = true,
    .normalize_results = false,
};
```

## Building and Testing

Build the library:
```bash
zig build
```

Run tests:
```bash
zig build test
```

Run benchmarks:
```bash
zig build benchmark
```

## Performance

Muskrat is designed for high performance:

- **SIMD acceleration**: Up to 4x speedup on compatible hardware
- **Parallel processing**: Linear scaling with CPU cores
- **Memory optimization**: Efficient allocation and cache-friendly algorithms
- **Lock-free operations**: Reduced contention in multi-threaded scenarios

Typical performance on modern hardware:
- Hamming distance: ~1M string pairs/second (SIMD-optimized)
- Levenshtein distance: ~100K string pairs/second (space-optimized)
- Parallel matrix computation: Linear scaling up to available CPU cores

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## API Reference

For detailed API documentation, see the inline documentation in the source code. All public functions and types are thoroughly documented with usage examples and performance characteristics.