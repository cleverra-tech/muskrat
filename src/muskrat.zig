//! Muskrat - A comprehensive string similarity computation library
//!
//! This library provides efficient implementations of various string distance measures,
//! similarity coefficients, and kernel functions for analyzing text similarity.
//!
//! Features:
//! - Multiple string representations (byte, token, bit)
//! - Distance measures (Hamming, Levenshtein, Jaro, Jaro-Winkler)
//! - Similarity coefficients (Jaccard, Dice, Simpson, Cosine)
//! - Kernel functions (spectrum, subsequence, mismatch)
//! - SIMD optimizations for performance
//! - Parallel computation support
//! - Multiple output formats (text, JSON, binary, CSV)
//! - Memory pooling and efficient allocation
//! - Comprehensive benchmarking and testing
//!
//! Basic usage:
//! ```zig
//! const muskrat = @import("muskrat");
//!
//! // Create string values
//! const str1 = try muskrat.StringValue.fromBytes(allocator, "hello");
//! const str2 = try muskrat.StringValue.fromBytes(allocator, "world");
//!
//! // Compute distance
//! const hamming_dist = muskrat.distances.hamming(str1, str2);
//!
//! // Create and populate similarity matrix
//! var matrix = try muskrat.Matrix.init(allocator, &[_]muskrat.StringValue{str1, str2});
//! defer matrix.deinit();
//! ```

const std = @import("std");
const testing = std.testing;

pub const StringValue = @import("string.zig").StringValue;
pub const StringType = @import("string.zig").StringType;
pub const Matrix = @import("matrix.zig").Matrix;
pub const Range = @import("matrix.zig").Range;
pub const Processor = @import("processing.zig").Processor;

// Distance measures
pub const distances = @import("distances.zig");

// Parallel computation
pub const ParallelCompute = @import("parallel.zig").ParallelCompute;
pub const ParallelConfig = @import("parallel.zig").ParallelConfig;

// Configuration
pub const Config = @import("config.zig").Config;

// Kernel functions
pub const kernels = @import("kernels.zig");

// Similarity coefficients
pub const coefficients = @import("coefficients.zig");

// Input readers
pub const readers = @import("readers.zig");

// Output formatters
pub const formatters = @import("formatters.zig");

// Memory management
pub const memory = @import("memory.zig");

// Benchmarking and testing
pub const benchmark = @import("benchmark.zig");

// Comprehensive testing
pub const comprehensive_test = @import("comprehensive_test.zig");

// SIMD optimizations
pub const simd = @import("simd.zig");

test {
    testing.refAllDecls(@This());
}
