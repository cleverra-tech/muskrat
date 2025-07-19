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
