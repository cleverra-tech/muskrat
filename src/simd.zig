const std = @import("std");
const testing = std.testing;
const StringValue = @import("string.zig").StringValue;

/// SIMD-optimized implementations of performance-critical functions
///
/// This module provides vectorized versions of string processing operations
/// that can significantly improve performance on supported architectures.
/// SIMD configuration constants
const VECTOR_SIZE_128 = 16; // 128-bit vectors = 16 bytes
const VECTOR_SIZE_256 = 32; // 256-bit vectors = 32 bytes
const VECTOR_SIZE_DEFAULT = VECTOR_SIZE_128; // Use 128-bit by default for compatibility

const Vector16u8 = @Vector(VECTOR_SIZE_128, u8);
const Vector32u8 = @Vector(VECTOR_SIZE_256, u8);

/// Platform capabilities for SIMD optimization
pub const SIMDCapabilities = struct {
    has_sse: bool = false,
    has_avx2: bool = false,
    has_neon: bool = false,
    preferred_vector_size: usize = VECTOR_SIZE_DEFAULT,
};

/// Detect SIMD capabilities at runtime
pub fn detectSIMDCapabilities() SIMDCapabilities {
    var caps = SIMDCapabilities{};

    // Zig's @Vector automatically uses available SIMD when beneficial
    // For now, assume we have basic SIMD support and use conservative defaults
    caps.preferred_vector_size = VECTOR_SIZE_DEFAULT;

    // On x86_64, we can generally assume SSE2 support
    // On ARM64, we can generally assume NEON support
    switch (@import("builtin").target.cpu.arch) {
        .x86_64 => {
            caps.has_sse = true;
            // More sophisticated detection could use CPUID here
        },
        .aarch64 => {
            caps.has_neon = true;
        },
        else => {},
    }

    return caps;
}

/// Check if the current platform supports SIMD operations
pub fn hasSIMDSupport() bool {
    const caps = detectSIMDCapabilities();
    return caps.has_sse or caps.has_neon;
}

/// SIMD-optimized Hamming distance for byte strings
///
/// This function uses vectorized operations to compare chunks of bytes
/// simultaneously, providing significant speedup for large strings.
/// Uses efficient bit manipulation for counting differences.
pub fn hammingDistanceSIMD(str1: StringValue, str2: StringValue) f64 {
    // Ensure we're working with byte strings of same length
    if (str1.getType() != .byte or str2.getType() != .byte) {
        // Fallback to scalar version for non-byte strings
        return hammingDistanceScalar(str1, str2);
    }

    if (str1.len != str2.len) {
        // Different lengths - Hamming distance is undefined, return infinity
        return std.math.inf(f64);
    }

    const data1 = switch (str1.data) {
        .byte => |bytes| bytes,
        else => {
            // SIMD Hamming distance only supports byte strings
            return std.math.inf(f64);
        },
    };
    const data2 = switch (str2.data) {
        .byte => |bytes| bytes,
        else => {
            // SIMD Hamming distance only supports byte strings
            return std.math.inf(f64);
        },
    };

    var distance: u32 = 0;
    const len = str1.len;
    const caps = detectSIMDCapabilities();
    const vector_size = caps.preferred_vector_size;

    // Process chunks using preferred vector size
    const simd_chunks = len / vector_size;
    var i: usize = 0;

    if (vector_size == VECTOR_SIZE_128) {
        while (i < simd_chunks) : (i += 1) {
            const offset = i * VECTOR_SIZE_128;

            // Load 16 bytes from each string
            const vec1: Vector16u8 = data1[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;
            const vec2: Vector16u8 = data2[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;

            // XOR the vectors to find differences
            const diff: Vector16u8 = vec1 ^ vec2;

            // Convert to boolean vector (0 -> false, non-zero -> true)
            const zero_vec: Vector16u8 = @splat(0);
            const diff_mask = diff != zero_vec;

            // Count the number of differences using efficient reduction
            distance += countTrueBits(diff_mask);
        }
    }

    // Handle remaining bytes that don't fit in a full vector
    const remaining_start = simd_chunks * vector_size;
    for (remaining_start..len) |idx| {
        if (data1[idx] != data2[idx]) {
            distance += 1;
        }
    }

    return @floatFromInt(distance);
}

/// Efficiently count the number of true bits in a boolean vector
fn countTrueBits(mask: @Vector(VECTOR_SIZE_128, bool)) u32 {
    // Convert boolean vector to integer vector for efficient processing
    const ones: @Vector(VECTOR_SIZE_128, u8) = @splat(1);
    const zeros: @Vector(VECTOR_SIZE_128, u8) = @splat(0);
    const as_int: @Vector(VECTOR_SIZE_128, u8) = @select(u8, mask, ones, zeros);

    // Use @reduce for efficient horizontal sum - this is optimized by Zig
    const sum = @reduce(.Add, as_int);

    return @as(u32, sum);
}

/// Scalar fallback for Hamming distance
fn hammingDistanceScalar(str1: StringValue, str2: StringValue) f64 {
    if (str1.len != str2.len) {
        return std.math.inf(f64);
    }

    var distance: u32 = 0;
    for (0..str1.len) |i| {
        if (str1.get(i) != str2.get(i)) {
            distance += 1;
        }
    }

    return @floatFromInt(distance);
}

/// SIMD-optimized case normalization to lowercase
///
/// Converts ASCII strings to lowercase using vectorized operations.
/// Non-ASCII characters are passed through unchanged.
/// Uses efficient vectorized range checking and conditional updates.
pub fn toLowerSIMD(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const output = try allocator.alloc(u8, input.len);
    errdefer allocator.free(output);

    const len = input.len;
    const caps = detectSIMDCapabilities();
    const vector_size = caps.preferred_vector_size;
    const simd_chunks = len / vector_size;
    var i: usize = 0;

    // SIMD processing for full chunks
    if (vector_size == VECTOR_SIZE_128) {
        while (i < simd_chunks) : (i += 1) {
            const offset = i * VECTOR_SIZE_128;

            // Load input chunk
            const input_vec: Vector16u8 = input[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;

            // Optimized lowercase conversion using SIMD
            const lowercase_vec = toLowerVector128(input_vec);

            // Store result
            output[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].* = lowercase_vec;
        }
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * vector_size;
    for (remaining_start..len) |idx| {
        output[idx] = std.ascii.toLower(input[idx]);
    }

    return output;
}

/// Efficiently convert a 128-bit vector to lowercase using SIMD operations
fn toLowerVector128(input_vec: Vector16u8) Vector16u8 {
    // Create mask for uppercase letters (A-Z = 65-90)
    const ge_A: Vector16u8 = @splat(65); // 'A'
    const le_Z: Vector16u8 = @splat(90); // 'Z'

    // Check if each byte is in range [A, Z] using efficient range check
    const is_upper_lower = input_vec >= ge_A;
    const is_upper_upper = input_vec <= le_Z;
    const is_upper_bool = is_upper_lower & is_upper_upper;

    // Convert boolean mask to u8 vector efficiently (true -> 32, false -> 0)
    const case_offset: Vector16u8 = @splat(32); // 'a' - 'A' = 32
    const zero_vec: Vector16u8 = @splat(0);
    const case_adjustment: Vector16u8 = @select(u8, is_upper_bool, case_offset, zero_vec);

    // Convert to lowercase by adding the adjustment
    return input_vec + case_adjustment;
}

/// SIMD-optimized character matching for Jaro algorithm
///
/// Checks if a character from str1 matches any character in a window of str2.
/// Returns the index of the first match, or null if no match found.
/// Uses efficient bit scanning for finding the first match position.
pub fn findCharMatchSIMD(target_char: u8, window: []const u8) ?usize {
    if (window.len == 0) return null;

    const len = window.len;
    const caps = detectSIMDCapabilities();
    const vector_size = caps.preferred_vector_size;
    const simd_chunks = len / vector_size;
    var i: usize = 0;

    // SIMD search in full chunks
    if (vector_size == VECTOR_SIZE_128) {
        const target_vec: Vector16u8 = @splat(target_char);

        while (i < simd_chunks) : (i += 1) {
            const offset = i * VECTOR_SIZE_128;
            const window_vec: Vector16u8 = window[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;

            // Compare target with all bytes in chunk
            const matches = target_vec == window_vec;

            // Find first match using efficient bit scanning
            const match_index = findFirstTrueBit(matches);
            if (match_index) |idx| {
                return offset + idx;
            }
        }
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * vector_size;
    for (remaining_start..len) |idx| {
        if (window[idx] == target_char) {
            return idx;
        }
    }

    return null;
}

/// Find the index of the first true bit in a boolean vector
/// Returns null if no true bits are found
fn findFirstTrueBit(mask: @Vector(VECTOR_SIZE_128, bool)) ?usize {
    // Convert boolean vector to bitmask for efficient scanning
    inline for (0..VECTOR_SIZE_128) |i| {
        if (mask[i]) {
            return i;
        }
    }
    return null;
}

/// SIMD-optimized memory comparison
///
/// Compares two byte arrays and returns true if they are identical.
/// Uses vectorized operations for better performance on large arrays.
/// Employs early termination for maximum efficiency.
pub fn memoryEqualSIMD(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    if (a.len == 0) return true;

    const len = a.len;
    const caps = detectSIMDCapabilities();
    const vector_size = caps.preferred_vector_size;
    const simd_chunks = len / vector_size;
    var i: usize = 0;

    // SIMD comparison for full chunks
    if (vector_size == VECTOR_SIZE_128) {
        while (i < simd_chunks) : (i += 1) {
            const offset = i * VECTOR_SIZE_128;
            const vec_a: Vector16u8 = a[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;
            const vec_b: Vector16u8 = b[offset .. offset + VECTOR_SIZE_128][0..VECTOR_SIZE_128].*;

            const equal = vec_a == vec_b;

            // Check if all bytes in chunk are equal using efficient reduction
            if (!allTrueBits(equal)) {
                return false;
            }
        }
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * vector_size;
    for (remaining_start..len) |idx| {
        if (a[idx] != b[idx]) {
            return false;
        }
    }

    return true;
}

/// Check if all bits in a boolean vector are true
fn allTrueBits(mask: @Vector(VECTOR_SIZE_128, bool)) bool {
    // Convert to integer representation for efficient testing
    const ones: @Vector(VECTOR_SIZE_128, u8) = @splat(1);
    const zeros: @Vector(VECTOR_SIZE_128, u8) = @splat(0);
    const as_int: @Vector(VECTOR_SIZE_128, u8) = @select(u8, mask, ones, zeros);

    // Use horizontal reduction to check if all are 1
    // If any bit is 0, the sum will be less than VECTOR_SIZE_128
    const sum = @reduce(.Add, as_int);
    return sum == VECTOR_SIZE_128;
}

// Tests
test "SIMD capabilities detection" {
    const caps = detectSIMDCapabilities();

    // Should have a reasonable preferred vector size
    try testing.expect(caps.preferred_vector_size == VECTOR_SIZE_128 or caps.preferred_vector_size == VECTOR_SIZE_256);

    // Should detect some form of SIMD support on modern architectures
    const has_support = hasSIMDSupport();

    // This might be false on very old architectures, but should generally be true
    // We just test that the function runs without crashing
    _ = has_support;
}

test "SIMD bit counting optimization" {
    // Test the optimized bit counting function
    const all_false: @Vector(VECTOR_SIZE_128, bool) = @splat(false);
    const all_true: @Vector(VECTOR_SIZE_128, bool) = @splat(true);

    try testing.expect(countTrueBits(all_false) == 0);
    try testing.expect(countTrueBits(all_true) == VECTOR_SIZE_128);

    // Test mixed pattern
    var mixed: @Vector(VECTOR_SIZE_128, bool) = @splat(false);
    mixed[0] = true;
    mixed[5] = true;
    mixed[15] = true;

    try testing.expect(countTrueBits(mixed) == 3);
}

test "SIMD all bits checking" {
    const all_false: @Vector(VECTOR_SIZE_128, bool) = @splat(false);
    const all_true: @Vector(VECTOR_SIZE_128, bool) = @splat(true);

    try testing.expect(!allTrueBits(all_false));
    try testing.expect(allTrueBits(all_true));

    // Test with one false bit
    var mostly_true: @Vector(VECTOR_SIZE_128, bool) = @splat(true);
    mostly_true[7] = false;

    try testing.expect(!allTrueBits(mostly_true));
}

test "SIMD Hamming distance basic functionality" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "hello");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "hallo");
    defer str2.deinit();

    const distance = hammingDistanceSIMD(str1, str2);
    try testing.expect(distance == 1.0); // 'e' vs 'a'
}

test "SIMD Hamming distance identical strings" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "identical");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "identical");
    defer str2.deinit();

    const distance = hammingDistanceSIMD(str1, str2);
    try testing.expect(distance == 0.0);
}

test "SIMD Hamming distance different lengths" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "short");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "longer");
    defer str2.deinit();

    const distance = hammingDistanceSIMD(str1, str2);
    try testing.expect(std.math.isInf(distance)); // undefined for different lengths
}

test "SIMD Hamming distance long strings" {
    const allocator = testing.allocator;

    // Test with strings longer than VECTOR_SIZE to exercise SIMD path
    const long_str1 = "this is a very long string that should trigger SIMD processing";
    const long_str2 = "this is a very long string that should trigger SIMD processing";

    var str1 = try StringValue.fromBytes(allocator, long_str1);
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, long_str2);
    defer str2.deinit();

    const distance = hammingDistanceSIMD(str1, str2);
    try testing.expect(distance == 0.0);
}

test "SIMD toLower functionality" {
    const allocator = testing.allocator;

    const input = "Hello World! 123";
    const result = try toLowerSIMD(allocator, input);
    defer allocator.free(result);

    try testing.expectEqualStrings("hello world! 123", result);
}

test "SIMD toLower long string" {
    const allocator = testing.allocator;

    const input = "THIS IS A VERY LONG STRING WITH UPPERCASE LETTERS THAT SHOULD BE CONVERTED";
    const expected = "this is a very long string with uppercase letters that should be converted";

    const result = try toLowerSIMD(allocator, input);
    defer allocator.free(result);

    try testing.expectEqualStrings(expected, result);
}

test "SIMD character matching" {
    const window = "abcdefghijklmnop";

    try testing.expect(findCharMatchSIMD('a', window) == 0);
    try testing.expect(findCharMatchSIMD('f', window) == 5);
    try testing.expect(findCharMatchSIMD('p', window) == 15);
    try testing.expect(findCharMatchSIMD('z', window) == null);
}

test "SIMD memory equality" {
    const a = "testing memory equality";
    const b = "testing memory equality";
    const c = "testing memory different";

    try testing.expect(memoryEqualSIMD(a, b) == true);
    try testing.expect(memoryEqualSIMD(a, c) == false);
    try testing.expect(memoryEqualSIMD("", "") == true);
}

test "SIMD memory equality different lengths" {
    const a = "short";
    const b = "longer";

    try testing.expect(memoryEqualSIMD(a, b) == false);
}
