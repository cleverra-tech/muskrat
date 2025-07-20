const std = @import("std");
const testing = std.testing;
const StringValue = @import("string.zig").StringValue;

/// SIMD-optimized implementations of performance-critical functions
///
/// This module provides vectorized versions of string processing operations
/// that can significantly improve performance on supported architectures.
/// Vector size for SIMD operations (128-bit vectors = 16 bytes)
const VECTOR_SIZE = 16;
const Vector16u8 = @Vector(VECTOR_SIZE, u8);

/// Check if the current platform supports SIMD operations
pub fn hasSIMDSupport() bool {
    // Zig's @Vector is portable and will use SIMD when available
    return true;
}

/// SIMD-optimized Hamming distance for byte strings
///
/// This function uses vectorized operations to compare chunks of bytes
/// simultaneously, providing significant speedup for large strings.
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

    // Process chunks of VECTOR_SIZE bytes using SIMD
    const simd_chunks = len / VECTOR_SIZE;
    var i: usize = 0;

    while (i < simd_chunks) : (i += 1) {
        const offset = i * VECTOR_SIZE;

        // Load VECTOR_SIZE bytes from each string
        const vec1: Vector16u8 = data1[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const vec2: Vector16u8 = data2[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;

        // XOR the vectors to find differences
        const diff: Vector16u8 = vec1 ^ vec2;

        // Count non-zero bytes (differences)
        for (0..VECTOR_SIZE) |j| {
            if (diff[j] != 0) {
                distance += 1;
            }
        }
    }

    // Handle remaining bytes that don't fit in a full vector
    const remaining_start = simd_chunks * VECTOR_SIZE;
    for (remaining_start..len) |idx| {
        if (data1[idx] != data2[idx]) {
            distance += 1;
        }
    }

    return @floatFromInt(distance);
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
pub fn toLowerSIMD(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const output = try allocator.alloc(u8, input.len);
    errdefer allocator.free(output);

    const len = input.len;
    const simd_chunks = len / VECTOR_SIZE;
    var i: usize = 0;

    // SIMD processing for full chunks
    while (i < simd_chunks) : (i += 1) {
        const offset = i * VECTOR_SIZE;

        // Load input chunk
        const input_vec: Vector16u8 = input[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;

        // Create mask for uppercase letters (A-Z = 65-90)
        const ge_A: Vector16u8 = @splat(65); // 'A'
        const le_Z: Vector16u8 = @splat(90); // 'Z'

        // Check if each byte is in range [A, Z]
        const is_upper_lower = input_vec >= ge_A;
        const is_upper_upper = input_vec <= le_Z;
        const is_upper_bool = is_upper_lower & is_upper_upper;

        // Convert boolean mask to u8 vector (true -> 32, false -> 0)
        const mask_add: Vector16u8 = @splat(32);
        const mask_zero: Vector16u8 = @splat(0);
        const case_adjustment: Vector16u8 = @select(u8, is_upper_bool, mask_add, mask_zero);

        // Convert to lowercase by adding the adjustment
        const lowercase_vec = input_vec + case_adjustment;

        // Store result
        output[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].* = lowercase_vec;
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * VECTOR_SIZE;
    for (remaining_start..len) |idx| {
        output[idx] = std.ascii.toLower(input[idx]);
    }

    return output;
}

/// SIMD-optimized character matching for Jaro algorithm
///
/// Checks if a character from str1 matches any character in a window of str2.
/// Returns the index of the first match, or null if no match found.
pub fn findCharMatchSIMD(target_char: u8, window: []const u8) ?usize {
    if (window.len == 0) return null;

    const len = window.len;
    const simd_chunks = len / VECTOR_SIZE;
    const target_vec: Vector16u8 = @splat(target_char);
    var i: usize = 0;

    // SIMD search in full chunks
    while (i < simd_chunks) : (i += 1) {
        const offset = i * VECTOR_SIZE;
        const window_vec: Vector16u8 = window[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;

        // Compare target with all bytes in chunk
        const matches = target_vec == window_vec;

        // Check if any matches found
        for (0..VECTOR_SIZE) |j| {
            if (matches[j]) {
                return offset + j;
            }
        }
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * VECTOR_SIZE;
    for (remaining_start..len) |idx| {
        if (window[idx] == target_char) {
            return idx;
        }
    }

    return null;
}

/// SIMD-optimized memory comparison
///
/// Compares two byte arrays and returns true if they are identical.
/// Uses vectorized operations for better performance on large arrays.
pub fn memoryEqualSIMD(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    if (a.len == 0) return true;

    const len = a.len;
    const simd_chunks = len / VECTOR_SIZE;
    var i: usize = 0;

    // SIMD comparison for full chunks
    while (i < simd_chunks) : (i += 1) {
        const offset = i * VECTOR_SIZE;
        const vec_a: Vector16u8 = a[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const vec_b: Vector16u8 = b[offset .. offset + VECTOR_SIZE][0..VECTOR_SIZE].*;

        const equal = vec_a == vec_b;

        // Check if all bytes in chunk are equal
        for (0..VECTOR_SIZE) |j| {
            if (!equal[j]) {
                return false;
            }
        }
    }

    // Handle remaining bytes
    const remaining_start = simd_chunks * VECTOR_SIZE;
    for (remaining_start..len) |idx| {
        if (a[idx] != b[idx]) {
            return false;
        }
    }

    return true;
}

// Tests
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
