const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;

/// Compute Hamming distance between two strings
/// Returns number of positions where characters differ
/// Strings must be of equal length
pub fn hamming(str1: StringValue, str2: StringValue) f64 {
    if (str1.len != str2.len) return std.math.inf(f64);
    if (str1.getType() != str2.getType()) return std.math.inf(f64);

    var distance: usize = 0;
    for (0..str1.len) |i| {
        if (str1.get(i) != str2.get(i)) {
            distance += 1;
        }
    }

    return @floatFromInt(distance);
}

/// Compute Levenshtein distance between two strings
/// Returns minimum number of single-character edits needed
pub fn levenshtein(allocator: Allocator, str1: StringValue, str2: StringValue) !f64 {
    if (str1.getType() != str2.getType()) return std.math.inf(f64);

    const m = str1.len;
    const n = str2.len;

    // Handle edge cases
    if (m == 0) return @floatFromInt(n);
    if (n == 0) return @floatFromInt(m);

    // Create DP matrix
    const matrix = try allocator.alloc([]usize, m + 1);
    defer allocator.free(matrix);

    for (0..m + 1) |i| {
        matrix[i] = try allocator.alloc(usize, n + 1);
    }
    defer for (0..m + 1) |i| {
        allocator.free(matrix[i]);
    };

    // Initialize base cases
    for (0..m + 1) |i| {
        matrix[i][0] = i;
    }
    for (0..n + 1) |j| {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            const cost: usize = if (str1.get(i - 1) == str2.get(j - 1)) 0 else 1;

            matrix[i][j] = @min(@min(matrix[i - 1][j] + 1, // deletion
                matrix[i][j - 1] + 1 // insertion
                ), matrix[i - 1][j - 1] + cost); // substitution
        }
    }

    return @floatFromInt(matrix[m][n]);
}

/// Compute optimized Levenshtein distance with O(min(m,n)) space
pub fn levenshteinOptimized(allocator: Allocator, str1: StringValue, str2: StringValue) !f64 {
    if (str1.getType() != str2.getType()) return std.math.inf(f64);

    var s1 = str1;
    var s2 = str2;

    // Ensure s1 is the shorter string for space optimization
    if (s1.len > s2.len) {
        const temp = s1;
        s1 = s2;
        s2 = temp;
    }

    const m = s1.len;
    const n = s2.len;

    if (m == 0) return @floatFromInt(n);

    // Use two rows instead of full matrix
    var prev_row = try allocator.alloc(usize, m + 1);
    defer allocator.free(prev_row);
    var curr_row = try allocator.alloc(usize, m + 1);
    defer allocator.free(curr_row);

    // Initialize first row
    for (0..m + 1) |i| {
        prev_row[i] = i;
    }

    for (1..n + 1) |j| {
        curr_row[0] = j;

        for (1..m + 1) |i| {
            const cost: usize = if (s1.get(i - 1) == s2.get(j - 1)) 0 else 1;

            curr_row[i] = @min(@min(prev_row[i] + 1, // deletion
                curr_row[i - 1] + 1 // insertion
                ), prev_row[i - 1] + cost); // substitution
        }

        // Swap rows
        const temp = prev_row;
        prev_row = curr_row;
        curr_row = temp;
    }

    return @floatFromInt(prev_row[m]);
}

/// Compute Jaro similarity between two strings
pub fn jaro(allocator: Allocator, str1: StringValue, str2: StringValue) !f64 {
    if (str1.getType() != str2.getType()) return 0.0;
    if (str1.len == 0 and str2.len == 0) return 1.0;
    if (str1.len == 0 or str2.len == 0) return 0.0;

    const m = str1.len;
    const n = str2.len;

    // Calculate matching window
    const match_window = (@max(m, n) / 2) -| 1;

    var s1_matches = try allocator.alloc(bool, m);
    defer allocator.free(s1_matches);
    var s2_matches = try allocator.alloc(bool, n);
    defer allocator.free(s2_matches);

    @memset(s1_matches, false);
    @memset(s2_matches, false);

    var matches: f64 = 0;
    var transpositions: f64 = 0;

    // Find matches
    for (0..m) |i| {
        const start = if (i >= match_window) i - match_window else 0;
        const end = @min(i + match_window + 1, n);

        for (start..end) |j| {
            if (s2_matches[j] or str1.get(i) != str2.get(j)) continue;

            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if (matches == 0) return 0.0;

    // Count transpositions
    var k: usize = 0;
    for (0..m) |i| {
        if (!s1_matches[i]) continue;

        while (!s2_matches[k]) k += 1;

        if (str1.get(i) != str2.get(k)) {
            transpositions += 1;
        }
        k += 1;
    }

    const jaro_similarity = (matches / @as(f64, @floatFromInt(m)) +
        matches / @as(f64, @floatFromInt(n)) +
        (matches - transpositions / 2.0) / matches) / 3.0;

    return jaro_similarity;
}

/// Compute Jaro-Winkler similarity with prefix scaling
pub fn jaroWinkler(allocator: Allocator, str1: StringValue, str2: StringValue) !f64 {
    const jaro_sim = try jaro(allocator, str1, str2);

    if (jaro_sim < 0.7) return jaro_sim;

    // Calculate common prefix length (up to 4 characters)
    var prefix_len: usize = 0;
    const max_prefix = @min(@min(str1.len, str2.len), 4);

    for (0..max_prefix) |i| {
        if (str1.get(i) == str2.get(i)) {
            prefix_len += 1;
        } else {
            break;
        }
    }

    const scaling_factor = 0.1;
    return jaro_sim + (scaling_factor * @as(f64, @floatFromInt(prefix_len)) * (1.0 - jaro_sim));
}

/// Compute Jaro-Winkler distance (1 - similarity)
pub fn jaroWinklerDistance(allocator: Allocator, str1: StringValue, str2: StringValue) !f64 {
    const similarity = try jaroWinkler(allocator, str1, str2);
    return 1.0 - similarity;
}

// Tests
test "Hamming distance basic" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "kitten");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "sitten");
    defer str2.deinit();

    const dist = hamming(str1, str2);
    try testing.expect(dist == 1.0); // Only first character differs
}

test "Hamming distance different lengths" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "cat");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "cats");
    defer str2.deinit();

    const dist = hamming(str1, str2);
    try testing.expect(std.math.isInf(dist)); // Different lengths
}

test "Hamming distance identical" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "hello");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "hello");
    defer str2.deinit();

    const dist = hamming(str1, str2);
    try testing.expect(dist == 0.0);
}

test "Levenshtein distance basic" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "kitten");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "sitting");
    defer str2.deinit();

    const dist = try levenshtein(allocator, str1, str2);
    try testing.expect(dist == 3.0); // k->s, e->i, insert t
}

test "Levenshtein distance empty strings" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "abc");
    defer str2.deinit();

    const dist = try levenshtein(allocator, str1, str2);
    try testing.expect(dist == 3.0);
}

test "Levenshtein optimized vs standard" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "intention");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "execution");
    defer str2.deinit();

    const dist1 = try levenshtein(allocator, str1, str2);
    const dist2 = try levenshteinOptimized(allocator, str1, str2);

    try testing.expect(dist1 == dist2);
    try testing.expect(dist1 == 5.0);
}

test "Jaro similarity basic" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "martha");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "marhta");
    defer str2.deinit();

    const sim = try jaro(allocator, str1, str2);
    // Jaro similarity for "martha" and "marhta" should be around 0.944
    try testing.expect(sim > 0.9 and sim < 1.0);
}

test "Jaro similarity identical" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "test");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "test");
    defer str2.deinit();

    const sim = try jaro(allocator, str1, str2);
    try testing.expect(sim == 1.0);
}

test "Jaro similarity empty strings" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "");
    defer str2.deinit();

    const sim = try jaro(allocator, str1, str2);
    try testing.expect(sim == 1.0);
}

test "Jaro-Winkler distance" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "dixon");
    defer str1.deinit();
    var str2 = try StringValue.fromBytes(allocator, "dicksonx");
    defer str2.deinit();

    const dist = try jaroWinklerDistance(allocator, str1, str2);
    try testing.expect(dist > 0.0 and dist < 1.0);
}

test "Distance measures with tokens" {
    const allocator = testing.allocator;

    const tokens1 = [_]u64{ 1, 2, 3 };
    const tokens2 = [_]u64{ 1, 3, 2 };

    var str1 = try StringValue.fromTokens(allocator, &tokens1);
    defer str1.deinit();
    var str2 = try StringValue.fromTokens(allocator, &tokens2);
    defer str2.deinit();

    const hamming_dist = hamming(str1, str2);
    try testing.expect(hamming_dist == 2.0); // positions 1 and 2 differ

    const levenshtein_dist = try levenshtein(allocator, str1, str2);
    try testing.expect(levenshtein_dist == 2.0); // swap operations
}
