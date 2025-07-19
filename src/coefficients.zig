const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;

/// Set operations for similarity coefficient computation
const SetOperations = struct {
    /// Extract n-grams from string as a set
    pub fn extractNgrams(allocator: Allocator, str: StringValue, n: usize) !std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage) {
        var ngrams = std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage).init(allocator);

        if (str.len < n or n == 0) return ngrams;

        const data = switch (str.data) {
            .byte => |bytes| bytes,
            else => return ngrams, // Only support byte strings for now
        };

        for (0..str.len - n + 1) |i| {
            const ngram = data[i .. i + n];
            const owned_ngram = try allocator.dupe(u8, ngram);

            const result = try ngrams.getOrPut(owned_ngram);
            if (result.found_existing) {
                // Free the duplicate we just created
                allocator.free(owned_ngram);
            }
        }

        return ngrams;
    }

    /// Extract character set from string
    pub fn extractCharSet(allocator: Allocator, str: StringValue) !std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage) {
        var chars = std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage).init(allocator);

        const data = switch (str.data) {
            .byte => |bytes| bytes,
            else => return chars, // Only support byte strings for now
        };

        for (data) |char| {
            try chars.put(char, {});
        }

        return chars;
    }

    /// Compute intersection size of two n-gram sets
    pub fn intersectionSize(set1: *const std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage), set2: *const std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage)) usize {
        var count: usize = 0;
        var iter = set1.iterator();
        while (iter.next()) |entry| {
            if (set2.contains(entry.key_ptr.*)) {
                count += 1;
            }
        }
        return count;
    }

    /// Compute intersection size of two character sets
    pub fn charIntersectionSize(set1: *const std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage), set2: *const std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage)) usize {
        var count: usize = 0;
        var iter = set1.iterator();
        while (iter.next()) |entry| {
            if (set2.contains(entry.key_ptr.*)) {
                count += 1;
            }
        }
        return count;
    }

    /// Compute union size of two n-gram sets
    pub fn unionSize(set1: *const std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage), set2: *const std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage)) usize {
        return set1.count() + set2.count() - intersectionSize(set1, set2);
    }

    /// Compute union size of two character sets
    pub fn charUnionSize(set1: *const std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage), set2: *const std.HashMap(u8, void, std.hash_map.AutoContext(u8), std.hash_map.default_max_load_percentage)) usize {
        return set1.count() + set2.count() - charIntersectionSize(set1, set2);
    }

    /// Cleanup n-gram set memory
    pub fn cleanupNgramSet(allocator: Allocator, set: *std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage)) void {
        var iter = set.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        set.deinit();
    }
};

/// Jaccard similarity coefficient
pub const JaccardCoefficient = struct {
    n: usize, // n-gram size
    allocator: Allocator,

    const Self = @This();

    /// Initialize Jaccard coefficient with n-gram size
    pub fn init(allocator: Allocator, n: usize) Self {
        return Self{
            .allocator = allocator,
            .n = n,
        };
    }

    /// Compute Jaccard similarity between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.n == 0) return 0.0;

        var set1 = try SetOperations.extractNgrams(self.allocator, str1, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set1);

        var set2 = try SetOperations.extractNgrams(self.allocator, str2, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set2);

        const intersection = SetOperations.intersectionSize(&set1, &set2);
        const union_size = SetOperations.unionSize(&set1, &set2);

        if (union_size == 0) return 1.0; // Both empty
        return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_size));
    }

    /// Compute character-based Jaccard similarity
    pub fn computeCharBased(self: Self, str1: StringValue, str2: StringValue) !f64 {
        var set1 = try SetOperations.extractCharSet(self.allocator, str1);
        defer set1.deinit();

        var set2 = try SetOperations.extractCharSet(self.allocator, str2);
        defer set2.deinit();

        const intersection = SetOperations.charIntersectionSize(&set1, &set2);
        const union_size = SetOperations.charUnionSize(&set1, &set2);

        if (union_size == 0) return 1.0; // Both empty
        return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_size));
    }
};

/// Dice similarity coefficient (also known as Sørensen-Dice)
pub const DiceCoefficient = struct {
    n: usize, // n-gram size
    allocator: Allocator,

    const Self = @This();

    /// Initialize Dice coefficient with n-gram size
    pub fn init(allocator: Allocator, n: usize) Self {
        return Self{
            .allocator = allocator,
            .n = n,
        };
    }

    /// Compute Dice similarity between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.n == 0) return 0.0;

        var set1 = try SetOperations.extractNgrams(self.allocator, str1, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set1);

        var set2 = try SetOperations.extractNgrams(self.allocator, str2, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set2);

        const intersection = SetOperations.intersectionSize(&set1, &set2);
        const total_size = set1.count() + set2.count();

        if (total_size == 0) return 1.0; // Both empty
        return (2.0 * @as(f64, @floatFromInt(intersection))) / @as(f64, @floatFromInt(total_size));
    }

    /// Compute character-based Dice similarity
    pub fn computeCharBased(self: Self, str1: StringValue, str2: StringValue) !f64 {
        var set1 = try SetOperations.extractCharSet(self.allocator, str1);
        defer set1.deinit();

        var set2 = try SetOperations.extractCharSet(self.allocator, str2);
        defer set2.deinit();

        const intersection = SetOperations.charIntersectionSize(&set1, &set2);
        const total_size = set1.count() + set2.count();

        if (total_size == 0) return 1.0; // Both empty
        return (2.0 * @as(f64, @floatFromInt(intersection))) / @as(f64, @floatFromInt(total_size));
    }
};

/// Simpson similarity coefficient (also known as overlap coefficient)
pub const SimpsonCoefficient = struct {
    n: usize, // n-gram size
    allocator: Allocator,

    const Self = @This();

    /// Initialize Simpson coefficient with n-gram size
    pub fn init(allocator: Allocator, n: usize) Self {
        return Self{
            .allocator = allocator,
            .n = n,
        };
    }

    /// Compute Simpson similarity between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.n == 0) return 0.0;

        var set1 = try SetOperations.extractNgrams(self.allocator, str1, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set1);

        var set2 = try SetOperations.extractNgrams(self.allocator, str2, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set2);

        const intersection = SetOperations.intersectionSize(&set1, &set2);
        const min_size = @min(set1.count(), set2.count());

        if (min_size == 0) return 1.0; // At least one empty
        return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(min_size));
    }

    /// Compute character-based Simpson similarity
    pub fn computeCharBased(self: Self, str1: StringValue, str2: StringValue) !f64 {
        var set1 = try SetOperations.extractCharSet(self.allocator, str1);
        defer set1.deinit();

        var set2 = try SetOperations.extractCharSet(self.allocator, str2);
        defer set2.deinit();

        const intersection = SetOperations.charIntersectionSize(&set1, &set2);
        const min_size = @min(set1.count(), set2.count());

        if (min_size == 0) return 1.0; // At least one empty
        return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(min_size));
    }
};

/// Cosine similarity coefficient
pub const CosineCoefficient = struct {
    n: usize, // n-gram size
    allocator: Allocator,

    const Self = @This();

    /// Initialize Cosine coefficient with n-gram size
    pub fn init(allocator: Allocator, n: usize) Self {
        return Self{
            .allocator = allocator,
            .n = n,
        };
    }

    /// Compute Cosine similarity between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.n == 0) return 0.0;

        var set1 = try SetOperations.extractNgrams(self.allocator, str1, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set1);

        var set2 = try SetOperations.extractNgrams(self.allocator, str2, self.n);
        defer SetOperations.cleanupNgramSet(self.allocator, &set2);

        const intersection = SetOperations.intersectionSize(&set1, &set2);
        const denominator = std.math.sqrt(@as(f64, @floatFromInt(set1.count())) * @as(f64, @floatFromInt(set2.count())));

        if (denominator == 0.0) return 1.0; // Both empty
        return @as(f64, @floatFromInt(intersection)) / denominator;
    }

    /// Compute character-based Cosine similarity
    pub fn computeCharBased(self: Self, str1: StringValue, str2: StringValue) !f64 {
        var set1 = try SetOperations.extractCharSet(self.allocator, str1);
        defer set1.deinit();

        var set2 = try SetOperations.extractCharSet(self.allocator, str2);
        defer set2.deinit();

        const intersection = SetOperations.charIntersectionSize(&set1, &set2);
        const denominator = std.math.sqrt(@as(f64, @floatFromInt(set1.count())) * @as(f64, @floatFromInt(set2.count())));

        if (denominator == 0.0) return 1.0; // Both empty
        return @as(f64, @floatFromInt(intersection)) / denominator;
    }
};

/// String context for HashMap
const StringContext = struct {
    pub fn hash(self: @This(), key: []const u8) u64 {
        _ = self;
        return std.hash_map.hashString(key);
    }

    pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
        _ = self;
        return std.mem.eql(u8, a, b);
    }
};

// Tests
test "JaccardCoefficient basic functionality" {
    const allocator = testing.allocator;

    var jaccard = JaccardCoefficient.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abcd");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abcd");
    defer str2.deinit();

    var str3 = try StringValue.fromBytes(allocator, "efgh");
    defer str3.deinit();

    // Identical strings should have similarity 1.0
    const sim1 = try jaccard.compute(str1, str2);
    try testing.expect(sim1 == 1.0);

    // Completely different strings should have similarity 0.0
    const sim2 = try jaccard.compute(str1, str3);
    try testing.expect(sim2 == 0.0);
}

test "JaccardCoefficient partial overlap" {
    const allocator = testing.allocator;

    var jaccard = JaccardCoefficient.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    // "abc" has 2-grams: {ab, bc}
    // "bcd" has 2-grams: {bc, cd}
    // Intersection: {bc} = 1
    // Union: {ab, bc, cd} = 3
    // Jaccard = 1/3 ≈ 0.333
    const sim = try jaccard.compute(str1, str2);
    try testing.expect(sim > 0.3 and sim < 0.4);
}

test "JaccardCoefficient character-based" {
    const allocator = testing.allocator;

    var jaccard = JaccardCoefficient.init(allocator, 1);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    const sim = try jaccard.computeCharBased(str1, str2);
    // Characters in str1: {a, b, c}
    // Characters in str2: {b, c, d}
    // Intersection: {b, c} = 2
    // Union: {a, b, c, d} = 4
    // Jaccard = 2/4 = 0.5
    try testing.expect(sim == 0.5);
}

test "DiceCoefficient basic functionality" {
    const allocator = testing.allocator;

    var dice = DiceCoefficient.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    // "abc" has 2-grams: {ab, bc}
    // "bcd" has 2-grams: {bc, cd}
    // Intersection: {bc} = 1
    // Total: 2 + 2 = 4
    // Dice = 2*1/4 = 0.5
    const sim = try dice.compute(str1, str2);
    try testing.expect(sim == 0.5);
}

test "DiceCoefficient character-based" {
    const allocator = testing.allocator;

    var dice = DiceCoefficient.init(allocator, 1);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    const sim = try dice.computeCharBased(str1, str2);
    // Characters in str1: {a, b, c} = 3
    // Characters in str2: {b, c, d} = 3
    // Intersection: {b, c} = 2
    // Dice = 2*2/(3+3) = 4/6 ≈ 0.667
    try testing.expect(sim > 0.66 and sim < 0.68);
}

test "SimpsonCoefficient basic functionality" {
    const allocator = testing.allocator;

    var simpson = SimpsonCoefficient.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    // "abc" has 2-grams: {ab, bc}
    // "bcd" has 2-grams: {bc, cd}
    // Intersection: {bc} = 1
    // Min size: min(2, 2) = 2
    // Simpson = 1/2 = 0.5
    const sim = try simpson.compute(str1, str2);
    try testing.expect(sim == 0.5);
}

test "SimpsonCoefficient character-based" {
    const allocator = testing.allocator;

    var simpson = SimpsonCoefficient.init(allocator, 1);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    const sim = try simpson.computeCharBased(str1, str2);
    // Characters in str1: {a, b, c} = 3
    // Characters in str2: {b, c, d} = 3
    // Intersection: {b, c} = 2
    // Simpson = 2/min(3,3) = 2/3 ≈ 0.667
    try testing.expect(sim > 0.66 and sim < 0.68);
}

test "CosineCoefficient basic functionality" {
    const allocator = testing.allocator;

    var cosine = CosineCoefficient.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    // "abc" has 2-grams: {ab, bc}
    // "bcd" has 2-grams: {bc, cd}
    // Intersection: {bc} = 1
    // Cosine = 1/sqrt(2*2) = 1/2 = 0.5
    const sim = try cosine.compute(str1, str2);
    try testing.expect(sim == 0.5);
}

test "CosineCoefficient character-based" {
    const allocator = testing.allocator;

    var cosine = CosineCoefficient.init(allocator, 1);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "bcd");
    defer str2.deinit();

    const sim = try cosine.computeCharBased(str1, str2);
    // Characters in str1: {a, b, c} = 3
    // Characters in str2: {b, c, d} = 3
    // Intersection: {b, c} = 2
    // Cosine = 2/sqrt(3*3) = 2/3 ≈ 0.667
    try testing.expect(sim > 0.66 and sim < 0.68);
}

test "Empty string handling" {
    const allocator = testing.allocator;

    var jaccard = JaccardCoefficient.init(allocator, 2);

    var empty1 = try StringValue.fromBytes(allocator, "");
    defer empty1.deinit();

    var empty2 = try StringValue.fromBytes(allocator, "");
    defer empty2.deinit();

    var non_empty = try StringValue.fromBytes(allocator, "abc");
    defer non_empty.deinit();

    // Two empty strings should have similarity 1.0
    const sim1 = try jaccard.compute(empty1, empty2);
    try testing.expect(sim1 == 1.0);

    // Empty vs non-empty should have similarity 0.0
    const sim2 = try jaccard.compute(empty1, non_empty);
    try testing.expect(sim2 == 0.0);
}

test "Edge cases with n=0" {
    const allocator = testing.allocator;

    var jaccard = JaccardCoefficient.init(allocator, 0);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "def");
    defer str2.deinit();

    // n=0 should return 0.0
    const sim = try jaccard.compute(str1, str2);
    try testing.expect(sim == 0.0);
}
