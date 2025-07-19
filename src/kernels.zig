const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;
const Symbol = @import("string.zig").Symbol;

/// Spectrum kernel implementation
pub const SpectrumKernel = struct {
    k: usize,
    allocator: Allocator,

    const Self = @This();

    /// Initialize spectrum kernel with k-gram length
    pub fn init(allocator: Allocator, k: usize) Self {
        return Self{
            .allocator = allocator,
            .k = k,
        };
    }

    /// Compute spectrum kernel between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.k == 0) return 0.0;
        if (str1.len < self.k or str2.len < self.k) return 0.0;

        var kgrams1 = std.HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer self.cleanupKgrams(&kgrams1);

        var kgrams2 = std.HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer self.cleanupKgrams(&kgrams2);

        // Extract k-grams from first string
        try self.extractKgrams(str1, &kgrams1);

        // Extract k-grams from second string
        try self.extractKgrams(str2, &kgrams2);

        // Compute dot product
        var dot_product: f64 = 0.0;
        var iter = kgrams1.iterator();
        while (iter.next()) |entry| {
            if (kgrams2.get(entry.key_ptr.*)) |count2| {
                dot_product += @as(f64, @floatFromInt(entry.value_ptr.*)) * @as(f64, @floatFromInt(count2));
            }
        }

        return dot_product;
    }

    /// Extract k-grams from string and count frequencies
    fn extractKgrams(self: Self, str: StringValue, kgrams: *std.HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage)) !void {
        if (str.len < self.k) return;

        switch (str.data) {
            .byte => |bytes| {
                for (0..str.len - self.k + 1) |i| {
                    const kgram = bytes[i .. i + self.k];

                    // Create owned copy of k-gram
                    const owned_kgram = try self.allocator.dupe(u8, kgram);

                    const result = try kgrams.getOrPut(owned_kgram);
                    if (result.found_existing) {
                        // Free the duplicate we just created
                        self.allocator.free(owned_kgram);
                        result.value_ptr.* += 1;
                    } else {
                        result.value_ptr.* = 1;
                    }
                }
            },
            .token => |tokens| {
                for (0..str.len - self.k + 1) |i| {
                    // Create k-gram representation as concatenated token IDs
                    var kgram_data = std.ArrayList(u8).init(self.allocator);
                    defer kgram_data.deinit();

                    for (0..self.k) |j| {
                        const token_bytes = std.mem.asBytes(&tokens[i + j]);
                        try kgram_data.appendSlice(token_bytes);
                    }

                    const owned_kgram = try kgram_data.toOwnedSlice();

                    const result = try kgrams.getOrPut(owned_kgram);
                    if (result.found_existing) {
                        // Free the duplicate we just created
                        self.allocator.free(owned_kgram);
                        result.value_ptr.* += 1;
                    } else {
                        result.value_ptr.* = 1;
                    }
                }
            },
            .bit => |bits| {
                for (0..str.len - self.k + 1) |i| {
                    // Create k-gram representation from bit sequence
                    const byte_size = (self.k + 7) / 8;
                    var kgram_data = try self.allocator.alloc(u8, byte_size);
                    @memset(kgram_data, 0);

                    for (0..self.k) |j| {
                        const bit_pos = i + j;
                        const byte_idx = bit_pos / 8;
                        const bit_idx = @as(u3, @intCast(bit_pos % 8));
                        const bit_val = (bits[byte_idx] >> (7 - bit_idx)) & 1;

                        const out_byte_idx = j / 8;
                        const out_bit_idx = @as(u3, @intCast(j % 8));
                        if (bit_val == 1) {
                            kgram_data[out_byte_idx] |= @as(u8, 1) << (7 - out_bit_idx);
                        }
                    }

                    const result = try kgrams.getOrPut(kgram_data);
                    if (result.found_existing) {
                        // Free the duplicate we just created
                        self.allocator.free(kgram_data);
                        result.value_ptr.* += 1;
                    } else {
                        result.value_ptr.* = 1;
                    }
                }
            },
        }
    }

    /// Cleanup k-gram memory
    fn cleanupKgrams(self: Self, kgrams: *std.HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage)) void {
        var iter = kgrams.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        kgrams.deinit();
    }
};

/// Subsequence kernel implementation
pub const SubsequenceKernel = struct {
    k: usize,
    lambda: f64,
    allocator: Allocator,

    const Self = @This();

    /// Initialize subsequence kernel
    pub fn init(allocator: Allocator, k: usize, lambda: f64) Self {
        return Self{
            .allocator = allocator,
            .k = k,
            .lambda = lambda,
        };
    }

    /// Compute subsequence kernel between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) !f64 {
        if (self.k == 0 or self.lambda <= 0.0 or self.lambda >= 1.0) return 0.0;
        if (str1.len < self.k or str2.len < self.k) return 0.0;

        // Ensure both strings have the same type
        if (@intFromEnum(str1.data) != @intFromEnum(str2.data)) return 0.0;

        switch (str1.data) {
            .byte => |bytes1| {
                const bytes2 = str2.data.byte;
                return self.computeByteSubsequence(bytes1, bytes2);
            },
            .token => |tokens1| {
                const tokens2 = str2.data.token;
                return self.computeTokenSubsequence(tokens1, tokens2);
            },
            .bit => |bits1| {
                const bits2 = str2.data.bit;
                return self.computeBitSubsequence(bits1, bits2, str1.len, str2.len);
            },
        }
    }

    /// Compute subsequence kernel for byte strings
    fn computeByteSubsequence(self: Self, data1: []const u8, data2: []const u8) !f64 {
        // Dynamic programming approach
        var dp = try self.allocator.alloc([]f64, data1.len + 1);
        defer self.allocator.free(dp);

        for (0..dp.len) |i| {
            dp[i] = try self.allocator.alloc(f64, data2.len + 1);
        }
        defer for (dp) |row| self.allocator.free(row);

        // Initialize base cases
        for (0..dp.len) |i| {
            for (0..dp[i].len) |j| {
                dp[i][j] = 0.0;
            }
        }

        // Fill DP table for k=1
        if (self.k == 1) {
            for (1..dp.len) |i| {
                for (1..dp[i].len) |j| {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                    if (data1[i - 1] == data2[j - 1]) {
                        dp[i][j] += self.lambda * self.lambda;
                    }
                    dp[i][j] *= self.lambda;
                }
            }
            return dp[data1.len][data2.len];
        }

        // For k > 1, use recursive approach (simplified)
        return self.computeRecursiveBytes(data1, data2, self.k);
    }

    /// Compute subsequence kernel for token strings
    fn computeTokenSubsequence(self: Self, tokens1: []const u64, tokens2: []const u64) f64 {
        if (self.k == 1) {
            var count: f64 = 0.0;
            for (tokens1) |t1| {
                for (tokens2) |t2| {
                    if (t1 == t2) {
                        count += self.lambda * self.lambda;
                    }
                }
            }
            return count;
        }

        // For k > 1, use recursive approach with tokens
        return self.computeRecursiveTokens(tokens1, tokens2, self.k);
    }

    /// Compute subsequence kernel for bit strings
    fn computeBitSubsequence(self: Self, bits1: []const u8, bits2: []const u8, len1: usize, len2: usize) f64 {
        if (self.k == 1) {
            var count: f64 = 0.0;
            for (0..len1) |i| {
                for (0..len2) |j| {
                    const bit1 = self.getBit(bits1, i);
                    const bit2 = self.getBit(bits2, j);
                    if (bit1 == bit2) {
                        count += self.lambda * self.lambda;
                    }
                }
            }
            return count;
        }

        // For k > 1, use recursive approach with bits
        return self.computeRecursiveBits(bits1, bits2, len1, len2, self.k);
    }

    /// Get bit at position from bit array
    fn getBit(self: Self, bits: []const u8, pos: usize) u1 {
        _ = self;
        const byte_idx = pos / 8;
        const bit_idx = @as(u3, @intCast(pos % 8));
        return @intCast((bits[byte_idx] >> (7 - bit_idx)) & 1);
    }

    /// Recursive computation for byte subsequence kernel
    fn computeRecursiveBytes(self: Self, s1: []const u8, s2: []const u8, k: usize) f64 {
        if (k == 0) return 1.0;
        if (s1.len < k or s2.len < k) return 0.0;

        var sum: f64 = 0.0;

        // Find matching characters
        for (0..s1.len) |i| {
            for (0..s2.len) |j| {
                if (s1[i] == s2[j]) {
                    const penalty = std.math.pow(f64, self.lambda, @as(f64, @floatFromInt(i + j + 2)));
                    sum += penalty * self.computeRecursiveBytes(s1[i + 1 ..], s2[j + 1 ..], k - 1);
                }
            }
        }

        return sum;
    }

    /// Recursive computation for token subsequence kernel
    fn computeRecursiveTokens(self: Self, s1: []const u64, s2: []const u64, k: usize) f64 {
        if (k == 0) return 1.0;
        if (s1.len < k or s2.len < k) return 0.0;

        var sum: f64 = 0.0;

        // Find matching tokens
        for (0..s1.len) |i| {
            for (0..s2.len) |j| {
                if (s1[i] == s2[j]) {
                    const penalty = std.math.pow(f64, self.lambda, @as(f64, @floatFromInt(i + j + 2)));
                    sum += penalty * self.computeRecursiveTokens(s1[i + 1 ..], s2[j + 1 ..], k - 1);
                }
            }
        }

        return sum;
    }

    /// Recursive computation for bit subsequence kernel
    fn computeRecursiveBits(self: Self, bits1: []const u8, bits2: []const u8, len1: usize, len2: usize, k: usize) f64 {
        if (k == 0) return 1.0;
        if (len1 < k or len2 < k) return 0.0;

        var sum: f64 = 0.0;

        // Find matching bits
        for (0..len1) |i| {
            for (0..len2) |j| {
                const bit1 = self.getBit(bits1, i);
                const bit2 = self.getBit(bits2, j);
                if (bit1 == bit2) {
                    const penalty = std.math.pow(f64, self.lambda, @as(f64, @floatFromInt(i + j + 2)));
                    // For recursive call, we need to adjust the bit arrays and lengths
                    const remaining_len1 = len1 - i - 1;
                    const remaining_len2 = len2 - j - 1;
                    sum += penalty * self.computeRecursiveBits(bits1, bits2, remaining_len1, remaining_len2, k - 1);
                }
            }
        }

        return sum;
    }
};

/// Weighted degree kernel implementation
pub const WeightedDegreeKernel = struct {
    degree: usize,
    weights: []f64,
    allocator: Allocator,

    const Self = @This();

    /// Initialize weighted degree kernel
    pub fn init(allocator: Allocator, degree: usize) !Self {
        var weights = try allocator.alloc(f64, degree);

        // Initialize with exponentially decreasing weights
        for (0..degree) |i| {
            weights[i] = 1.0 / @as(f64, @floatFromInt(i + 1));
        }

        return Self{
            .allocator = allocator,
            .degree = degree,
            .weights = weights,
        };
    }

    /// Cleanup
    pub fn deinit(self: Self) void {
        self.allocator.free(self.weights);
    }

    /// Set custom weights
    pub fn setWeights(self: *Self, weights: []const f64) !void {
        if (weights.len != self.degree) return error.InvalidWeightCount;
        @memcpy(self.weights, weights);
    }

    /// Compute weighted degree kernel between two strings
    pub fn compute(self: Self, str1: StringValue, str2: StringValue) f64 {
        if (str1.len != str2.len) return 0.0;
        if (str1.len == 0) return 0.0;

        // Ensure both strings have the same type
        if (@intFromEnum(str1.data) != @intFromEnum(str2.data)) return 0.0;

        switch (str1.data) {
            .byte => |bytes1| {
                const bytes2 = str2.data.byte;
                return self.computeByteWeightedDegree(bytes1, bytes2);
            },
            .token => |tokens1| {
                const tokens2 = str2.data.token;
                return self.computeTokenWeightedDegree(tokens1, tokens2);
            },
            .bit => |bits1| {
                const bits2 = str2.data.bit;
                return self.computeBitWeightedDegree(bits1, bits2, str1.len);
            },
        }
    }

    /// Compute weighted degree kernel for byte strings
    fn computeByteWeightedDegree(self: Self, data1: []const u8, data2: []const u8) f64 {
        var kernel_value: f64 = 0.0;

        // For each position in the strings
        for (0..data1.len) |pos| {
            // For each degree (k-mer length)
            for (1..@min(self.degree + 1, data1.len - pos + 1)) |k| {
                if (pos + k > data1.len) break;

                // Check if k-mers match at this position
                var match = true;
                for (0..k) |offset| {
                    if (data1[pos + offset] != data2[pos + offset]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    kernel_value += self.weights[k - 1];
                }
            }
        }

        return kernel_value;
    }

    /// Compute weighted degree kernel for token strings
    fn computeTokenWeightedDegree(self: Self, tokens1: []const u64, tokens2: []const u64) f64 {
        var kernel_value: f64 = 0.0;

        // For each position in the strings
        for (0..tokens1.len) |pos| {
            // For each degree (k-mer length)
            for (1..@min(self.degree + 1, tokens1.len - pos + 1)) |k| {
                if (pos + k > tokens1.len) break;

                // Check if k-mers match at this position
                var match = true;
                for (0..k) |offset| {
                    if (tokens1[pos + offset] != tokens2[pos + offset]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    kernel_value += self.weights[k - 1];
                }
            }
        }

        return kernel_value;
    }

    /// Compute weighted degree kernel for bit strings
    fn computeBitWeightedDegree(self: Self, bits1: []const u8, bits2: []const u8, bit_len: usize) f64 {
        var kernel_value: f64 = 0.0;

        // For each position in the bit strings
        for (0..bit_len) |pos| {
            // For each degree (k-mer length)
            for (1..@min(self.degree + 1, bit_len - pos + 1)) |k| {
                if (pos + k > bit_len) break;

                // Check if k-mers match at this position
                var match = true;
                for (0..k) |offset| {
                    const bit1 = self.getBitAt(bits1, pos + offset);
                    const bit2 = self.getBitAt(bits2, pos + offset);
                    if (bit1 != bit2) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    kernel_value += self.weights[k - 1];
                }
            }
        }

        return kernel_value;
    }

    /// Get bit at position from bit array
    fn getBitAt(self: Self, bits: []const u8, pos: usize) u1 {
        _ = self;
        const byte_idx = pos / 8;
        const bit_idx = @as(u3, @intCast(pos % 8));
        return @intCast((bits[byte_idx] >> (7 - bit_idx)) & 1);
    }

    /// Compute normalized weighted degree kernel
    pub fn computeNormalized(self: Self, str1: StringValue, str2: StringValue) f64 {
        const k12 = self.compute(str1, str2);
        const k11 = self.compute(str1, str1);
        const k22 = self.compute(str2, str2);

        const denominator = std.math.sqrt(k11 * k22);
        if (denominator == 0.0) return 0.0;

        return k12 / denominator;
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
test "SpectrumKernel basic functionality" {
    const allocator = testing.allocator;

    var kernel = SpectrumKernel.init(allocator, 2);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abc");
    defer str2.deinit();

    var str3 = try StringValue.fromBytes(allocator, "def");
    defer str3.deinit();

    // Identical strings should have high similarity
    const sim1 = try kernel.compute(str1, str2);
    try testing.expect(sim1 > 0.0);

    // Different strings should have lower similarity
    const sim2 = try kernel.compute(str1, str3);
    try testing.expect(sim2 == 0.0);
}

test "SpectrumKernel k-gram extraction" {
    const allocator = testing.allocator;

    var kernel = SpectrumKernel.init(allocator, 3);

    var str1 = try StringValue.fromBytes(allocator, "abcab");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abcde");
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Should have "abc" in common
}

test "SpectrumKernel edge cases" {
    const allocator = testing.allocator;

    var kernel = SpectrumKernel.init(allocator, 0);

    var str1 = try StringValue.fromBytes(allocator, "test");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "test");
    defer str2.deinit();

    // k=0 should return 0
    const sim1 = try kernel.compute(str1, str2);
    try testing.expect(sim1 == 0.0);

    // Strings shorter than k should return 0
    kernel.k = 10;
    const sim2 = try kernel.compute(str1, str2);
    try testing.expect(sim2 == 0.0);
}

test "SubsequenceKernel basic functionality" {
    const allocator = testing.allocator;

    var kernel = SubsequenceKernel.init(allocator, 1, 0.5);

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abc");
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0);
}

test "SubsequenceKernel edge cases" {
    const allocator = testing.allocator;

    var kernel = SubsequenceKernel.init(allocator, 0, 0.5);

    var str1 = try StringValue.fromBytes(allocator, "test");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "test");
    defer str2.deinit();

    // k=0 should return 0
    const sim1 = try kernel.compute(str1, str2);
    try testing.expect(sim1 == 0.0);

    // Invalid lambda should return 0
    kernel.lambda = 0.0;
    kernel.k = 1;
    const sim2 = try kernel.compute(str1, str2);
    try testing.expect(sim2 == 0.0);

    kernel.lambda = 1.0;
    const sim3 = try kernel.compute(str1, str2);
    try testing.expect(sim3 == 0.0);
}

test "WeightedDegreeKernel basic functionality" {
    const allocator = testing.allocator;

    var kernel = try WeightedDegreeKernel.init(allocator, 3);
    defer kernel.deinit();

    var str1 = try StringValue.fromBytes(allocator, "abcd");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abcd");
    defer str2.deinit();

    var str3 = try StringValue.fromBytes(allocator, "efgh");
    defer str3.deinit();

    // Identical strings
    const sim1 = kernel.compute(str1, str2);
    try testing.expect(sim1 > 0.0);

    // Different strings
    const sim2 = kernel.compute(str1, str3);
    try testing.expect(sim2 == 0.0);

    // Normalized version
    const norm_sim = kernel.computeNormalized(str1, str2);
    try testing.expect(norm_sim > 0.0 and norm_sim <= 1.0);
}

test "WeightedDegreeKernel custom weights" {
    const allocator = testing.allocator;

    var kernel = try WeightedDegreeKernel.init(allocator, 2);
    defer kernel.deinit();

    const custom_weights = [_]f64{ 0.8, 0.2 };
    try kernel.setWeights(&custom_weights);

    var str1 = try StringValue.fromBytes(allocator, "ab");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "ab");
    defer str2.deinit();

    const sim = kernel.compute(str1, str2);
    // Should be 0.8 (1-gram at pos 0) + 0.8 (1-gram at pos 1) + 0.2 (2-gram) = 1.8
    try testing.expect(sim > 1.7 and sim < 1.9);
}

test "WeightedDegreeKernel different lengths" {
    const allocator = testing.allocator;

    var kernel = try WeightedDegreeKernel.init(allocator, 2);
    defer kernel.deinit();

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "ab");
    defer str2.deinit();

    // Different length strings should return 0
    const sim = kernel.compute(str1, str2);
    try testing.expect(sim == 0.0);
}

test "SpectrumKernel with token strings" {
    const allocator = testing.allocator;

    var kernel = SpectrumKernel.init(allocator, 2);

    const tokens1 = [_]Symbol{ 1, 2, 3, 4 };
    const tokens2 = [_]Symbol{ 1, 2, 5, 6 };

    var str1 = try StringValue.fromTokens(allocator, &tokens1);
    defer str1.deinit();

    var str2 = try StringValue.fromTokens(allocator, &tokens2);
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Should have common 2-gram [1,2]
}

test "SpectrumKernel with bit strings" {
    const allocator = testing.allocator;

    var kernel = SpectrumKernel.init(allocator, 3);

    const bits1 = [_]u8{0b11010000}; // 8 bits: 11010000
    const bits2 = [_]u8{0b11011000}; // 8 bits: 11011000

    var str1 = try StringValue.fromBits(allocator, &bits1, 8);
    defer str1.deinit();

    var str2 = try StringValue.fromBits(allocator, &bits2, 8);
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Should have common bit patterns
}

test "SubsequenceKernel with token strings" {
    const allocator = testing.allocator;

    var kernel = SubsequenceKernel.init(allocator, 1, 0.5);

    const tokens1 = [_]Symbol{ 1, 2, 3 };
    const tokens2 = [_]Symbol{ 1, 3, 2 };

    var str1 = try StringValue.fromTokens(allocator, &tokens1);
    defer str1.deinit();

    var str2 = try StringValue.fromTokens(allocator, &tokens2);
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Should have matching tokens
}

test "SubsequenceKernel with bit strings" {
    const allocator = testing.allocator;

    var kernel = SubsequenceKernel.init(allocator, 1, 0.5);

    const bits1 = [_]u8{0b11000000}; // 4 bits: 1100
    const bits2 = [_]u8{0b10100000}; // 4 bits: 1010

    var str1 = try StringValue.fromBits(allocator, &bits1, 4);
    defer str1.deinit();

    var str2 = try StringValue.fromBits(allocator, &bits2, 4);
    defer str2.deinit();

    const sim = try kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Should have matching bits
}

test "WeightedDegreeKernel with token strings" {
    const allocator = testing.allocator;

    var kernel = try WeightedDegreeKernel.init(allocator, 2);
    defer kernel.deinit();

    const tokens1 = [_]Symbol{ 1, 2, 3 };
    const tokens2 = [_]Symbol{ 1, 2, 3 };

    var str1 = try StringValue.fromTokens(allocator, &tokens1);
    defer str1.deinit();

    var str2 = try StringValue.fromTokens(allocator, &tokens2);
    defer str2.deinit();

    const sim = kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Identical token strings should have high similarity
}

test "WeightedDegreeKernel with bit strings" {
    const allocator = testing.allocator;

    var kernel = try WeightedDegreeKernel.init(allocator, 3);
    defer kernel.deinit();

    const bits1 = [_]u8{0b11010000}; // 5 bits: 11010
    const bits2 = [_]u8{0b11010000}; // 5 bits: 11010

    var str1 = try StringValue.fromBits(allocator, &bits1, 5);
    defer str1.deinit();

    var str2 = try StringValue.fromBits(allocator, &bits2, 5);
    defer str2.deinit();

    const sim = kernel.compute(str1, str2);
    try testing.expect(sim > 0.0); // Identical bit strings should have high similarity
}

test "Kernels with mixed string types should return 0" {
    const allocator = testing.allocator;

    var spectrum_kernel = SpectrumKernel.init(allocator, 2);
    var subsequence_kernel = SubsequenceKernel.init(allocator, 1, 0.5);
    var weighted_kernel = try WeightedDegreeKernel.init(allocator, 2);
    defer weighted_kernel.deinit();

    var byte_str = try StringValue.fromBytes(allocator, "abc");
    defer byte_str.deinit();

    const tokens = [_]Symbol{ 1, 2, 3 };
    var token_str = try StringValue.fromTokens(allocator, &tokens);
    defer token_str.deinit();

    // Mixed types should return 0
    const spectrum_sim = try spectrum_kernel.compute(byte_str, token_str);
    try testing.expect(spectrum_sim == 0.0);

    const subsequence_sim = try subsequence_kernel.compute(byte_str, token_str);
    try testing.expect(subsequence_sim == 0.0);

    const weighted_sim = weighted_kernel.compute(byte_str, token_str);
    try testing.expect(weighted_sim == 0.0);
}
