const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Symbol type for tokenized strings
pub const Symbol = u64;

/// String representation types
pub const StringType = enum {
    byte, // Raw byte representation
    token, // Tokenized word representation
    bit, // Bit-level representation
};

/// Union for different string representations
pub const StringData = union(StringType) {
    byte: []const u8,
    token: []const Symbol,
    bit: []const u8, // Packed bit representation
};

/// Core string structure for similarity computations
pub const StringValue = struct {
    data: StringData,
    len: usize,
    source: ?[]const u8 = null, // Optional source identifier
    label: ?f32 = null, // Optional label for ML applications
    allocator: ?Allocator = null, // Track allocator for cleanup

    const Self = @This();

    /// Initialize from byte string
    pub fn fromBytes(allocator: Allocator, bytes: []const u8) !Self {
        const owned_bytes = try allocator.dupe(u8, bytes);
        return Self{
            .data = StringData{ .byte = owned_bytes },
            .len = bytes.len,
            .allocator = allocator,
        };
    }

    /// Initialize from tokens
    pub fn fromTokens(allocator: Allocator, tokens: []const Symbol) !Self {
        const owned_tokens = try allocator.dupe(Symbol, tokens);
        return Self{
            .data = StringData{ .token = owned_tokens },
            .len = tokens.len,
            .allocator = allocator,
        };
    }

    /// Initialize from bits (packed representation)
    pub fn fromBits(allocator: Allocator, bits: []const u8, bit_len: usize) !Self {
        const owned_bits = try allocator.dupe(u8, bits);
        return Self{
            .data = StringData{ .bit = owned_bits },
            .len = bit_len,
            .allocator = allocator,
        };
    }

    /// Get the string type
    pub fn getType(self: Self) StringType {
        return self.data;
    }

    /// Get character/symbol at position i
    pub fn get(self: Self, i: usize) u64 {
        std.debug.assert(i < self.len);

        switch (self.data) {
            .byte => |bytes| return bytes[i],
            .token => |tokens| return tokens[i],
            .bit => |bits| {
                const byte_idx = i / 8;
                const bit_idx = @as(u3, @intCast(i % 8));
                return (bits[byte_idx] >> (7 - bit_idx)) & 1;
            },
        }
    }

    /// Compare elements at positions i and j between two strings
    pub fn compare(self: Self, i: usize, other: Self, j: usize) i32 {
        std.debug.assert(self.getType() == other.getType());
        std.debug.assert(i < self.len and j < other.len);

        const a = self.get(i);
        const b = other.get(j);

        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }

    /// Set source identifier
    pub fn setSource(self: *Self, allocator: Allocator, source: []const u8) !void {
        if (self.source) |old_source| {
            if (self.allocator) |alloc| {
                alloc.free(old_source);
            }
        }
        self.source = try allocator.dupe(u8, source);
    }

    /// Set label for ML applications
    pub fn setLabel(self: *Self, label: f32) void {
        self.label = label;
    }

    /// Clean up allocated memory
    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| {
            switch (self.data) {
                .byte => |bytes| allocator.free(bytes),
                .token => |tokens| allocator.free(tokens),
                .bit => |bits| allocator.free(bits),
            }

            if (self.source) |source| {
                allocator.free(source);
            }
        }
        self.* = undefined;
    }
};

// Tests
test "StringValue byte initialization" {
    const allocator = testing.allocator;

    var str = try StringValue.fromBytes(allocator, "hello");
    defer str.deinit();

    try testing.expect(str.getType() == .byte);
    try testing.expect(str.len == 5);
    try testing.expect(str.get(0) == 'h');
    try testing.expect(str.get(4) == 'o');
}

test "StringValue token initialization" {
    const allocator = testing.allocator;

    const tokens = [_]Symbol{ 1, 2, 3, 4, 5 };
    var str = try StringValue.fromTokens(allocator, &tokens);
    defer str.deinit();

    try testing.expect(str.getType() == .token);
    try testing.expect(str.len == 5);
    try testing.expect(str.get(0) == 1);
    try testing.expect(str.get(4) == 5);
}

test "StringValue bit initialization" {
    const allocator = testing.allocator;

    // Bit pattern: 10110100 (8 bits)
    const bits = [_]u8{0b10110100};
    var str = try StringValue.fromBits(allocator, &bits, 8);
    defer str.deinit();

    try testing.expect(str.getType() == .bit);
    try testing.expect(str.len == 8);
    try testing.expect(str.get(0) == 1);
    try testing.expect(str.get(1) == 0);
    try testing.expect(str.get(2) == 1);
    try testing.expect(str.get(3) == 1);
    try testing.expect(str.get(7) == 0);
}

test "StringValue compare function" {
    const allocator = testing.allocator;

    var str1 = try StringValue.fromBytes(allocator, "abc");
    defer str1.deinit();

    var str2 = try StringValue.fromBytes(allocator, "abd");
    defer str2.deinit();

    try testing.expect(str1.compare(0, str2, 0) == 0); // 'a' == 'a'
    try testing.expect(str1.compare(2, str2, 2) < 0); // 'c' < 'd'
    try testing.expect(str2.compare(2, str1, 2) > 0); // 'd' > 'c'
}

test "StringValue source and label" {
    const allocator = testing.allocator;

    var str = try StringValue.fromBytes(allocator, "test");
    defer str.deinit();

    try str.setSource(allocator, "file.txt");
    str.setLabel(1.5);

    try testing.expect(std.mem.eql(u8, str.source.?, "file.txt"));
    try testing.expect(str.label.? == 1.5);
}
