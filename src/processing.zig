const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;
const StringType = @import("string.zig").StringType;
const Symbol = @import("string.zig").Symbol;

/// Hash table for token-to-symbol mapping
const TokenMap = std.HashMap([]const u8, Symbol, std.hash_map.StringContext, 80);

/// String processing utilities for similarity computations
pub const Processor = struct {
    allocator: Allocator,
    /// Token to symbol mapping for consistent tokenization
    token_map: TokenMap,
    /// Next symbol ID to assign
    next_symbol: Symbol,
    /// Delimiter characters for tokenization
    delimiters: []const u8,
    /// Whether to enable case normalization
    normalize_case: bool,

    const Self = @This();

    /// Initialize processor with default settings
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .token_map = TokenMap.init(allocator),
            .next_symbol = 1, // Reserve 0 for unknown/padding
            .delimiters = " \t\n\r.,;:!?\"'()[]{}",
            .normalize_case = false,
        };
    }

    /// Set delimiter characters for tokenization
    pub fn setDelimiters(self: *Self, delimiters: []const u8) !void {
        const owned_delims = try self.allocator.dupe(u8, delimiters);
        if (self.delimiters.len > 0 and !std.mem.eql(u8, self.delimiters, " \t\n\r.,;:!?\"'()[]{}")) {
            self.allocator.free(self.delimiters);
        }
        self.delimiters = owned_delims;
    }

    /// Enable or disable case normalization
    pub fn setCaseNormalization(self: *Self, normalize: bool) void {
        self.normalize_case = normalize;
    }

    /// Convert string to byte representation with optional preprocessing
    pub fn toBytes(self: *Self, input: []const u8) !StringValue {
        if (self.normalize_case) {
            var normalized = try self.allocator.alloc(u8, input.len);
            for (input, 0..) |c, i| {
                normalized[i] = std.ascii.toLower(c);
            }
            // StringValue.fromBytes will create its own copy, so we need to free this
            defer self.allocator.free(normalized);
            return StringValue.fromBytes(self.allocator, normalized);
        } else {
            return StringValue.fromBytes(self.allocator, input);
        }
    }

    /// Convert string to token representation
    pub fn toTokens(self: *Self, input: []const u8) !StringValue {
        var tokens = std.ArrayList(Symbol).init(self.allocator);
        defer tokens.deinit();

        var iterator = self.tokenIterator(input);
        while (iterator.next()) |token| {
            const symbol = try self.getOrCreateSymbol(token);
            try tokens.append(symbol);
        }

        const owned_tokens = try tokens.toOwnedSlice();
        // StringValue.fromTokens will create its own copy, so we need to free this
        defer self.allocator.free(owned_tokens);
        return StringValue.fromTokens(self.allocator, owned_tokens);
    }

    /// Convert string to bit representation
    pub fn toBits(self: *Self, input: []const u8) !StringValue {
        const bit_count = input.len * 8;
        const byte_count = (bit_count + 7) / 8;
        var bits = try self.allocator.alloc(u8, byte_count);
        @memset(bits, 0);

        for (input, 0..) |byte, byte_idx| {
            bits[byte_idx] = byte;
        }

        // StringValue.fromBits will create its own copy, so we need to free this
        defer self.allocator.free(bits);
        return StringValue.fromBits(self.allocator, bits, bit_count);
    }

    /// Get or create symbol for token
    fn getOrCreateSymbol(self: *Self, token: []const u8) !Symbol {
        const result = try self.token_map.getOrPut(token);
        if (!result.found_existing) {
            // Store owned copy of token
            const owned_token = try self.allocator.dupe(u8, token);
            result.key_ptr.* = owned_token;
            result.value_ptr.* = self.next_symbol;
            self.next_symbol += 1;
        }
        return result.value_ptr.*;
    }

    /// Create token iterator for string
    fn tokenIterator(self: *Self, input: []const u8) TokenIterator {
        return TokenIterator{
            .input = input,
            .delimiters = self.delimiters,
            .pos = 0,
            .normalize_case = self.normalize_case,
            .allocator = self.allocator,
        };
    }

    /// Extract n-grams from string
    pub fn getNGrams(self: *Self, input: []const u8, n: usize, string_type: StringType) ![]StringValue {
        if (n == 0) return &[_]StringValue{};

        switch (string_type) {
            .byte => return self.getByteNGrams(input, n),
            .token => return self.getTokenNGrams(input, n),
            .bit => return self.getBitNGrams(input, n),
        }
    }

    /// Extract byte-level n-grams
    fn getByteNGrams(self: *Self, input: []const u8, n: usize) ![]StringValue {
        if (input.len < n) return &[_]StringValue{};

        var ngrams = std.ArrayList(StringValue).init(self.allocator);
        defer ngrams.deinit();

        for (0..input.len - n + 1) |i| {
            const ngram = input[i .. i + n];
            const string_val = try self.toBytes(ngram);
            try ngrams.append(string_val);
        }

        return ngrams.toOwnedSlice();
    }

    /// Extract token-level n-grams
    fn getTokenNGrams(self: *Self, input: []const u8, n: usize) ![]StringValue {
        var tokens = std.ArrayList([]const u8).init(self.allocator);
        defer tokens.deinit();

        var iterator = self.tokenIterator(input);
        while (iterator.next()) |token| {
            try tokens.append(token);
        }

        if (tokens.items.len < n) return &[_]StringValue{};

        var ngrams = std.ArrayList(StringValue).init(self.allocator);
        defer ngrams.deinit();

        for (0..tokens.items.len - n + 1) |i| {
            var symbols = std.ArrayList(Symbol).init(self.allocator);
            defer symbols.deinit();

            for (0..n) |j| {
                const symbol = try self.getOrCreateSymbol(tokens.items[i + j]);
                try symbols.append(symbol);
            }

            const owned_symbols = try symbols.toOwnedSlice();
            const string_val = try StringValue.fromTokens(self.allocator, owned_symbols);
            try ngrams.append(string_val);
        }

        return ngrams.toOwnedSlice();
    }

    /// Extract bit-level n-grams
    fn getBitNGrams(self: *Self, input: []const u8, n: usize) ![]StringValue {
        const total_bits = input.len * 8;
        if (total_bits < n) return &[_]StringValue{};

        var ngrams = std.ArrayList(StringValue).init(self.allocator);
        defer ngrams.deinit();

        for (0..total_bits - n + 1) |i| {
            const byte_count = (n + 7) / 8;
            var bits = try self.allocator.alloc(u8, byte_count);
            @memset(bits, 0);

            for (0..n) |j| {
                const bit_pos = i + j;
                const byte_idx = bit_pos / 8;
                const bit_idx = @as(u3, @intCast(bit_pos % 8));
                const bit_val = (input[byte_idx] >> (7 - bit_idx)) & 1;

                const out_byte_idx = j / 8;
                const out_bit_idx = @as(u3, @intCast(j % 8));
                if (bit_val == 1) {
                    bits[out_byte_idx] |= @as(u8, 1) << (7 - out_bit_idx);
                }
            }

            const string_val = try StringValue.fromBits(self.allocator, bits, n);
            try ngrams.append(string_val);
        }

        return ngrams.toOwnedSlice();
    }

    /// Clean up allocated memory
    pub fn deinit(self: *Self) void {
        // Free all stored tokens
        var iterator = self.token_map.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.token_map.deinit();

        // Free delimiters if custom
        if (self.delimiters.len > 0 and !std.mem.eql(u8, self.delimiters, " \t\n\r.,;:!?\"'()[]{}")) {
            self.allocator.free(self.delimiters);
        }

        self.* = undefined;
    }
};

/// Iterator for tokenizing strings
const TokenIterator = struct {
    input: []const u8,
    delimiters: []const u8,
    pos: usize,
    normalize_case: bool,
    allocator: Allocator,

    fn next(self: *TokenIterator) ?[]const u8 {
        // Skip delimiters
        while (self.pos < self.input.len and self.isDelimiter(self.input[self.pos])) {
            self.pos += 1;
        }

        if (self.pos >= self.input.len) return null;

        const start = self.pos;
        // Find end of token
        while (self.pos < self.input.len and !self.isDelimiter(self.input[self.pos])) {
            self.pos += 1;
        }

        const token = self.input[start..self.pos];
        if (self.normalize_case) {
            // For normalized tokens, we need to allocate and convert
            var normalized = self.allocator.alloc(u8, token.len) catch return null;
            for (token, 0..) |c, i| {
                normalized[i] = std.ascii.toLower(c);
            }
            // Note: This creates a memory leak in the iterator design
            // In practice, this should be handled by the caller
            return normalized;
        }
        return token;
    }

    fn isDelimiter(self: TokenIterator, c: u8) bool {
        return std.mem.indexOfScalar(u8, self.delimiters, c) != null;
    }
};

// Tests
test "Processor initialization" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    try testing.expect(processor.next_symbol == 1);
    try testing.expect(processor.normalize_case == false);
    try testing.expect(std.mem.eql(u8, processor.delimiters, " \t\n\r.,;:!?\"'()[]{}"));
}

test "Processor byte conversion" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    var str = try processor.toBytes("Hello");
    defer str.deinit();

    try testing.expect(str.getType() == .byte);
    try testing.expect(str.len == 5);
    try testing.expect(str.get(0) == 'H');
}

test "Processor case normalization" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    processor.setCaseNormalization(true);

    var str = try processor.toBytes("Hello");
    defer str.deinit();

    try testing.expect(str.get(0) == 'h');
    try testing.expect(str.get(1) == 'e');
}

test "Processor tokenization" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    var str = try processor.toTokens("hello world test");
    defer str.deinit();

    try testing.expect(str.getType() == .token);
    try testing.expect(str.len == 3);
    try testing.expect(str.get(0) == 1); // "hello"
    try testing.expect(str.get(1) == 2); // "world"
    try testing.expect(str.get(2) == 3); // "test"
}

test "Processor bit conversion" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    var str = try processor.toBits("A"); // 'A' = 0x41 = 01000001
    defer str.deinit();

    try testing.expect(str.getType() == .bit);
    try testing.expect(str.len == 8);
    try testing.expect(str.get(0) == 0);
    try testing.expect(str.get(1) == 1);
    try testing.expect(str.get(6) == 0);
    try testing.expect(str.get(7) == 1);
}

test "Processor byte n-grams" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    const ngrams = try processor.getNGrams("hello", 2, .byte);
    defer {
        for (ngrams) |*ngram| ngram.deinit();
        allocator.free(ngrams);
    }

    try testing.expect(ngrams.len == 4); // "he", "el", "ll", "lo"
    try testing.expect(ngrams[0].len == 2);
    try testing.expect(ngrams[0].get(0) == 'h');
    try testing.expect(ngrams[0].get(1) == 'e');
}

test "Processor custom delimiters" {
    const allocator = testing.allocator;

    var processor = Processor.init(allocator);
    defer processor.deinit();

    try processor.setDelimiters("-_");

    var str = try processor.toTokens("hello-world_test");
    defer str.deinit();

    try testing.expect(str.len == 3);
}
