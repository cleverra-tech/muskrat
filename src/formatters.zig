const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;
const Symbol = @import("string.zig").Symbol;
const Matrix = @import("matrix.zig").Matrix;

/// Metadata for output formatting
pub const OutputMetadata = struct {
    computation_time_ms: ?f64 = null,
    string_count: usize = 0,
    matrix_size: ?struct { rows: usize, cols: usize } = null,
    distance_measure: ?[]const u8 = null,
    timestamp: ?i64 = null,

    const Self = @This();

    /// Set computation time in milliseconds
    pub fn setComputationTime(self: *Self, time_ms: f64) void {
        self.computation_time_ms = time_ms;
    }

    /// Set current timestamp
    pub fn setTimestamp(self: *Self) void {
        self.timestamp = std.time.milliTimestamp();
    }
};

/// Generic output formatter interface
pub const OutputFormatter = struct {
    allocator: Allocator,
    include_metadata: bool,
    precision: u8,

    const Self = @This();

    /// Initialize output formatter
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .include_metadata = true,
            .precision = 6,
        };
    }

    /// Set metadata inclusion
    pub fn setIncludeMetadata(self: *Self, include: bool) void {
        self.include_metadata = include;
    }

    /// Set floating point precision
    pub fn setPrecision(self: *Self, precision: u8) void {
        self.precision = precision;
    }
};

/// Text formatter for human-readable output
pub const TextFormatter = struct {
    formatter: OutputFormatter,
    field_separator: []const u8,
    show_labels: bool,

    const Self = @This();

    /// Initialize text formatter
    pub fn init(allocator: Allocator) Self {
        return Self{
            .formatter = OutputFormatter.init(allocator),
            .field_separator = "\t",
            .show_labels = true,
        };
    }

    /// Set field separator
    pub fn setFieldSeparator(self: *Self, separator: []const u8) void {
        self.field_separator = separator;
    }

    /// Set label display
    pub fn setShowLabels(self: *Self, show: bool) void {
        self.show_labels = show;
    }

    /// Format matrix to text
    pub fn formatMatrix(self: Self, matrix: *const Matrix, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();

        // Write metadata if requested
        if (self.formatter.include_metadata and metadata != null) {
            try self.writeMetadata(writer, metadata.?);
            try writer.writeAll("\n");
        }

        // Write header row if labels are shown
        if (self.show_labels) {
            try writer.writeAll("String");
            for (0..strings.len) |j| {
                try writer.print("{s}{}", .{ self.field_separator, j });
            }
            try writer.writeAll("\n");
        }

        // Write matrix data
        for (matrix.row_range.start..matrix.row_range.end) |i| {
            // Write row label
            if (self.show_labels) {
                try writer.print("{}", .{i});
            }

            // Write row values
            for (matrix.col_range.start..matrix.col_range.end) |j| {
                const value = matrix.get(i, j);
                if (self.show_labels or j > 0) {
                    try writer.writeAll(self.field_separator);
                }
                try writer.print("{d:.3}", .{value});
            }
            try writer.writeAll("\n");
        }

        return output.toOwnedSlice();
    }

    /// Format string list to text
    pub fn formatStrings(self: Self, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();

        // Write metadata if requested
        if (self.formatter.include_metadata and metadata != null) {
            try self.writeMetadata(writer, metadata.?);
            try writer.writeAll("\n");
        }

        // Write strings
        for (strings, 0..) |string, i| {
            if (self.show_labels) {
                try writer.print("{}{s}", .{ i, self.field_separator });
            }

            switch (string.data) {
                .byte => |bytes| try writer.writeAll(bytes),
                .token => |tokens| {
                    // Format tokens as comma-separated list: [1,2,3]
                    try writer.writeAll("[");
                    for (tokens, 0..) |token, j| {
                        if (j > 0) try writer.writeAll(",");
                        try writer.print("{d}", .{token});
                    }
                    try writer.writeAll("]");
                },
                .bit => |bits| {
                    // Format bits as hexadecimal: 0x1a2b3c
                    try writer.writeAll("0x");
                    for (bits) |byte| {
                        try writer.print("{x:0>2}", .{byte});
                    }
                },
            }

            if (string.label) |label| {
                try writer.print("{s}{d:.3}", .{ self.field_separator, label });
            }
            try writer.writeAll("\n");
        }

        return output.toOwnedSlice();
    }

    /// Write metadata to writer
    fn writeMetadata(self: Self, writer: anytype, metadata: OutputMetadata) !void {
        _ = self;
        try writer.writeAll("# Metadata\n");

        if (metadata.timestamp) |ts| {
            try writer.print("# Timestamp: {}\n", .{ts});
        }

        try writer.print("# String count: {}\n", .{metadata.string_count});

        if (metadata.matrix_size) |size| {
            try writer.print("# Matrix size: {}x{}\n", .{ size.rows, size.cols });
        }

        if (metadata.distance_measure) |measure| {
            try writer.print("# Distance measure: {s}\n", .{measure});
        }

        if (metadata.computation_time_ms) |time_ms| {
            try writer.print("# Computation time: {d:.3}ms\n", .{time_ms});
        }
    }
};

/// JSON formatter for structured output
pub const JsonFormatter = struct {
    formatter: OutputFormatter,
    pretty_print: bool,

    const Self = @This();

    /// Initialize JSON formatter
    pub fn init(allocator: Allocator) Self {
        return Self{
            .formatter = OutputFormatter.init(allocator),
            .pretty_print = true,
        };
    }

    /// Set pretty printing
    pub fn setPrettyPrint(self: *Self, pretty: bool) void {
        self.pretty_print = pretty;
    }

    /// Format matrix to JSON
    pub fn formatMatrix(self: Self, matrix: *const Matrix, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();
        _ = self.pretty_print; // Unused but preserved for future use

        // Build JSON structure
        try writer.writeAll("{\n");

        // Write metadata
        if (self.formatter.include_metadata and metadata != null) {
            try self.writeJsonMetadata(writer, metadata.?);
            try writer.writeAll(",\n");
        }

        // Write strings array
        try writer.writeAll("  \"strings\": [\n");
        for (strings, 0..) |string, i| {
            if (i > 0) try writer.writeAll(",\n");
            try writer.writeAll("    {\n");
            try writer.print("      \"index\": {},\n", .{i});
            try writer.print("      \"type\": \"{s}\",\n", .{@tagName(string.data)});
            try writer.writeAll("      \"content\": ");

            switch (string.data) {
                .byte => |bytes| try std.json.stringify(bytes, .{}, writer),
                .token => |tokens| {
                    // Serialize tokens as JSON array of numbers
                    try writer.writeAll("[");
                    for (tokens, 0..) |token, j| {
                        if (j > 0) try writer.writeAll(",");
                        try writer.print("{d}", .{token});
                    }
                    try writer.writeAll("]");
                },
                .bit => |bits| {
                    // Serialize bits as JSON array of bytes (for readability)
                    try writer.writeAll("[");
                    for (bits, 0..) |byte, j| {
                        if (j > 0) try writer.writeAll(",");
                        try writer.print("{d}", .{byte});
                    }
                    try writer.writeAll("]");
                },
            }

            if (string.label) |label| {
                try writer.print(",\n      \"label\": {d:.3}", .{label});
            }

            try writer.writeAll("\n    }");
        }
        try writer.writeAll("\n  ],\n");

        // Write matrix data
        try writer.writeAll("  \"matrix\": {\n");
        try writer.print("    \"rows\": {},\n", .{matrix.row_range.length()});
        try writer.print("    \"cols\": {},\n", .{matrix.col_range.length()});
        try writer.print("    \"triangular\": {},\n", .{matrix.triangular});
        try writer.writeAll("    \"data\": [\n");

        for (matrix.row_range.start..matrix.row_range.end) |i| {
            if (i > matrix.row_range.start) try writer.writeAll(",\n");
            try writer.writeAll("      [");

            for (matrix.col_range.start..matrix.col_range.end) |j| {
                if (j > matrix.col_range.start) try writer.writeAll(", ");
                const value = matrix.get(i, j);
                try writer.print("{d:.3}", .{value});
            }
            try writer.writeAll("]");
        }

        try writer.writeAll("\n    ]\n");
        try writer.writeAll("  }\n");
        try writer.writeAll("}\n");

        return output.toOwnedSlice();
    }

    /// Format string list to JSON
    pub fn formatStrings(self: Self, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();
        _ = self.pretty_print; // Unused but preserved for future use

        try writer.writeAll("{\n");

        // Write metadata
        if (self.formatter.include_metadata and metadata != null) {
            try self.writeJsonMetadata(writer, metadata.?);
            try writer.writeAll(",\n");
        }

        // Write strings
        try writer.writeAll("  \"strings\": [\n");
        for (strings, 0..) |string, i| {
            if (i > 0) try writer.writeAll(",\n");
            try writer.writeAll("    {\n");
            try writer.print("      \"index\": {},\n", .{i});

            try writer.print("      \"type\": \"{s}\",\n", .{@tagName(string.data)});
            try writer.writeAll("      \"content\": ");

            switch (string.data) {
                .byte => |bytes| try std.json.stringify(bytes, .{}, writer),
                .token => |tokens| {
                    // Serialize tokens as JSON array of numbers
                    try writer.writeAll("[");
                    for (tokens, 0..) |token, j| {
                        if (j > 0) try writer.writeAll(",");
                        try writer.print("{d}", .{token});
                    }
                    try writer.writeAll("]");
                },
                .bit => |bits| {
                    // Serialize bits as JSON array of bytes (for readability)
                    try writer.writeAll("[");
                    for (bits, 0..) |byte, j| {
                        if (j > 0) try writer.writeAll(",");
                        try writer.print("{d}", .{byte});
                    }
                    try writer.writeAll("]");
                },
            }

            if (string.label) |label| {
                try writer.print(",\n      \"label\": {d:.3}", .{label});
            }

            try writer.writeAll("\n    }");
        }
        try writer.writeAll("\n  ]\n");
        try writer.writeAll("}\n");

        return output.toOwnedSlice();
    }

    /// Write JSON metadata
    fn writeJsonMetadata(self: Self, writer: anytype, metadata: OutputMetadata) !void {
        _ = self;

        try writer.writeAll("  \"metadata\": {\n");

        if (metadata.timestamp) |ts| {
            try writer.print("    \"timestamp\": {},\n", .{ts});
        }

        try writer.print("    \"string_count\": {},\n", .{metadata.string_count});

        if (metadata.matrix_size) |size| {
            try writer.writeAll("    \"matrix_size\": {\"rows\": ");
            try writer.print("{}", .{size.rows});
            try writer.writeAll(", \"cols\": ");
            try writer.print("{}", .{size.cols});
            try writer.writeAll("},\n");
        }

        if (metadata.distance_measure) |measure| {
            try writer.writeAll("    \"distance_measure\": ");
            try std.json.stringify(measure, .{}, writer);
            try writer.writeAll(",\n");
        }

        if (metadata.computation_time_ms) |time_ms| {
            try writer.print("    \"computation_time_ms\": {d:.3}\n", .{time_ms});
        }

        try writer.writeAll("  }");
    }
};

/// Binary formatter for compact output
pub const BinaryFormatter = struct {
    formatter: OutputFormatter,

    const Self = @This();

    /// Initialize binary formatter
    pub fn init(allocator: Allocator) Self {
        return Self{
            .formatter = OutputFormatter.init(allocator),
        };
    }

    /// Format matrix to binary
    pub fn formatMatrix(self: Self, matrix: *const Matrix, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();

        // Write header
        try writer.writeAll("MUSK"); // Magic number
        try writer.writeInt(u32, 1, .little); // Version

        // Write metadata
        const has_metadata: u8 = if (self.formatter.include_metadata and metadata != null) 1 else 0;
        try writer.writeInt(u8, has_metadata, .little);

        if (has_metadata == 1) {
            try self.writeBinaryMetadata(writer, metadata.?);
        }

        // Write strings count
        try writer.writeInt(u32, @intCast(strings.len), .little);

        // Write strings
        for (strings) |string| {
            // Write string type first
            try writer.writeInt(u8, @intFromEnum(string.data), .little);

            switch (string.data) {
                .byte => |bytes| {
                    try writer.writeInt(u32, @intCast(bytes.len), .little);
                    try writer.writeAll(bytes);
                },
                .token => |tokens| {
                    try writer.writeInt(u32, @intCast(tokens.len), .little);
                    for (tokens) |token| {
                        try writer.writeInt(u64, token, .little);
                    }
                },
                .bit => |bits| {
                    try writer.writeInt(u32, @intCast(string.len), .little); // bit count
                    try writer.writeInt(u32, @intCast(bits.len), .little); // byte count
                    try writer.writeAll(bits);
                },
            }

            // Write label
            const has_label: u8 = if (string.label != null) 1 else 0;
            try writer.writeInt(u8, has_label, .little);
            if (has_label == 1) {
                try writer.writeInt(u32, @bitCast(string.label.?), .little);
            }
        }

        // Write matrix dimensions
        try writer.writeInt(u32, @intCast(matrix.row_range.length()), .little);
        try writer.writeInt(u32, @intCast(matrix.col_range.length()), .little);
        try writer.writeInt(u8, if (matrix.triangular) 1 else 0, .little);

        // Write matrix data
        for (matrix.row_range.start..matrix.row_range.end) |i| {
            for (matrix.col_range.start..matrix.col_range.end) |j| {
                const value = matrix.get(i, j);
                try writer.writeInt(u32, @bitCast(value), .little);
            }
        }

        return output.toOwnedSlice();
    }

    /// Write binary metadata
    fn writeBinaryMetadata(self: Self, writer: anytype, metadata: OutputMetadata) !void {
        _ = self;

        // Write timestamp
        const has_timestamp: u8 = if (metadata.timestamp != null) 1 else 0;
        try writer.writeInt(u8, has_timestamp, .little);
        if (has_timestamp == 1) {
            try writer.writeInt(i64, metadata.timestamp.?, .little);
        }

        // Write string count
        try writer.writeInt(u32, @intCast(metadata.string_count), .little);

        // Write matrix size
        const has_matrix_size: u8 = if (metadata.matrix_size != null) 1 else 0;
        try writer.writeInt(u8, has_matrix_size, .little);
        if (has_matrix_size == 1) {
            try writer.writeInt(u32, @intCast(metadata.matrix_size.?.rows), .little);
            try writer.writeInt(u32, @intCast(metadata.matrix_size.?.cols), .little);
        }

        // Write distance measure
        const distance_measure = metadata.distance_measure orelse "";
        try writer.writeInt(u32, @intCast(distance_measure.len), .little);
        try writer.writeAll(distance_measure);

        // Write computation time
        const has_time: u8 = if (metadata.computation_time_ms != null) 1 else 0;
        try writer.writeInt(u8, has_time, .little);
        if (has_time == 1) {
            try writer.writeInt(u64, @bitCast(metadata.computation_time_ms.?), .little);
        }
    }
};

/// CSV formatter for spreadsheet compatibility
pub const CsvFormatter = struct {
    formatter: OutputFormatter,
    delimiter: u8,
    quote_char: u8,
    escape_quotes: bool,

    const Self = @This();

    /// Initialize CSV formatter
    pub fn init(allocator: Allocator) Self {
        return Self{
            .formatter = OutputFormatter.init(allocator),
            .delimiter = ',',
            .quote_char = '"',
            .escape_quotes = true,
        };
    }

    /// Set delimiter character
    pub fn setDelimiter(self: *Self, delimiter: u8) void {
        self.delimiter = delimiter;
    }

    /// Format matrix to CSV
    pub fn formatMatrix(self: Self, matrix: *const Matrix, strings: []const StringValue, metadata: ?OutputMetadata) ![]u8 {
        var output = std.ArrayList(u8).init(self.formatter.allocator);
        defer output.deinit();

        const writer = output.writer();

        // Write header row
        try writer.writeAll("String");
        for (0..strings.len) |j| {
            try writer.writeByte(self.delimiter);
            try writer.print("{}", .{j});
        }
        try writer.writeAll("\n");

        // Write matrix data
        for (matrix.row_range.start..matrix.row_range.end) |i| {
            try writer.print("{}", .{i});

            for (matrix.col_range.start..matrix.col_range.end) |j| {
                try writer.writeByte(self.delimiter);
                const value = matrix.get(i, j);
                try writer.print("{d:.3}", .{value});
            }
            try writer.writeAll("\n");
        }

        // Write metadata as comments if requested
        if (self.formatter.include_metadata and metadata != null) {
            try writer.writeAll("\n# Metadata\n");
            if (metadata.?.timestamp) |ts| {
                try writer.print("# Timestamp{c}{}\n", .{ self.delimiter, ts });
            }
            try writer.print("# String count{c}{}\n", .{ self.delimiter, metadata.?.string_count });
            if (metadata.?.computation_time_ms) |time_ms| {
                try writer.print("# Computation time (ms){c}{d:.3}\n", .{ self.delimiter, time_ms });
            }
        }

        return output.toOwnedSlice();
    }

    /// Escape CSV field if needed
    fn escapeField(self: Self, allocator: Allocator, field: []const u8) ![]u8 {
        var needs_quoting = false;

        // Check if field contains delimiter, quote, or newline
        for (field) |char| {
            if (char == self.delimiter or char == self.quote_char or char == '\n' or char == '\r') {
                needs_quoting = true;
                break;
            }
        }

        if (!needs_quoting) {
            return allocator.dupe(u8, field);
        }

        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        try result.append(self.quote_char);

        for (field) |char| {
            if (char == self.quote_char and self.escape_quotes) {
                try result.append(self.quote_char); // Double the quote
            }
            try result.append(char);
        }

        try result.append(self.quote_char);

        return result.toOwnedSlice();
    }
};

// Tests
test "TextFormatter basic functionality" {
    const allocator = testing.allocator;

    var formatter = TextFormatter.init(allocator);

    // Create test strings
    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "hello"),
        try StringValue.fromBytes(allocator, "world"),
    };
    defer for (&strings) |*s| s.deinit();

    // Create test metadata
    var metadata = OutputMetadata{
        .string_count = 2,
        .computation_time_ms = 123.45,
        .distance_measure = "hamming",
    };
    metadata.setTimestamp();

    const output = try formatter.formatStrings(&strings, metadata);
    defer allocator.free(output);

    // Check that output contains expected elements
    try testing.expect(std.mem.indexOf(u8, output, "hello") != null);
    try testing.expect(std.mem.indexOf(u8, output, "world") != null);
    try testing.expect(std.mem.indexOf(u8, output, "Metadata") != null);
}

test "JsonFormatter basic functionality" {
    const allocator = testing.allocator;

    var formatter = JsonFormatter.init(allocator);

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "test1"),
        try StringValue.fromBytes(allocator, "test2"),
    };
    defer for (&strings) |*s| s.deinit();

    const metadata = OutputMetadata{
        .string_count = 2,
        .computation_time_ms = 100.0,
    };

    const output = try formatter.formatStrings(&strings, metadata);
    defer allocator.free(output);

    // Verify JSON structure
    try testing.expect(std.mem.indexOf(u8, output, "\"strings\"") != null);
    try testing.expect(std.mem.indexOf(u8, output, "\"metadata\"") != null);
    try testing.expect(std.mem.indexOf(u8, output, "test1") != null);
}

test "BinaryFormatter basic functionality" {
    const allocator = testing.allocator;

    var formatter = BinaryFormatter.init(allocator);

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
    };
    defer for (&strings) |*s| s.deinit();

    // Create a simple 2x2 matrix for testing
    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    matrix.set(0, 0, 0.0);
    matrix.set(0, 1, 1.0);
    matrix.set(1, 0, 1.0);
    matrix.set(1, 1, 0.0);

    const metadata = OutputMetadata{
        .string_count = 2,
        .matrix_size = .{ .rows = 2, .cols = 2 },
    };

    const output = try formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(output);

    // Check magic number
    try testing.expect(output.len >= 4);
    try testing.expect(std.mem.eql(u8, output[0..4], "MUSK"));
}

test "BinaryFormatter with all string types" {
    const allocator = testing.allocator;

    var formatter = BinaryFormatter.init(allocator);

    // Test with byte, token, and bit string types
    const tokens = [_]Symbol{ 1, 2, 3 };
    const bits = [_]u8{0b10110100}; // Sample bit pattern

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "hello"),
        try StringValue.fromTokens(allocator, &tokens),
        try StringValue.fromBits(allocator, &bits, 8),
    };
    defer for (&strings) |*s| s.deinit();

    // Create a 3x3 matrix for testing
    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Set some test values
    matrix.set(0, 1, 0.5);
    matrix.set(1, 2, 0.8);
    matrix.set(0, 2, 0.3);

    const metadata = OutputMetadata{
        .string_count = 3,
        .matrix_size = .{ .rows = 3, .cols = 3 },
    };

    const output = try formatter.formatMatrix(&matrix, &strings, metadata);
    defer allocator.free(output);

    // Check magic number and that we have substantial output
    try testing.expect(output.len >= 4);
    try testing.expect(std.mem.eql(u8, output[0..4], "MUSK"));
    try testing.expect(output.len > 50); // Should have meaningful binary data
}

test "CsvFormatter basic functionality" {
    const allocator = testing.allocator;

    var formatter = CsvFormatter.init(allocator);

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "x"),
        try StringValue.fromBytes(allocator, "y"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    matrix.set(0, 0, 0.0);
    matrix.set(0, 1, 0.5);
    matrix.set(1, 0, 0.5);
    matrix.set(1, 1, 0.0);

    const output = try formatter.formatMatrix(&matrix, &strings, null);
    defer allocator.free(output);

    // Check CSV structure
    try testing.expect(std.mem.indexOf(u8, output, "String,0,1") != null);
    try testing.expect(std.mem.indexOf(u8, output, "0,0.000,0.500") != null);
}

test "OutputMetadata functionality" {
    var metadata = OutputMetadata{
        .string_count = 10,
    };

    metadata.setComputationTime(456.78);
    metadata.setTimestamp();

    try testing.expect(metadata.computation_time_ms.? == 456.78);
    try testing.expect(metadata.timestamp != null);
    try testing.expect(metadata.string_count == 10);
}

test "TextFormatter without metadata" {
    const allocator = testing.allocator;

    var formatter = TextFormatter.init(allocator);
    formatter.formatter.setIncludeMetadata(false);

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "simple"),
    };
    defer for (&strings) |*s| s.deinit();

    const output = try formatter.formatStrings(&strings, null);
    defer allocator.free(output);

    // Should not contain metadata
    try testing.expect(std.mem.indexOf(u8, output, "Metadata") == null);
    try testing.expect(std.mem.indexOf(u8, output, "simple") != null);
}
