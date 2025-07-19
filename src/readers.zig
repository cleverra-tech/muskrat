const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;

/// Generic reader interface for string input
pub const StringReader = struct {
    allocator: Allocator,
    strings: std.ArrayList(StringValue),

    const Self = @This();

    /// Initialize string reader
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .strings = std.ArrayList(StringValue).init(allocator),
        };
    }

    /// Cleanup allocated strings
    pub fn deinit(self: *Self) void {
        for (self.strings.items) |*str| {
            str.deinit();
        }
        self.strings.deinit();
    }

    /// Add a string to the collection
    pub fn addString(self: *Self, string: StringValue) !void {
        try self.strings.append(string);
    }

    /// Get all strings
    pub fn getStrings(self: *const Self) []const StringValue {
        return self.strings.items;
    }

    /// Get string count
    pub fn count(self: *const Self) usize {
        return self.strings.items.len;
    }

    /// Clear all strings
    pub fn clear(self: *Self) void {
        for (self.strings.items) |*str| {
            str.deinit();
        }
        self.strings.clearAndFree();
    }
};

/// File reader for loading strings from text files
pub const FileReader = struct {
    reader: StringReader,
    separator: []const u8,
    max_line_length: usize,

    const Self = @This();

    /// Initialize file reader
    pub fn init(allocator: Allocator) Self {
        return Self{
            .reader = StringReader.init(allocator),
            .separator = "\n",
            .max_line_length = 64 * 1024, // 64KB default
        };
    }

    /// Cleanup
    pub fn deinit(self: *Self) void {
        self.reader.deinit();
    }

    /// Set line separator
    pub fn setSeparator(self: *Self, separator: []const u8) void {
        self.separator = separator;
    }

    /// Set maximum line length
    pub fn setMaxLineLength(self: *Self, max_length: usize) void {
        self.max_line_length = max_length;
    }

    /// Read strings from file
    pub fn readFromFile(self: *Self, file_path: []const u8) !void {
        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            error.AccessDenied => return error.AccessDenied,
            else => return err,
        };
        defer file.close();

        const contents = try file.readToEndAlloc(self.reader.allocator, 100 * 1024 * 1024); // 100MB limit
        defer self.reader.allocator.free(contents);

        try self.parseContent(contents);
    }

    /// Parse content with separator
    fn parseContent(self: *Self, content: []const u8) !void {
        var iterator = std.mem.splitSequence(u8, content, self.separator);

        while (iterator.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed.len == 0) continue; // Skip empty lines
            if (trimmed.len > self.max_line_length) continue; // Skip lines that are too long

            const string_value = try StringValue.fromBytes(self.reader.allocator, trimmed);
            try self.reader.addString(string_value);
        }
    }

    /// Get all strings
    pub fn getStrings(self: *const Self) []const StringValue {
        return self.reader.getStrings();
    }

    /// Get string count
    pub fn count(self: *const Self) usize {
        return self.reader.count();
    }
};

/// Standard input reader
pub const StdinReader = struct {
    reader: StringReader,
    separator: []const u8,
    max_line_length: usize,

    const Self = @This();

    /// Initialize stdin reader
    pub fn init(allocator: Allocator) Self {
        return Self{
            .reader = StringReader.init(allocator),
            .separator = "\n",
            .max_line_length = 64 * 1024,
        };
    }

    /// Cleanup
    pub fn deinit(self: *Self) void {
        self.reader.deinit();
    }

    /// Set line separator
    pub fn setSeparator(self: *Self, separator: []const u8) void {
        self.separator = separator;
    }

    /// Set maximum line length
    pub fn setMaxLineLength(self: *Self, max_length: usize) void {
        self.max_line_length = max_length;
    }

    /// Read strings from stdin
    pub fn readFromStdin(self: *Self) !void {
        const stdin = std.io.getStdIn().reader();

        const buffer = try self.reader.allocator.alloc(u8, self.max_line_length);
        defer self.reader.allocator.free(buffer);

        while (true) {
            if (stdin.readUntilDelimiterOrEof(buffer, '\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    // Skip this line if it's too long
                    continue;
                },
                else => return err,
            }) |line| {
                const trimmed = std.mem.trim(u8, line, " \t\r\n");
                if (trimmed.len == 0) continue;

                const string_value = try StringValue.fromBytes(self.reader.allocator, trimmed);
                try self.reader.addString(string_value);
            } else {
                break; // EOF
            }
        }
    }

    /// Get all strings
    pub fn getStrings(self: *const Self) []const StringValue {
        return self.reader.getStrings();
    }

    /// Get string count
    pub fn count(self: *const Self) usize {
        return self.reader.count();
    }
};

/// Directory reader for processing multiple files
pub const DirectoryReader = struct {
    reader: StringReader,
    file_pattern: []const u8,
    recursive: bool,

    const Self = @This();

    /// Initialize directory reader
    pub fn init(allocator: Allocator) Self {
        return Self{
            .reader = StringReader.init(allocator),
            .file_pattern = "*.txt",
            .recursive = false,
        };
    }

    /// Cleanup
    pub fn deinit(self: *Self) void {
        self.reader.deinit();
    }

    /// Set file pattern
    pub fn setFilePattern(self: *Self, pattern: []const u8) void {
        self.file_pattern = pattern;
    }

    /// Set recursive processing
    pub fn setRecursive(self: *Self, recursive: bool) void {
        self.recursive = recursive;
    }

    /// Read strings from directory
    pub fn readFromDirectory(self: *Self, dir_path: []const u8) !void {
        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return error.DirectoryNotFound,
            error.AccessDenied => return error.AccessDenied,
            else => return err,
        };
        defer dir.close();

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            switch (entry.kind) {
                .file => {
                    if (self.matchesPattern(entry.name)) {
                        const file_path = try std.fmt.allocPrint(self.reader.allocator, "{s}/{s}", .{ dir_path, entry.name });
                        defer self.reader.allocator.free(file_path);

                        var file_reader = FileReader.init(self.reader.allocator);
                        defer file_reader.deinit();

                        file_reader.readFromFile(file_path) catch |err| switch (err) {
                            error.FileNotFound, error.AccessDenied => continue, // Skip inaccessible files
                            else => return err,
                        };

                        // Add all strings from the file
                        for (file_reader.getStrings()) |str| {
                            const copied_str = try StringValue.fromBytes(self.reader.allocator, switch (str.data) {
                                .byte => |bytes| bytes,
                                else => continue, // Skip non-byte strings
                            });
                            try self.reader.addString(copied_str);
                        }
                    }
                },
                .directory => {
                    if (self.recursive and !std.mem.eql(u8, entry.name, ".") and !std.mem.eql(u8, entry.name, "..")) {
                        const sub_path = try std.fmt.allocPrint(self.reader.allocator, "{s}/{s}", .{ dir_path, entry.name });
                        defer self.reader.allocator.free(sub_path);
                        try self.readFromDirectory(sub_path);
                    }
                },
                else => continue,
            }
        }
    }

    /// Check if filename matches pattern (simplified glob matching)
    fn matchesPattern(self: Self, filename: []const u8) bool {
        // Simple pattern matching - for now just check extension
        if (std.mem.startsWith(u8, self.file_pattern, "*")) {
            const extension = self.file_pattern[1..]; // Remove *
            return std.mem.endsWith(u8, filename, extension);
        } else {
            return std.mem.eql(u8, filename, self.file_pattern);
        }
    }

    /// Get all strings
    pub fn getStrings(self: *const Self) []const StringValue {
        return self.reader.getStrings();
    }

    /// Get string count
    pub fn count(self: *const Self) usize {
        return self.reader.count();
    }
};

/// Memory reader for in-memory string collections
pub const MemoryReader = struct {
    reader: StringReader,

    const Self = @This();

    /// Initialize memory reader
    pub fn init(allocator: Allocator) Self {
        return Self{
            .reader = StringReader.init(allocator),
        };
    }

    /// Cleanup
    pub fn deinit(self: *Self) void {
        self.reader.deinit();
    }

    /// Add string from slice
    pub fn addFromSlice(self: *Self, string: []const u8) !void {
        const string_value = try StringValue.fromBytes(self.reader.allocator, string);
        try self.reader.addString(string_value);
    }

    /// Add multiple strings from slice array
    pub fn addFromSlices(self: *Self, strings: []const []const u8) !void {
        for (strings) |string| {
            try self.addFromSlice(string);
        }
    }

    /// Parse delimited content
    pub fn parseDelimited(self: *Self, content: []const u8, delimiter: []const u8) !void {
        var iterator = std.mem.splitSequence(u8, content, delimiter);

        while (iterator.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\r\n");
            if (trimmed.len > 0) {
                try self.addFromSlice(trimmed);
            }
        }
    }

    /// Get all strings
    pub fn getStrings(self: *const Self) []const StringValue {
        return self.reader.getStrings();
    }

    /// Get string count
    pub fn count(self: *const Self) usize {
        return self.reader.count();
    }
};

// Tests
test "StringReader basic functionality" {
    const allocator = testing.allocator;

    var reader = StringReader.init(allocator);
    defer reader.deinit();

    // Add some strings
    const str1 = try StringValue.fromBytes(allocator, "hello");
    const str2 = try StringValue.fromBytes(allocator, "world");

    try reader.addString(str1);
    try reader.addString(str2);

    // Check count
    try testing.expect(reader.count() == 2);

    // Check strings
    const strings = reader.getStrings();
    try testing.expect(strings.len == 2);

    // Check content (accessing byte data)
    const data1 = switch (strings[0].data) {
        .byte => |bytes| bytes,
        else => unreachable,
    };
    const data2 = switch (strings[1].data) {
        .byte => |bytes| bytes,
        else => unreachable,
    };

    try testing.expect(std.mem.eql(u8, data1, "hello"));
    try testing.expect(std.mem.eql(u8, data2, "world"));
}

test "MemoryReader functionality" {
    const allocator = testing.allocator;

    var reader = MemoryReader.init(allocator);
    defer reader.deinit();

    // Add strings
    try reader.addFromSlice("test1");
    try reader.addFromSlice("test2");

    try testing.expect(reader.count() == 2);

    // Add multiple at once
    const strings = [_][]const u8{ "batch1", "batch2", "batch3" };
    try reader.addFromSlices(&strings);

    try testing.expect(reader.count() == 5);
}

test "MemoryReader delimited parsing" {
    const allocator = testing.allocator;

    var reader = MemoryReader.init(allocator);
    defer reader.deinit();

    const content = "apple,banana,cherry,date";
    try reader.parseDelimited(content, ",");

    try testing.expect(reader.count() == 4);

    const strings = reader.getStrings();
    const data0 = switch (strings[0].data) {
        .byte => |bytes| bytes,
        else => unreachable,
    };
    try testing.expect(std.mem.eql(u8, data0, "apple"));
}

test "FileReader simulation" {
    const allocator = testing.allocator;

    // Create a temporary file for testing
    const test_content = "line1\nline2\nline3\n";
    const temp_file = "test_input.txt";

    // Write test file
    {
        const file = try std.fs.cwd().createFile(temp_file, .{});
        defer file.close();
        try file.writeAll(test_content);
    }
    defer std.fs.cwd().deleteFile(temp_file) catch {};

    var reader = FileReader.init(allocator);
    defer reader.deinit();

    try reader.readFromFile(temp_file);

    try testing.expect(reader.count() == 3);

    const strings = reader.getStrings();
    const data0 = switch (strings[0].data) {
        .byte => |bytes| bytes,
        else => unreachable,
    };
    try testing.expect(std.mem.eql(u8, data0, "line1"));
}

test "FileReader empty lines handling" {
    const allocator = testing.allocator;

    const test_content = "line1\n\nline2\n   \nline3";
    const temp_file = "test_empty_lines.txt";

    {
        const file = try std.fs.cwd().createFile(temp_file, .{});
        defer file.close();
        try file.writeAll(test_content);
    }
    defer std.fs.cwd().deleteFile(temp_file) catch {};

    var reader = FileReader.init(allocator);
    defer reader.deinit();

    try reader.readFromFile(temp_file);

    // Should skip empty lines and whitespace-only lines
    try testing.expect(reader.count() == 3);
}

test "FileReader custom separator" {
    const allocator = testing.allocator;

    const test_content = "item1|item2|item3";
    const temp_file = "test_separator.txt";

    {
        const file = try std.fs.cwd().createFile(temp_file, .{});
        defer file.close();
        try file.writeAll(test_content);
    }
    defer std.fs.cwd().deleteFile(temp_file) catch {};

    var reader = FileReader.init(allocator);
    defer reader.deinit();

    reader.setSeparator("|");
    try reader.readFromFile(temp_file);

    try testing.expect(reader.count() == 3);

    const strings = reader.getStrings();
    const data1 = switch (strings[1].data) {
        .byte => |bytes| bytes,
        else => unreachable,
    };
    try testing.expect(std.mem.eql(u8, data1, "item2"));
}

test "DirectoryReader pattern matching" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Test pattern matching
    reader.setFilePattern("*.txt");
    try testing.expect(reader.matchesPattern("test.txt"));
    try testing.expect(!reader.matchesPattern("test.log"));

    reader.setFilePattern("specific.txt");
    try testing.expect(reader.matchesPattern("specific.txt"));
    try testing.expect(!reader.matchesPattern("other.txt"));
}

test "Error handling for nonexistent files" {
    const allocator = testing.allocator;

    var reader = FileReader.init(allocator);
    defer reader.deinit();

    // Should return error for nonexistent file
    const result = reader.readFromFile("nonexistent_file.txt");
    try testing.expectError(error.FileNotFound, result);
}
