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
            error.FileNotFound => {
                // File not found, return appropriate error
                return error.FileNotFound;
            },
            error.AccessDenied => {
                // Access denied to file, return appropriate error
                return error.AccessDenied;
            },
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
            error.FileNotFound => {
                // Directory not found, return appropriate error
                return error.DirectoryNotFound;
            },
            error.AccessDenied => {
                // Access denied to directory, return appropriate error
                return error.AccessDenied;
            },
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
                            error.FileNotFound => {
                                // File not found (possibly deleted during processing), skip it
                                continue;
                            },
                            error.AccessDenied => {
                                // Access denied to file, skip it
                                continue;
                            },
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

    /// Check if filename matches glob pattern
    fn matchesPattern(self: Self, filename: []const u8) bool {
        return globMatch(self.file_pattern, filename);
    }

    /// Glob pattern matching implementation
    fn globMatch(pattern: []const u8, text: []const u8) bool {
        return globMatchRecursive(pattern, 0, text, 0);
    }

    /// Recursive glob pattern matcher
    fn globMatchRecursive(pattern: []const u8, p_idx: usize, text: []const u8, t_idx: usize) bool {
        // If we've consumed both pattern and text, it's a match
        if (p_idx >= pattern.len and t_idx >= text.len) return true;

        // If pattern is consumed but text remains, no match (unless pattern ends with *)
        if (p_idx >= pattern.len) return false;

        // If text is consumed but pattern remains, check if remaining pattern is all *
        if (t_idx >= text.len) {
            for (p_idx..pattern.len) |i| {
                if (pattern[i] != '*') return false;
            }
            return true;
        }

        const p_char = pattern[p_idx];
        const t_char = text[t_idx];

        switch (p_char) {
            '*' => {
                // Try matching * with zero characters (skip the *)
                if (globMatchRecursive(pattern, p_idx + 1, text, t_idx)) return true;

                // Try matching * with one or more characters
                var i = t_idx;
                while (i < text.len) {
                    if (globMatchRecursive(pattern, p_idx + 1, text, i + 1)) return true;
                    i += 1;
                }
                return false;
            },
            '?' => {
                // ? matches any single character
                return globMatchRecursive(pattern, p_idx + 1, text, t_idx + 1);
            },
            '[' => {
                // Character class matching [abc], [a-z], [!abc]
                const close_bracket = std.mem.indexOfScalarPos(u8, pattern, p_idx + 1, ']') orelse return false;
                const char_set = pattern[p_idx + 1 .. close_bracket];

                if (char_set.len == 0) return false; // Empty character class

                const negate = char_set[0] == '!';
                const actual_set = if (negate) char_set[1..] else char_set;

                const matches = matchesCharacterClass(t_char, actual_set);
                const result = if (negate) !matches else matches;

                if (result) {
                    return globMatchRecursive(pattern, close_bracket + 1, text, t_idx + 1);
                } else {
                    return false;
                }
            },
            '\\' => {
                // Escape character - match the next character literally
                if (p_idx + 1 >= pattern.len) return false;
                const escaped_char = pattern[p_idx + 1];
                if (escaped_char == t_char) {
                    return globMatchRecursive(pattern, p_idx + 2, text, t_idx + 1);
                } else {
                    return false;
                }
            },
            else => {
                // Regular character - must match exactly
                if (p_char == t_char) {
                    return globMatchRecursive(pattern, p_idx + 1, text, t_idx + 1);
                } else {
                    return false;
                }
            },
        }
    }

    /// Check if character matches a character class like "abc" or "a-z"
    fn matchesCharacterClass(char: u8, char_set: []const u8) bool {
        var i: usize = 0;
        while (i < char_set.len) {
            // Check for range notation like "a-z"
            if (i + 2 < char_set.len and char_set[i + 1] == '-') {
                const start = char_set[i];
                const end = char_set[i + 2];
                if (char >= start and char <= end) return true;
                i += 3;
            } else {
                // Check for exact match
                if (char == char_set[i]) return true;
                i += 1;
            }
        }
        return false;
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

test "DirectoryReader glob pattern matching - wildcard" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Test * wildcard
    reader.setFilePattern("*.txt");
    try testing.expect(reader.matchesPattern("file.txt"));
    try testing.expect(reader.matchesPattern("test.txt"));
    try testing.expect(reader.matchesPattern("very_long_filename.txt"));
    try testing.expect(!reader.matchesPattern("file.log"));
    try testing.expect(!reader.matchesPattern("txtfile"));

    // Test multiple wildcards
    reader.setFilePattern("test*.log");
    try testing.expect(reader.matchesPattern("test.log"));
    try testing.expect(reader.matchesPattern("test123.log"));
    try testing.expect(reader.matchesPattern("testfile.log"));
    try testing.expect(!reader.matchesPattern("test.txt"));
    try testing.expect(!reader.matchesPattern("file.log"));

    // Test wildcard in middle
    reader.setFilePattern("file*name.txt");
    try testing.expect(reader.matchesPattern("filename.txt"));
    try testing.expect(reader.matchesPattern("file_long_name.txt"));
    try testing.expect(reader.matchesPattern("file123name.txt"));
    try testing.expect(!reader.matchesPattern("filename.log"));
    try testing.expect(!reader.matchesPattern("otherfile.txt"));
}

test "DirectoryReader glob pattern matching - question mark" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Test ? single character wildcard
    reader.setFilePattern("test?.txt");
    try testing.expect(reader.matchesPattern("test1.txt"));
    try testing.expect(reader.matchesPattern("testa.txt"));
    try testing.expect(reader.matchesPattern("test_.txt"));
    try testing.expect(!reader.matchesPattern("test.txt"));
    try testing.expect(!reader.matchesPattern("test12.txt"));
    try testing.expect(!reader.matchesPattern("test1.log"));

    // Test multiple ? wildcards
    reader.setFilePattern("???.txt");
    try testing.expect(reader.matchesPattern("abc.txt"));
    try testing.expect(reader.matchesPattern("123.txt"));
    try testing.expect(!reader.matchesPattern("ab.txt"));
    try testing.expect(!reader.matchesPattern("abcd.txt"));
}

test "DirectoryReader glob pattern matching - character classes" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Test character class [abc]
    reader.setFilePattern("test[abc].txt");
    try testing.expect(reader.matchesPattern("testa.txt"));
    try testing.expect(reader.matchesPattern("testb.txt"));
    try testing.expect(reader.matchesPattern("testc.txt"));
    try testing.expect(!reader.matchesPattern("testd.txt"));
    try testing.expect(!reader.matchesPattern("test.txt"));

    // Test character range [a-z]
    reader.setFilePattern("file[0-9].log");
    try testing.expect(reader.matchesPattern("file0.log"));
    try testing.expect(reader.matchesPattern("file5.log"));
    try testing.expect(reader.matchesPattern("file9.log"));
    try testing.expect(!reader.matchesPattern("filea.log"));
    try testing.expect(!reader.matchesPattern("file.log"));

    // Test negated character class [!abc]
    reader.setFilePattern("test[!0-9].txt");
    try testing.expect(reader.matchesPattern("testa.txt"));
    try testing.expect(reader.matchesPattern("testZ.txt"));
    try testing.expect(!reader.matchesPattern("test1.txt"));
    try testing.expect(!reader.matchesPattern("test9.txt"));
}

test "DirectoryReader glob pattern matching - escape sequences" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Test escaped special characters
    reader.setFilePattern("test\\*.txt");
    try testing.expect(reader.matchesPattern("test*.txt"));
    try testing.expect(!reader.matchesPattern("testfile.txt"));
    try testing.expect(!reader.matchesPattern("test.txt"));

    reader.setFilePattern("test\\?.log");
    try testing.expect(reader.matchesPattern("test?.log"));
    try testing.expect(!reader.matchesPattern("testa.log"));
    try testing.expect(!reader.matchesPattern("test.log"));
}

test "DirectoryReader glob pattern matching - complex patterns" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Complex pattern combining multiple features
    reader.setFilePattern("*test[0-9]*.txt");
    try testing.expect(reader.matchesPattern("mytest1file.txt"));
    try testing.expect(reader.matchesPattern("test5.txt"));
    try testing.expect(reader.matchesPattern("prefix_test9_suffix.txt"));
    try testing.expect(!reader.matchesPattern("testfile.txt"));
    try testing.expect(!reader.matchesPattern("mytest1file.log"));

    // Pattern with multiple character classes
    reader.setFilePattern("[Tt]est[0-9][a-z].log");
    try testing.expect(reader.matchesPattern("Test1a.log"));
    try testing.expect(reader.matchesPattern("test9z.log"));
    try testing.expect(!reader.matchesPattern("test1.log"));
    try testing.expect(!reader.matchesPattern("Test1A.log"));
    try testing.expect(!reader.matchesPattern("best1a.log"));
}

test "DirectoryReader glob pattern matching - edge cases" {
    const allocator = testing.allocator;

    var reader = DirectoryReader.init(allocator);
    defer reader.deinit();

    // Empty pattern
    reader.setFilePattern("");
    try testing.expect(reader.matchesPattern(""));
    try testing.expect(!reader.matchesPattern("file.txt"));

    // Pattern with only wildcards
    reader.setFilePattern("*");
    try testing.expect(reader.matchesPattern("anything.txt"));
    try testing.expect(reader.matchesPattern(""));
    try testing.expect(reader.matchesPattern("file"));

    reader.setFilePattern("***");
    try testing.expect(reader.matchesPattern("anything.txt"));
    try testing.expect(reader.matchesPattern(""));

    // Invalid character class (unclosed)
    reader.setFilePattern("test[abc.txt");
    try testing.expect(!reader.matchesPattern("testa.txt"));
    try testing.expect(!reader.matchesPattern("test[abc.txt"));
}

test "Error handling for nonexistent files" {
    const allocator = testing.allocator;

    var reader = FileReader.init(allocator);
    defer reader.deinit();

    // Should return error for nonexistent file
    const result = reader.readFromFile("nonexistent_file.txt");
    try testing.expectError(error.FileNotFound, result);
}
