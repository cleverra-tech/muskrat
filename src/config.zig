const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Configuration for string similarity computation
pub const Config = struct {
    /// String processing configuration
    string: StringConfig = .{},
    /// Matrix computation configuration
    matrix: MatrixConfig = .{},
    /// Distance measure configuration
    distance: DistanceConfig = .{},
    /// Parallel computation configuration
    parallel: ParallelConfig = .{},
    /// Output configuration
    output: OutputConfig = .{},

    const Self = @This();

    /// Load configuration from JSON file
    pub fn fromJsonFile(allocator: Allocator, path: []const u8) !Self {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                // Configuration file not found, using default configuration
                return Self{};
            },
            error.AccessDenied => {
                // Access denied to configuration file
                return err;
            },
            else => return err,
        };
        defer file.close();

        const contents = try file.readToEndAlloc(allocator, 1024 * 1024);
        defer allocator.free(contents);

        return fromJson(allocator, contents) catch |err| {
            // Failed to parse configuration file, propagate error
            return err;
        };
    }

    /// Load configuration from JSON string
    pub fn fromJson(allocator: Allocator, json_str: []const u8) !Self {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch |err| switch (err) {
            error.SyntaxError => {
                // Invalid JSON syntax in configuration, using default configuration
                return Self{};
            },
            else => return err,
        };
        defer parsed.deinit();

        return parseJsonValue(parsed.value);
    }

    /// Save configuration to JSON file
    pub fn toJsonFile(self: Self, allocator: Allocator, path: []const u8) !void {
        const json_str = try self.toJson(allocator);
        defer allocator.free(json_str);

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        try file.writeAll(json_str);
    }

    /// Convert configuration to JSON string
    pub fn toJson(self: Self, allocator: Allocator) ![]u8 {
        var string = std.ArrayList(u8).init(allocator);
        defer string.deinit();

        try std.json.stringify(self, .{ .whitespace = .indent_2 }, string.writer());
        return string.toOwnedSlice();
    }

    /// Parse configuration from JSON value
    fn parseJsonValue(value: std.json.Value) !Self {
        var config = Self{};

        if (value != .object) return config;
        const obj = value.object;

        if (obj.get("string")) |string_val| {
            config.string = try parseStringConfig(string_val);
        }

        if (obj.get("matrix")) |matrix_val| {
            config.matrix = try parseMatrixConfig(matrix_val);
        }

        if (obj.get("distance")) |distance_val| {
            config.distance = try parseDistanceConfig(distance_val);
        }

        if (obj.get("parallel")) |parallel_val| {
            config.parallel = try parseParallelConfig(parallel_val);
        }

        if (obj.get("output")) |output_val| {
            config.output = try parseOutputConfig(output_val);
        }

        return config;
    }
};

/// String processing configuration
pub const StringConfig = struct {
    /// Type of string representation
    type: StringType = .byte,
    /// Delimiter characters for tokenization
    delimiters: []const u8 = " \t\n\r",
    /// Whether to normalize case
    normalize_case: bool = false,
    /// Maximum string length
    max_length: usize = 1024 * 1024,

    /// String representation type
    pub const StringType = enum {
        byte,
        token,
        bit,
    };
};

/// Matrix computation configuration
pub const MatrixConfig = struct {
    /// Use triangular storage optimization
    triangular: bool = true,
    /// Preallocate matrix memory
    preallocate: bool = true,
    /// Maximum matrix size
    max_size: usize = 10000,
};

/// Distance measure configuration
pub const DistanceConfig = struct {
    /// Primary distance measure
    measure: Measure = .hamming,
    /// Normalization method
    normalization: Normalization = .none,
    /// Cost parameters for edit distances
    costs: EditCosts = .{},

    /// Distance measure types
    pub const Measure = enum {
        hamming,
        levenshtein,
        jaro,
        jaro_winkler,
        jaccard,
        dice,
        simpson,
    };

    /// Normalization methods
    pub const Normalization = enum {
        none,
        length,
        max_length,
        min_length,
    };

    /// Edit distance cost parameters
    pub const EditCosts = struct {
        insertion: f32 = 1.0,
        deletion: f32 = 1.0,
        substitution: f32 = 1.0,
    };
};

/// Parallel computation configuration
pub const ParallelConfig = struct {
    /// Enable parallel computation
    enabled: bool = true,
    /// Number of threads (0 = auto-detect)
    thread_count: usize = 0,
    /// Minimum work per thread
    min_work_per_thread: usize = 100,
    /// Maximum threads
    max_threads: usize = 16,
    /// Load balancing strategy
    load_balancing: LoadBalancing = .rows,

    /// Load balancing strategies
    pub const LoadBalancing = enum {
        rows,
        work_queue,
    };
};

/// Output configuration
pub const OutputConfig = struct {
    /// Output format
    format: Format = .text,
    /// Include timing information
    include_timing: bool = false,
    /// Include metadata
    include_metadata: bool = true,
    /// Precision for floating point values
    precision: u8 = 6,

    /// Output format types
    pub const Format = enum {
        text,
        json,
        binary,
        csv,
    };
};

// JSON parsing helpers
fn parseStringConfig(value: std.json.Value) !StringConfig {
    var config = StringConfig{};
    if (value != .object) return config;
    const obj = value.object;

    if (obj.get("type")) |type_val| {
        if (type_val == .string) {
            const type_str = type_val.string;
            if (std.mem.eql(u8, type_str, "byte")) config.type = .byte;
            if (std.mem.eql(u8, type_str, "token")) config.type = .token;
            if (std.mem.eql(u8, type_str, "bit")) config.type = .bit;
        }
    }

    if (obj.get("normalize_case")) |val| {
        if (val == .bool) config.normalize_case = val.bool;
    }

    if (obj.get("max_length")) |val| {
        if (val == .integer) config.max_length = @intCast(val.integer);
    }

    return config;
}

fn parseMatrixConfig(value: std.json.Value) !MatrixConfig {
    var config = MatrixConfig{};
    if (value != .object) return config;
    const obj = value.object;

    if (obj.get("triangular")) |val| {
        if (val == .bool) config.triangular = val.bool;
    }

    if (obj.get("preallocate")) |val| {
        if (val == .bool) config.preallocate = val.bool;
    }

    if (obj.get("max_size")) |val| {
        if (val == .integer) config.max_size = @intCast(val.integer);
    }

    return config;
}

fn parseDistanceConfig(value: std.json.Value) !DistanceConfig {
    var config = DistanceConfig{};
    if (value != .object) return config;
    const obj = value.object;

    if (obj.get("measure")) |val| {
        if (val == .string) {
            const measure_str = val.string;
            if (std.mem.eql(u8, measure_str, "hamming")) config.measure = .hamming;
            if (std.mem.eql(u8, measure_str, "levenshtein")) config.measure = .levenshtein;
            if (std.mem.eql(u8, measure_str, "jaro")) config.measure = .jaro;
            if (std.mem.eql(u8, measure_str, "jaro_winkler")) config.measure = .jaro_winkler;
            if (std.mem.eql(u8, measure_str, "jaccard")) config.measure = .jaccard;
            if (std.mem.eql(u8, measure_str, "dice")) config.measure = .dice;
            if (std.mem.eql(u8, measure_str, "simpson")) config.measure = .simpson;
        }
    }

    if (obj.get("normalization")) |val| {
        if (val == .string) {
            const norm_str = val.string;
            if (std.mem.eql(u8, norm_str, "none")) config.normalization = .none;
            if (std.mem.eql(u8, norm_str, "length")) config.normalization = .length;
            if (std.mem.eql(u8, norm_str, "max_length")) config.normalization = .max_length;
            if (std.mem.eql(u8, norm_str, "min_length")) config.normalization = .min_length;
        }
    }

    return config;
}

fn parseParallelConfig(value: std.json.Value) !ParallelConfig {
    var config = ParallelConfig{};
    if (value != .object) return config;
    const obj = value.object;

    if (obj.get("enabled")) |val| {
        if (val == .bool) config.enabled = val.bool;
    }

    if (obj.get("thread_count")) |val| {
        if (val == .integer) config.thread_count = @intCast(val.integer);
    }

    if (obj.get("min_work_per_thread")) |val| {
        if (val == .integer) config.min_work_per_thread = @intCast(val.integer);
    }

    if (obj.get("max_threads")) |val| {
        if (val == .integer) config.max_threads = @intCast(val.integer);
    }

    return config;
}

fn parseOutputConfig(value: std.json.Value) !OutputConfig {
    var config = OutputConfig{};
    if (value != .object) return config;
    const obj = value.object;

    if (obj.get("format")) |val| {
        if (val == .string) {
            const format_str = val.string;
            if (std.mem.eql(u8, format_str, "text")) config.format = .text;
            if (std.mem.eql(u8, format_str, "json")) config.format = .json;
            if (std.mem.eql(u8, format_str, "binary")) config.format = .binary;
            if (std.mem.eql(u8, format_str, "csv")) config.format = .csv;
        }
    }

    if (obj.get("include_timing")) |val| {
        if (val == .bool) config.include_timing = val.bool;
    }

    if (obj.get("include_metadata")) |val| {
        if (val == .bool) config.include_metadata = val.bool;
    }

    if (obj.get("precision")) |val| {
        if (val == .integer) config.precision = @intCast(val.integer);
    }

    return config;
}

// Tests
test "Config default values" {
    const config = Config{};

    try testing.expect(config.string.type == .byte);
    try testing.expect(config.matrix.triangular == true);
    try testing.expect(config.distance.measure == .hamming);
    try testing.expect(config.parallel.enabled == true);
    try testing.expect(config.output.format == .text);
}

test "Config from empty JSON" {
    const allocator = testing.allocator;

    const config = try Config.fromJson(allocator, "{}");

    try testing.expect(config.string.type == .byte);
    try testing.expect(config.matrix.triangular == true);
}

test "Config from complete JSON" {
    const allocator = testing.allocator;

    const json_str =
        \\{
        \\  "string": {
        \\    "type": "token",
        \\    "normalize_case": true,
        \\    "max_length": 2048
        \\  },
        \\  "matrix": {
        \\    "triangular": false,
        \\    "preallocate": false,
        \\    "max_size": 5000
        \\  },
        \\  "distance": {
        \\    "measure": "levenshtein",
        \\    "normalization": "length"
        \\  },
        \\  "parallel": {
        \\    "enabled": false,
        \\    "thread_count": 4,
        \\    "max_threads": 8
        \\  },
        \\  "output": {
        \\    "format": "json",
        \\    "include_timing": true,
        \\    "precision": 4
        \\  }
        \\}
    ;

    const config = try Config.fromJson(allocator, json_str);

    try testing.expect(config.string.type == .token);
    try testing.expect(config.string.normalize_case == true);
    try testing.expect(config.string.max_length == 2048);

    try testing.expect(config.matrix.triangular == false);
    try testing.expect(config.matrix.preallocate == false);
    try testing.expect(config.matrix.max_size == 5000);

    try testing.expect(config.distance.measure == .levenshtein);
    try testing.expect(config.distance.normalization == .length);

    try testing.expect(config.parallel.enabled == false);
    try testing.expect(config.parallel.thread_count == 4);
    try testing.expect(config.parallel.max_threads == 8);

    try testing.expect(config.output.format == .json);
    try testing.expect(config.output.include_timing == true);
    try testing.expect(config.output.precision == 4);
}

test "Config to JSON and back" {
    const allocator = testing.allocator;

    var original = Config{};
    original.string.type = .token;
    original.string.normalize_case = true;
    original.distance.measure = .levenshtein;
    original.parallel.thread_count = 4;
    original.output.format = .json;

    const json_str = try original.toJson(allocator);
    defer allocator.free(json_str);

    const parsed = try Config.fromJson(allocator, json_str);

    try testing.expect(parsed.string.type == .token);
    try testing.expect(parsed.string.normalize_case == true);
    try testing.expect(parsed.distance.measure == .levenshtein);
    try testing.expect(parsed.parallel.thread_count == 4);
    try testing.expect(parsed.output.format == .json);
}

test "Config invalid JSON handling" {
    const allocator = testing.allocator;

    // Invalid JSON should return default config
    const config = try Config.fromJson(allocator, "invalid json");
    try testing.expect(config.string.type == .byte);
}

test "Config file operations" {
    const allocator = testing.allocator;

    var config = Config{};
    config.string.type = .token;
    config.distance.measure = .levenshtein;

    // Write to file
    const temp_path = "test_config.json";
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    try config.toJsonFile(allocator, temp_path);

    // Read from file
    const loaded = try Config.fromJsonFile(allocator, temp_path);

    try testing.expect(loaded.string.type == .token);
    try testing.expect(loaded.distance.measure == .levenshtein);
}

test "Config nonexistent file handling" {
    const allocator = testing.allocator;

    // Nonexistent file should return default config
    const config = try Config.fromJsonFile(allocator, "nonexistent.json");
    try testing.expect(config.string.type == .byte);
}
