const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const StringValue = @import("string.zig").StringValue;

/// Range specification for matrix operations
pub const Range = struct {
    start: usize,
    end: usize,

    pub fn length(self: Range) usize {
        return self.end - self.start;
    }
};

/// Matrix for storing similarity computations between strings
pub const Matrix = struct {
    /// String labels for rows/columns
    labels: ?[]const f32 = null,
    /// Source identifiers for strings
    sources: ?[]const []const u8 = null,
    /// Number of strings
    num: usize,

    /// Similarity values storage
    values: []f32,
    /// Memory allocator
    allocator: Allocator,

    /// Column range for computation
    col_range: Range,
    /// Row range for computation
    row_range: Range,
    /// Whether to use triangular storage optimization
    triangular: bool,

    const Self = @This();

    /// Initialize matrix for given strings
    pub fn init(allocator: Allocator, strings: []const StringValue) !Self {
        const num = strings.len;
        const size = if (num > 0) (num * (num + 1)) / 2 else 0;

        const values = try allocator.alloc(f32, size);
        @memset(values, 0.0);

        return Self{
            .num = num,
            .values = values,
            .allocator = allocator,
            .col_range = Range{ .start = 0, .end = num },
            .row_range = Range{ .start = 0, .end = num },
            .triangular = true,
        };
    }

    /// Set column range for computation
    pub fn setColRange(self: *Self, start: usize, end: usize) !void {
        if (start > end) return error.InvalidRange;
        if (end > self.num) return error.RangeOutOfBounds;
        self.col_range = Range{ .start = start, .end = end };
    }

    /// Set row range for computation
    pub fn setRowRange(self: *Self, start: usize, end: usize) !void {
        if (start > end) return error.InvalidRange;
        if (end > self.num) return error.RangeOutOfBounds;
        self.row_range = Range{ .start = start, .end = end };
    }

    /// Enable or disable triangular storage
    pub fn setTriangular(self: *Self, triangular: bool) !void {
        if (self.triangular == triangular) return;

        const new_size = if (triangular)
            (self.num * (self.num + 1)) / 2
        else
            self.num * self.num;

        const new_values = try self.allocator.alloc(f32, new_size);
        @memset(new_values, 0.0);

        // Copy existing values if any
        if (self.values.len > 0) {
            const copy_size = @min(self.values.len, new_values.len);
            @memcpy(new_values[0..copy_size], self.values[0..copy_size]);
        }

        self.allocator.free(self.values);
        self.values = new_values;
        self.triangular = triangular;
    }

    /// Get matrix element at (row, col)
    pub fn get(self: Self, row: usize, col: usize) !f32 {
        if (row >= self.num or col >= self.num) return error.IndexOutOfBounds;

        if (self.triangular) {
            const i = @min(row, col);
            const j = @max(row, col);
            const idx = (j * (j + 1)) / 2 + i;
            return self.values[idx];
        } else {
            const idx = row * self.num + col;
            return self.values[idx];
        }
    }

    /// Set matrix element at (row, col)
    pub fn set(self: Self, row: usize, col: usize, value: f32) !void {
        if (row >= self.num or col >= self.num) return error.IndexOutOfBounds;

        if (self.triangular) {
            const i = @min(row, col);
            const j = @max(row, col);
            const idx = (j * (j + 1)) / 2 + i;
            self.values[idx] = value;
        } else {
            const idx = row * self.num + col;
            self.values[idx] = value;
        }
    }

    /// Compute similarity matrix using provided measure function
    pub fn compute(self: Self, strings: []const StringValue, measure: fn (StringValue, StringValue) f64) !void {
        for (self.row_range.start..self.row_range.end) |i| {
            for (self.col_range.start..self.col_range.end) |j| {
                if (self.triangular and i > j) continue;

                const similarity = measure(strings[i], strings[j]);
                try self.set(i, j, @floatCast(similarity));
            }
        }
    }

    /// Get total number of computations needed
    pub fn getComputations(self: Self) usize {
        const rows = self.row_range.length();
        const cols = self.col_range.length();

        if (self.triangular) {
            // Count upper triangle elements within range
            var count: usize = 0;
            for (self.row_range.start..self.row_range.end) |i| {
                for (self.col_range.start..self.col_range.end) |j| {
                    if (i <= j) count += 1;
                }
            }
            return count;
        } else {
            return rows * cols;
        }
    }

    /// Clean up allocated memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.values);
        if (self.sources) |sources| {
            for (sources) |source| {
                self.allocator.free(source);
            }
            self.allocator.free(sources);
        }
        if (self.labels) |labels| {
            self.allocator.free(labels);
        }
        self.* = undefined;
    }
};

// Tests
test "Matrix initialization" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "abc"),
        try StringValue.fromBytes(allocator, "def"),
        try StringValue.fromBytes(allocator, "ghi"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    try testing.expect(matrix.num == 3);
    try testing.expect(matrix.triangular == true);
    try testing.expect(matrix.values.len == 6); // (3 * 4) / 2
    try testing.expect(matrix.col_range.start == 0);
    try testing.expect(matrix.col_range.end == 3);
    try testing.expect(matrix.row_range.start == 0);
    try testing.expect(matrix.row_range.end == 3);
}

test "Matrix get/set operations" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test setting and getting values
    try matrix.set(0, 0, 1.0);
    try matrix.set(0, 1, 0.5);
    try matrix.set(1, 0, 0.5); // Should map to same location as (0,1) in triangular
    try matrix.set(1, 1, 1.0);

    try testing.expect(try matrix.get(0, 0) == 1.0);
    try testing.expect(try matrix.get(0, 1) == 0.5);
    try testing.expect(try matrix.get(1, 0) == 0.5);
    try testing.expect(try matrix.get(1, 1) == 1.0);
}

test "Matrix range operations" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
        try StringValue.fromBytes(allocator, "c"),
        try StringValue.fromBytes(allocator, "d"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test setting ranges
    try matrix.setRowRange(1, 3);
    try matrix.setColRange(0, 2);

    try testing.expect(matrix.row_range.start == 1);
    try testing.expect(matrix.row_range.end == 3);
    try testing.expect(matrix.col_range.start == 0);
    try testing.expect(matrix.col_range.end == 2);
    try testing.expect(matrix.row_range.length() == 2);
    try testing.expect(matrix.col_range.length() == 2);
}

test "Matrix triangular storage toggle" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Initially triangular
    try testing.expect(matrix.triangular == true);
    try testing.expect(matrix.values.len == 3); // (2 * 3) / 2

    // Switch to full matrix
    try matrix.setTriangular(false);
    try testing.expect(matrix.triangular == false);
    try testing.expect(matrix.values.len == 4); // 2 * 2

    // Switch back to triangular
    try matrix.setTriangular(true);
    try testing.expect(matrix.triangular == true);
    try testing.expect(matrix.values.len == 3); // (2 * 3) / 2
}

test "Matrix computations count" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
        try StringValue.fromBytes(allocator, "c"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Full triangular matrix
    try testing.expect(matrix.getComputations() == 6); // 3*3 but triangular

    // Test with ranges
    try matrix.setRowRange(0, 2);
    try matrix.setColRange(0, 2);
    try testing.expect(matrix.getComputations() == 3); // Only upper triangle of 2x2

    // Test full matrix
    try matrix.setTriangular(false);
    try testing.expect(matrix.getComputations() == 4); // Full 2x2 matrix
}

test "Matrix range validation errors" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
        try StringValue.fromBytes(allocator, "c"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test invalid range (start > end)
    try testing.expectError(error.InvalidRange, matrix.setRowRange(2, 1));
    try testing.expectError(error.InvalidRange, matrix.setColRange(3, 0));

    // Test out of bounds range
    try testing.expectError(error.RangeOutOfBounds, matrix.setRowRange(0, 4)); // matrix.num = 3
    try testing.expectError(error.RangeOutOfBounds, matrix.setColRange(1, 5));

    // Test boundary conditions - these should work
    try matrix.setRowRange(0, 3); // Exactly matrix.num
    try matrix.setColRange(0, 3);
    try matrix.setRowRange(3, 3); // Empty range is valid
    try matrix.setColRange(2, 2); // Empty range is valid
}

test "Matrix index validation errors" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test out of bounds access for get
    try testing.expectError(error.IndexOutOfBounds, matrix.get(2, 0)); // row >= matrix.num
    try testing.expectError(error.IndexOutOfBounds, matrix.get(0, 2)); // col >= matrix.num
    try testing.expectError(error.IndexOutOfBounds, matrix.get(2, 2)); // both out of bounds

    // Test out of bounds access for set
    try testing.expectError(error.IndexOutOfBounds, matrix.set(2, 0, 1.0)); // row >= matrix.num
    try testing.expectError(error.IndexOutOfBounds, matrix.set(0, 2, 1.0)); // col >= matrix.num
    try testing.expectError(error.IndexOutOfBounds, matrix.set(2, 2, 1.0)); // both out of bounds

    // Test boundary conditions - these should work
    try matrix.set(0, 0, 1.0);
    try matrix.set(1, 1, 2.0);
    try testing.expect(try matrix.get(0, 0) == 1.0);
    try testing.expect(try matrix.get(1, 1) == 2.0);
}

test "Matrix validation with empty matrix" {
    const allocator = testing.allocator;

    const strings = [_]StringValue{};
    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test operations on empty matrix
    try testing.expectError(error.IndexOutOfBounds, matrix.get(0, 0));
    try testing.expectError(error.IndexOutOfBounds, matrix.set(0, 0, 1.0));

    // Range operations should work with empty ranges
    try matrix.setRowRange(0, 0);
    try matrix.setColRange(0, 0);

    // But any non-zero range should fail
    try testing.expectError(error.RangeOutOfBounds, matrix.setRowRange(0, 1));
    try testing.expectError(error.RangeOutOfBounds, matrix.setColRange(0, 1));
}

test "Matrix validation with different storage modes" {
    const allocator = testing.allocator;

    var strings = [_]StringValue{
        try StringValue.fromBytes(allocator, "a"),
        try StringValue.fromBytes(allocator, "b"),
        try StringValue.fromBytes(allocator, "c"),
    };
    defer for (&strings) |*s| s.deinit();

    var matrix = try Matrix.init(allocator, &strings);
    defer matrix.deinit();

    // Test validation works in triangular mode
    try matrix.set(0, 1, 0.5);
    try testing.expect(try matrix.get(0, 1) == 0.5);
    try testing.expectError(error.IndexOutOfBounds, matrix.get(3, 0));

    // Switch to full mode and test validation still works
    try matrix.setTriangular(false);
    try matrix.set(1, 0, 0.7);
    try testing.expect(try matrix.get(1, 0) == 0.7);
    try testing.expectError(error.IndexOutOfBounds, matrix.set(3, 0, 1.0));

    // Switch back to triangular and verify validation still works
    try matrix.setTriangular(true);
    try testing.expectError(error.IndexOutOfBounds, matrix.get(0, 3));
}
