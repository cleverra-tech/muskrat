const std = @import("std");
const testing = std.testing;

pub const StringValue = @import("string.zig").StringValue;
pub const StringType = @import("string.zig").StringType;
pub const Matrix = @import("matrix.zig").Matrix;
pub const Range = @import("matrix.zig").Range;

test {
    testing.refAllDecls(@This());
}
