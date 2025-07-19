const std = @import("std");
const testing = std.testing;

pub const StringValue = @import("string.zig").StringValue;
pub const StringType = @import("string.zig").StringType;

test {
    testing.refAllDecls(@This());
}
