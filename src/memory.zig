const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Object pool for reusing allocated memory blocks
pub fn ObjectPool(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            data: T,
            next: ?*Node,
        };

        allocator: Allocator,
        free_list: ?*Node,
        pool_size: usize,
        created_count: usize,
        reused_count: usize,

        /// Initialize object pool
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .free_list = null,
                .pool_size = 0,
                .created_count = 0,
                .reused_count = 0,
            };
        }

        /// Cleanup all pooled objects
        pub fn deinit(self: *Self) void {
            var current = self.free_list;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.free_list = null;
            self.pool_size = 0;
        }

        /// Get an object from the pool or create new one
        pub fn acquire(self: *Self) !*T {
            if (self.free_list) |node| {
                self.free_list = node.next;
                self.pool_size -= 1;
                self.reused_count += 1;
                return &node.data;
            }

            // Create new object
            const node = try self.allocator.create(Node);
            node.data = std.mem.zeroes(T);
            node.next = null;
            self.created_count += 1;
            return &node.data;
        }

        /// Return an object to the pool
        pub fn release(self: *Self, obj: *T) void {
            const node = @as(*Node, @alignCast(@fieldParentPtr("data", obj)));
            node.next = self.free_list;
            self.free_list = node;
            self.pool_size += 1;
        }

        /// Get pool statistics
        pub fn getStats(self: *const Self) PoolStats {
            return PoolStats{
                .pool_size = self.pool_size,
                .created_count = self.created_count,
                .reused_count = self.reused_count,
                .hit_rate = if (self.created_count + self.reused_count > 0)
                    @as(f64, @floatFromInt(self.reused_count)) / @as(f64, @floatFromInt(self.created_count + self.reused_count))
                else
                    0.0,
            };
        }
    };
}

/// Pool statistics
pub const PoolStats = struct {
    pool_size: usize,
    created_count: usize,
    reused_count: usize,
    hit_rate: f64,
};

/// Memory pool for variable-sized allocations
pub const MemoryPool = struct {
    const Self = @This();
    const Block = struct {
        data: []u8,
        next: ?*Block,
    };

    allocator: Allocator,
    block_size: usize,
    free_blocks: std.ArrayList(*Block),
    all_blocks: std.ArrayList(*Block),
    current_offset: usize,
    current_block: ?*Block,

    /// Initialize memory pool with block size
    pub fn init(allocator: Allocator, block_size: usize) Self {
        return Self{
            .allocator = allocator,
            .block_size = block_size,
            .free_blocks = std.ArrayList(*Block).init(allocator),
            .all_blocks = std.ArrayList(*Block).init(allocator),
            .current_offset = 0,
            .current_block = null,
        };
    }

    /// Cleanup all allocated blocks
    pub fn deinit(self: *Self) void {
        for (self.all_blocks.items) |block| {
            self.allocator.free(block.data);
            self.allocator.destroy(block);
        }
        self.free_blocks.deinit();
        self.all_blocks.deinit();
    }

    /// Allocate memory from pool
    pub fn alloc(self: *Self, size: usize, alignment: u8) ![]u8 {
        // If size is larger than block size, allocate directly
        if (size > self.block_size) {
            const data = try self.allocator.alloc(u8, size);
            const block = try self.allocator.create(Block);
            block.data = data;
            block.next = null;
            try self.all_blocks.append(block);
            return data;
        }

        // Try to allocate from current block
        if (self.current_block) |block| {
            const aligned_offset = std.mem.alignForward(usize, self.current_offset, alignment);
            if (aligned_offset + size <= block.data.len) {
                const result = block.data[aligned_offset .. aligned_offset + size];
                self.current_offset = aligned_offset + size;
                return result;
            }
        }

        // Need a new block
        try self.allocateNewBlock();
        const block = self.current_block.?;
        const aligned_offset = std.mem.alignForward(usize, 0, alignment);
        const result = block.data[aligned_offset .. aligned_offset + size];
        self.current_offset = aligned_offset + size;
        return result;
    }

    /// Reset pool for reuse (keeps allocated blocks)
    pub fn reset(self: *Self) void {
        // Move current block to free blocks if it exists
        if (self.current_block) |block| {
            self.free_blocks.append(block) catch {
                // This is a memory leak but not a critical failure - continue operation
            };
        }

        // Get a block from free blocks or keep current
        if (self.free_blocks.items.len > 0) {
            self.current_block = self.free_blocks.pop();
        }
        self.current_offset = 0;
    }

    /// Allocate a new block
    fn allocateNewBlock(self: *Self) !void {
        // Try to reuse a free block first
        if (self.free_blocks.items.len > 0) {
            self.current_block = self.free_blocks.pop();
            self.current_offset = 0;
            return;
        }

        // Allocate new block
        const data = try self.allocator.alloc(u8, self.block_size);
        const block = try self.allocator.create(Block);
        block.data = data;
        block.next = null;
        try self.all_blocks.append(block);
        self.current_block = block;
        self.current_offset = 0;
    }

    /// Get memory pool statistics
    pub fn getStats(self: *const Self) PoolStats {
        return PoolStats{
            .pool_size = self.free_blocks.items.len,
            .created_count = self.all_blocks.items.len,
            .reused_count = self.all_blocks.items.len - self.free_blocks.items.len,
            .hit_rate = if (self.all_blocks.items.len > 0)
                @as(f64, @floatFromInt(self.all_blocks.items.len - self.free_blocks.items.len)) / @as(f64, @floatFromInt(self.all_blocks.items.len))
            else
                0.0,
        };
    }
};

/// String cache for deduplicating common strings
pub const StringCache = struct {
    const Self = @This();
    const Entry = struct {
        key: []const u8,
        value: []const u8,
        ref_count: usize,
    };

    allocator: Allocator,
    cache: std.HashMap(u64, Entry, std.hash_map.AutoContext(u64), 80),
    hit_count: usize,
    miss_count: usize,

    /// Initialize string cache
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .cache = std.HashMap(u64, Entry, std.hash_map.AutoContext(u64), 80).init(allocator),
            .hit_count = 0,
            .miss_count = 0,
        };
    }

    /// Cleanup cache
    pub fn deinit(self: *Self) void {
        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.value);
        }
        self.cache.deinit();
    }

    /// Get or create cached string
    pub fn intern(self: *Self, string: []const u8) ![]const u8 {
        const hash = std.hash_map.hashString(string);

        if (self.cache.getPtr(hash)) |entry| {
            entry.ref_count += 1;
            self.hit_count += 1;
            return entry.value;
        }

        // Create new entry
        const owned_string = try self.allocator.dupe(u8, string);
        const entry = Entry{
            .key = owned_string,
            .value = owned_string,
            .ref_count = 1,
        };

        try self.cache.put(hash, entry);
        self.miss_count += 1;
        return owned_string;
    }

    /// Release cached string
    pub fn release(self: *Self, string: []const u8) void {
        const hash = std.hash_map.hashString(string);
        if (self.cache.getPtr(hash)) |entry| {
            entry.ref_count -= 1;
            if (entry.ref_count == 0) {
                self.allocator.free(entry.value);
                _ = self.cache.remove(hash);
            }
        }
    }

    /// Get cache statistics
    pub fn getStats(self: *const Self) CacheStats {
        return CacheStats{
            .entries = self.cache.count(),
            .hit_count = self.hit_count,
            .miss_count = self.miss_count,
            .hit_rate = if (self.hit_count + self.miss_count > 0)
                @as(f64, @floatFromInt(self.hit_count)) / @as(f64, @floatFromInt(self.hit_count + self.miss_count))
            else
                0.0,
        };
    }
};

/// Cache statistics
pub const CacheStats = struct {
    entries: u32,
    hit_count: usize,
    miss_count: usize,
    hit_rate: f64,
};

/// Arena allocator wrapper for temporary allocations
pub const TempArena = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,

    /// Initialize temporary arena
    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    /// Cleanup arena (frees all allocations)
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Get allocator interface
    pub fn allocator(self: *Self) Allocator {
        return self.arena.allocator();
    }

    /// Reset arena (keeps backing memory)
    pub fn reset(self: *Self) void {
        _ = self.arena.reset(.retain_capacity);
    }
};

// Tests
test "ObjectPool basic functionality" {
    const allocator = testing.allocator;

    const TestStruct = struct {
        value: i32,
        data: [16]u8,
    };

    var pool = ObjectPool(TestStruct).init(allocator);
    defer pool.deinit();

    // Test acquiring objects
    const obj1 = try pool.acquire();
    obj1.value = 42;

    const obj2 = try pool.acquire();
    obj2.value = 84;

    // Test releasing and reusing
    pool.release(obj1);
    const obj3 = try pool.acquire(); // Should reuse obj1

    // Release remaining objects
    pool.release(obj2);
    pool.release(obj3);

    try testing.expect(pool.getStats().reused_count > 0);
    try testing.expect(pool.getStats().hit_rate > 0.0);
}

test "MemoryPool allocation" {
    const allocator = testing.allocator;

    var pool = MemoryPool.init(allocator, 1024);
    defer pool.deinit();

    // Test small allocations
    const data1 = try pool.alloc(64, 8);
    const data2 = try pool.alloc(128, 8);

    try testing.expect(data1.len == 64);
    try testing.expect(data2.len == 128);

    // Test large allocation (bigger than block size)
    const large_data = try pool.alloc(2048, 8);
    try testing.expect(large_data.len == 2048);

    // Test reset
    pool.reset();
    const data3 = try pool.alloc(64, 8);
    try testing.expect(data3.len == 64);
}

test "StringCache deduplication" {
    const allocator = testing.allocator;

    var cache = StringCache.init(allocator);
    defer cache.deinit();

    // Test interning same string
    const str1 = try cache.intern("hello");
    const str2 = try cache.intern("hello");
    const str3 = try cache.intern("world");

    try testing.expect(str1.ptr == str2.ptr); // Same pointer (deduplicated)
    try testing.expect(str1.ptr != str3.ptr); // Different strings

    const stats = cache.getStats();
    try testing.expect(stats.hit_count == 1); // Second "hello" was a hit
    try testing.expect(stats.miss_count == 2); // First "hello" and "world" were misses

    // Test releasing
    cache.release(str1);
    cache.release(str2);
    cache.release(str3);
}

test "TempArena basic usage" {
    const allocator = testing.allocator;

    var arena = TempArena.init(allocator);
    defer arena.deinit();

    const temp_allocator = arena.allocator();

    // Allocate some temporary data
    const data1 = try temp_allocator.alloc(u8, 100);
    const data2 = try temp_allocator.alloc(i32, 50);

    try testing.expect(data1.len == 100);
    try testing.expect(data2.len == 50);

    // Reset arena
    arena.reset();

    // Allocate again (should reuse memory)
    const data3 = try temp_allocator.alloc(u8, 200);
    try testing.expect(data3.len == 200);
}

test "ObjectPool statistics" {
    const allocator = testing.allocator;

    var pool = ObjectPool(i32).init(allocator);
    defer pool.deinit();

    // Create and release objects to test statistics
    const obj1 = try pool.acquire();
    const obj2 = try pool.acquire();
    pool.release(obj1);
    pool.release(obj2);

    const obj3 = try pool.acquire(); // Reuse
    const obj4 = try pool.acquire(); // Reuse

    // Release remaining objects
    pool.release(obj3);
    pool.release(obj4);

    const stats = pool.getStats();
    try testing.expect(stats.created_count == 2);
    try testing.expect(stats.reused_count == 2);
    try testing.expect(stats.hit_rate == 0.5); // 2 reuses out of 4 total acquisitions
}

test "MemoryPool alignment" {
    const allocator = testing.allocator;

    var pool = MemoryPool.init(allocator, 1024);
    defer pool.deinit();

    // Test different alignments
    const data1 = try pool.alloc(1, 1);
    const data2 = try pool.alloc(1, 8);
    const data3 = try pool.alloc(1, 16);

    try testing.expect(@intFromPtr(data1.ptr) % 1 == 0);
    try testing.expect(@intFromPtr(data2.ptr) % 8 == 0);
    try testing.expect(@intFromPtr(data3.ptr) % 16 == 0);
}

test "StringCache reference counting" {
    const allocator = testing.allocator;

    var cache = StringCache.init(allocator);
    defer cache.deinit();

    const str1 = try cache.intern("test");
    const str2 = try cache.intern("test"); // Same string, should increment ref count

    try testing.expect(str1.ptr == str2.ptr);

    // Release one reference - string should still be in cache
    cache.release(str1);
    const str3 = try cache.intern("test"); // Should hit cache
    try testing.expect(str2.ptr == str3.ptr);

    // Release remaining references
    cache.release(str2);
    cache.release(str3);

    const stats = cache.getStats();
    try testing.expect(stats.entries == 0); // Should be removed when ref count reaches 0
}
