def what_is_a_kv_cache_block():
    """
    A KV cache block is the fundamental unit of memory management in PagedAttention.
    
    During transformer inference, each token in the context produces two vectors:
    a Key (K) and a Value (V) — these must be stored so future tokens can attend to them.
    
    In PagedAttention, these K/V vectors are stored in fixed-size blocks.
    block_size=16 means each block holds the K and V vectors for 16 tokens.
    
    Block size of 16 is the default because:
    - It is large enough to amortize the overhead of block table lookups
    - Small enough to limit wasted space (at most 15 tokens wasted at end of sequence)
    - Aligns well with GPU warp/tile sizes for efficient memory access
    
    Each block is a contiguous chunk of VRAM, but blocks themselves
    do not need to be adjacent to each other.
    """
    block_size = 16

    # Each block stores K and V vectors for `block_size` tokens
    # Simulating a block as a dict with key/value tensor placeholders
    def create_kv_block(num_heads, head_dim):
        return {
            "keys":   [[0.0] * head_dim for _ in range(block_size * num_heads)],
            "values": [[0.0] * head_dim for _ in range(block_size * num_heads)],
        }

    num_heads = 8
    head_dim = 64
    block = create_kv_block(num_heads, head_dim)

    print(f"Block size: {block_size} tokens")
    print(f"Keys shape:   [{block_size * num_heads}, {head_dim}]")
    print(f"Values shape: [{block_size * num_heads}, {head_dim}]")
    print(f"Max wasted tokens per request: {block_size - 1}")
    return block
def what_is_a_block_table():
    """
    A block table is a per-request mapping from logical block indices
    to physical block indices in VRAM.
    
    Logical block: the position in the sequence (block 0 = tokens 0-15, block 1 = tokens 16-31, ...)
    Physical block: the actual memory location in VRAM's block pool
    
    Example with 3 concurrent requests and 10 physical blocks [B0..B9]:
    
      Request A (50 tokens → 4 blocks):
        logical[0] → B2   (physical block 2)
        logical[1] → B5   (physical block 5)
        logical[2] → B0   (physical block 0)
        logical[3] → B8   (physical block 8)
    
      Request B (20 tokens → 2 blocks):
        logical[0] → B1
        logical[1] → B7
    
      Request C (16 tokens → 1 block):
        logical[0] → B3
    
    Notice: A's blocks are scattered (B2, B5, B0, B8) — not contiguous!
    The block table makes this transparent to the attention computation.
    This is exactly how OS page tables map virtual addresses to physical RAM.
    """
    block_size = 16

    # Physical block pool (10 blocks, each identified by index)
    num_physical_blocks = 10
    physical_pool = list(range(num_physical_blocks))  # [0, 1, 2, ..., 9]
    allocated = set()

    def allocate_blocks(n):
        blocks = []
        for b in physical_pool:
            if b not in allocated and len(blocks) < n:
                blocks.append(b)
                allocated.add(b)
        return blocks

    import math

    requests = {
        "A": 50,
        "B": 20,
        "C": 16,
    }

    block_tables = {}
    for req_id, num_tokens in requests.items():
        num_blocks = math.ceil(num_tokens / block_size)
        physical_blocks = allocate_blocks(num_blocks)
        block_tables[req_id] = {
            logical_idx: phys
            for logical_idx, phys in enumerate(physical_blocks)
        }

    for req_id, table in block_tables.items():
        print(f"Request {req_id} block table:")
        for logical, physical in table.items():
            print(f"  logical[{logical}] → B{physical}")

    return block_tables


def demonstrate_fragmentation_reduction():
    """
    Internal fragmentation occurs when allocated memory is larger than needed.
    
    In naive KV cache allocation:
      - Each request pre-allocates a contiguous block for max_context_len tokens
      - A request using only 200 of 4096 tokens wastes 3896 token-slots
      - Fragmentation rate = (4096 - 200) / 4096 = 95% wasted!
    
    External fragmentation also occurs:
      - After some requests finish, free memory is scattered in small chunks
      - A new large request may not fit even if total free memory is sufficient
    
    PagedAttention solves both problems:
      - On-demand allocation: blocks are only allocated when tokens are ACTUALLY generated
      - A 200-token response uses exactly ceil(200/16) = 13 blocks = 13 * 16 = 208 slots
      - Maximum waste per request = block_size - 1 = 15 tokens (< 1 block)
      - No external fragmentation: any free block can be assigned to any request
      - Result: ~4% waste vs ~70% in naive systems (measured by vLLM paper)
    """
    import math

    block_size = 16
    max_context_len = 4096
    actual_tokens = 200

    # Naive allocation
    naive_allocated = max_context_len
    naive_wasted = naive_allocated - actual_tokens
    naive_fragmentation_rate = naive_wasted / naive_allocated

    # PagedAttention allocation
    paged_blocks = math.ceil(actual_tokens / block_size)
    paged_allocated = paged_blocks * block_size
    paged_wasted = paged_allocated - actual_tokens
    paged_fragmentation_rate = paged_wasted / paged_allocated

    print("Naive allocation:")
    print(f"  Allocated: {naive_allocated} token-slots")
    print(f"  Wasted:    {naive_wasted} token-slots")
    print(f"  Fragmentation: {naive_fragmentation_rate:.0%}")

    print("\nPagedAttention allocation:")
    print(f"  Blocks used: {paged_blocks}")
    print(f"  Allocated:   {paged_allocated} token-slots")
    print(f"  Wasted:      {paged_wasted} token-slots")
    print(f"  Fragmentation: {paged_fragmentation_rate:.1%}")

    return {
        "naive_fragmentation_rate": naive_fragmentation_rate,
        "paged_fragmentation_rate": paged_fragmentation_rate,
        "blocks_used": paged_blocks,
    }
