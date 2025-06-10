class MLA(nn.module):
    """
    Multi-Headed Latent Attention layer
    Attributes:
        dim (int) : Dimension of the input features (token embeddings)
        num_heads (int) : Number of attention heads.
        n_local_heads (int) : Number of local attention heads for distributed systems.
        q_lora_rank (int) : Rank for low rank query projection. 
        kv_lora_rank (int) : Rank for low rank key and value projection.
        qk_nope_head_dim (int) : Dimensionality for non-positional (not RoPE) query and key heads.
        qk_rope_head_dim (int) : Dimensionality for Rotary-positional(RoPE) query and key heads.
        qk_head_dim (int) : Dimensionality for query and key heads. (qk_nope_head_dim + qk_rope_head_dim)
        v_head_dim (int) : Dimensionality for value heads.
        softmax_scale (float) : Scaling factor for softmax in attention computation.
    """
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkb_b = ColumnParallelLinear(self.kv_lora_rank , self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
            self.rope = RotaryEmbedding(args.max_seq_len, self.qk_rope_head_dim)
            