from transformers import PretrainedConfig

class CyclicFormerConfig(PretrainedConfig):
    model_type = "cyclicformer"
    def __init__(
        self,
        vocab_size = 5000,
        hidden_size = 200,
        cyclic_size = 100,
        num_attention_heads = 4,
        num_hidden_layers = 4,
        drop_prob = 0.1,
        pad_token_id = 0,
        n_loop = 3,
        rms_norm_eps = 1e-6,
        initializer_range = 0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.cyclic_size = cyclic_size
        self.n_loop = n_loop
        self.num_hidden_layers = num_hidden_layers
        self.drop_prob = drop_prob
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(
            **kwargs,
        )
