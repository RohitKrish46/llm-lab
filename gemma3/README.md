# Gemma3-270m
This repository provides a cleanroom implementation of the Gemma3 270M architecture in PyTorch. The code is structured to highlight key components and architectural choices, making it easy to understand and experiment with. It features a custom Gemma3Model class built from scratch, showcasing the unique elements of the Gemma3 design, such as Grouped Query Attention (GQA), SwiGLU-style FeedForward Networks, and a hybrid of sliding and full attention.

## Key Features

Gemma3 Specifics: The code accurately replicates core architectural features, including:

- **RMSNorm**: A custom, zero-centered RMSNorm layer for enhanced stability.

- **RoPE (Rotary Position Embeddings)**: A direct implementation of rotary embeddings for handling sequence length, with dual rope_local_base and rope_base values for different attention mechanisms.

- **Grouped Query Attention (GQA)**: An efficient attention mechanism that groups key-value heads to reduce memory and computational overhead.

- **Sliding Window Attention**: A hybrid attention approach that combines a limited-size sliding window with a global attention mechanism, as seen in the provided code.

- **Clear and Modular Code**: Each core component—FeedForward, RMSNorm, GroupedQueryAttention, and TransformerBlock—is implemented as a separate PyTorch module, promoting readability and reusability.

## Gemma3 Architecture
<img width="1003" height="834" alt="image" src="https://github.com/user-attachments/assets/ac004b9b-5ec3-4c7d-b9e9-f80654153ae9" />


## Architectural Insights
Gemma3 employs several advanced techniques to achieve its efficiency and performance.

1. **Hybrid Attention**: Sliding + Full Attention
The model uses a mix of sliding window attention and full attention layers. This hybrid approach allows the model to capture both local dependencies (with the sliding window) and long-range dependencies (with the full attention layers), optimizing for both performance and memory usage.

2. **Pre-normalization and RMSNorm**
Instead of the more common post-normalization, Gemma3 uses pre-normalization before each attention and feedforward block. It also uses a unique, zero-centered RMSNorm variant.

3. **Rotary Position Embeddings (RoPE)**
Unlike models that use absolute or learned position embeddings, Gemma3 applies RoPE directly to the queries and keys. This allows the model to handle variable sequence lengths and improves performance on long-context tasks. The use of different theta values (rope_local_base and rope_base) allows for fine-tuning the positional information for different attention types.
