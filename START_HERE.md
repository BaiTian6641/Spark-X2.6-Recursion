# SparkX-Loop v0.2: Start Here

1. Open [`SparkX_Loop_v02_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v02_OneShot_Colab.ipynb) in Google Colab (or clone directly from `https://github.com/BaiTian6641/Spark-X2.6-Recursion`) and select the **G4 GPU runtime** (96 GB VRAM + 130 GB System RAM).
2. Add two secrets in **Colab Secrets**:
   - `HF_TOKEN` (or `HF_TOKEM`): used to upload full-finetuned stages (`warmup`, `R8`, `R12`, `R24`) and periodic intra-stage checkpoints (`SAVE_EVERY_STEPS=250`) to `https://huggingface.co/TassP0lar` (default repo ID: `TassP0lar/Spark-X2.6-4B-Loop-mHC`).
   - `OPENCODE_API_KEY` (or `OPENCODE_API`): used to query OpenCode (`https://opencode.ai/inference/openai/v1`) with `deepseek-v4-flash`, `haiku5.5`, and `mimo-v2.6-pro` for supervised CoT (`<think>...</think>`) synthesis and curriculum augmentation.
3. **Runtime > Run all**. Cell 1 clones/syncs `https://github.com/BaiTian6641/Spark-X2.6-Recursion.git`, runs [`setup_g4_env.sh`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/setup_g4_env.sh) (installing Unsloth Studio `unsloth` + `unsloth_zoo` + `bitsandbytes` + `cut-cross-entropy`), patches `XHToken/Spark-X2.6-4B` for **1M-context-safe fused SDPA** (`patch_spark_attention_for_long_context`), and launches the **23-hour Full Finetuning (`TRAIN_MODE='full'`)** run with `PagedAdamW8bit` + 130 GB System RAM sequence packing.
4. **Aggressive Looped Recursive + Memory-Tiered MoE-Engram Architecture**:
   - **Sublayer Dual-mHC (`mhc_attn` + `mhc_ffn`) & Pass Adapters**: Token-dependent FP32 RMSNorm + Sinkhorn Birkhoff routing before both attention and SwiGLU MLP across all 24 recurrent layers (`[4, 28)`, 96 Sinkhorn routing steps across 2 passes) with zero-initialized SwiGLU deliberation adapters (`rank=16`, `max_passes=4`).
   - **Memory-Tiered MoE-Engram Backbone (`TieredMoEEngram` / `HashedBigramEngram`)**: Inserted after the Prelude (`Layer 3`) and at every **4-layer Spark macro-block anchor (`Layers 7, 11, 15, 19, 23, 27`)** inside the recurrent core (`7` Engram layers × `16` experts = `112` MoE memory experts across the backbone):
     - **Tier 1 (Lexical N-Gram Tier)**: $O(1)$ multi-head causal hash lookup over `2-gram`, `3-gram`, and `skip-2-gram` windows (zero matmul FLOPs, reused across loop passes).
     - **Tier 2 (Hot Shared Knowledge Tier)**: DeepSeekMoE-style always-active compact shared expert (`shared_down -> SiLU -> shared_up`) for universal reasoning primitives.
     - **Tier 3 (Cold/Domain Sparse Routed Memory Experts)**: MoE + PEER/UltraMemV2 Top-$K$ (`K=2` of `16`) memory experts with pass-conditioned routing (`pass_router_bias`) and auxiliary-loss-free load balancing (`expert_bias`, `expert_load_ema`).
     - **Gentle Engram Formation**: Zero-initialized `project.weight = 0` preserves 100% original `Spark-X2.6-4B` behavior at Step 0 while compacting new knowledge during full finetuning (`LR_ADAPTER=1.5e-4` vs. gentle backbone `LR_FULL=1.5e-5`).
5. **Curated 26-Source Reasoning, Scientific & Multi-Turn Conversation Mixture**:
   - **Claude Fable & Claude Opus (4.6/4.7/4.8)** CoT traces
   - **GPT Astra / GPT-6 / GPT-5** reasoning traces
   - **Scientific Reasoning**: `nvidia/OpenScienceReasoning-2`, `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (`PHD-Science`), `MegaScience/MegaScience`, `MegaScience/TextbookReasoning`, `OpenDataArena/Spark-234K`, `HuggingFaceTB/smoltalk2` (`OpenThoughts3_1.2M`)
   - **Multi-Turn Conversation**: `nvidia/Nemotron-Instruction-Following-Chat-v1`, `NousResearch/Hermes-3-Dataset`, `HuggingFaceH4/ultrachat_200k`, `DataCreatorAI/Multi-Turn-Conversational-SFT`
   - **GSM8K** + live **OpenCode** (`deepseek-v4-flash`, `haiku5.5`, `mimo-v2.6-pro`) traces.
6. After export, load and run inference with [`load_inference.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/load_inference.py), which automatically applies the 1M-context fused SDPA patch and supports fast per-visit KV-cached looped generation (`generate_cached` via `_VisitCache`).
