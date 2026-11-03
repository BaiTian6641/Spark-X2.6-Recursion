# References & Citations (`Spark-X2.6-Recursion` / `SparkX-Loop v0.2`)

This document provides complete academic citations, upstream model attributions, and dataset references for the **`Spark-X2.6-Recursion`** (`SparkX-Loop v0.2`) architecture and full-finetuning pipeline implemented in [`sparkx_loop_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py) and [`SparkX_Loop_v02_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v02_OneShot_Colab.ipynb).

---

## 1. Core Architectural Research Papers

### 1.1 Manifold-Constrained Hyper-Connections (mHC) & Multi-Stream Residuals
Used in [`TwoStreamMHC`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L323-L485) for sublayer Dual-mHC (`mhc_attn` + `mhc_ffn`), token-dependent FP32 RMSNorm routing, and Sinkhorn-Knopp projection onto the Birkhoff polytope of doubly-stochastic matrices:

1. **Manifold-Constrained Hyper-Connections (mHC)**
   - **Authors:** Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang (*DeepSeek-AI*)
   - **Preprint:** [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (2025)
2. **Hyper-Connections (HC)**
   - **Authors:** Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (*ByteDance Seed*)
   - **Preprint:** [arXiv:2409.19606](https://arxiv.org/abs/2409.19606) (ICLR 2025)

```bibtex
@article{xie2025mhc,
  title={Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025},
  url={https://arxiv.org/abs/2512.24880}
}

@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://arxiv.org/abs/2409.19606}
}
```

---

### 1.2 Conditional Memory (Engram) & Memory-Tiered MoE Routing
Used in [`HashedBigramEngram` / `TieredMoEEngram`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L201-L320) and [`SparkXLoop`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L604-L625) to implement the 3-tier backbone memory architecture (Tier 1: $O(1)$ Multi-Head Causal N-Gram Lookup; Tier 2: Hot Shared Expert; Tier 3: Sparse Top-$K$ PEER/UltraMemV2 Routed Memory Experts with Auxiliary-Loss-Free Load Balancing and Context-Aware Gating):

1. **Engram: Conditional Memory via Scalable Lookup**
   - **Authors:** Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Chengqi Deng, Shangyan Zhou, Chenggang Zhao, Zhewen Hao, Yukun Li, Han Zhang, Zhengyan Zhang, Yixu Wei, M. Y. Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang (*DeepSeek-AI & Peking University*)
   - **Preprint:** [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) (2026)
2. **PEER: Mixture of a Million Experts (Parameter Efficient Expert Retrieval)**
   - **Author:** Xu Owen He (*Google DeepMind*)
   - **Preprint:** [arXiv:2407.04153](https://arxiv.org/abs/2407.04153) (2024)
3. **UltraMem & UltraMemV2: Ultra-Sparse Memory Networks**
   - **Authors:** Zihao Huang, Qiyang Min, Hongzhi Huang, Defa Zhu, Yutao Zeng, Ran Guo, Xun Zhou (*ByteDance Seed*)
   - **Preprints:** [arXiv:2411.12364](https://arxiv.org/abs/2411.12364) (2024) and [arXiv:2508.18756](https://arxiv.org/abs/2508.18756) (2025)
4. **DeepSeekMoE & Auxiliary-Loss-Free Load Balancing**
   - **Authors:** Damai Dai et al. (*DeepSeek-AI*), [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) (2024); Lean Wang et al. (*DeepSeek-AI*), [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) (2024)

```bibtex
@article{cheng2026engram,
  title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author={Cheng, Xin and Tian, Rui and Zeng, Wangding and Dai, Damai and Chen, Qinyu and Wang, Bingxuan and Xie, Zhenda and Huang, Kezhao and Yu, Xingkai and Deng, Chengqi and Zhou, Shangyan and Zhao, Chenggang and Hao, Zhewen and Li, Yukun and Zhang, Han and Zhang, Zhengyan and Wei, Yixu and Xu, M. Y. and Zhang, Huishuai and Zhao, Dongyan and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2601.07372},
  year={2026},
  url={https://arxiv.org/abs/2601.07372}
}

@article{he2024peer,
  title={Mixture of a Million Experts},
  author={He, Xu Owen},
  journal={arXiv preprint arXiv:2407.04153},
  year={2024},
  url={https://arxiv.org/abs/2407.04153}
}

@article{huang2024ultramem,
  title={Ultra-Sparse Memory Network},
  author={Huang, Zihao and Min, Qiyang and Huang, Hongzhi and Zhu, Defa and Zeng, Yutao and Guo, Ran and Zhou, Xun},
  journal={arXiv preprint arXiv:2411.12364},
  year={2024},
  url={https://arxiv.org/abs/2411.12364}
}

@article{huang2025ultramemv2,
  title={UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning},
  author={Huang, Zihao and Min, Qiyang and Huang, Hongzhi and Zhu, Defa and Zeng, Yutao and Guo, Ran and Zhou, Xun},
  journal={arXiv preprint arXiv:2508.18756},
  year={2025},
  url={https://arxiv.org/abs/2508.18756}
}

@article{dai2024deepseekmoe,
  title={DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models},
  author={Dai, Damai and Deng, Chengqi and Zhao, Chenggang and Xu, R. X. and Gao, Huazuo and Chen, Deli and Li, Jiashi and Zeng, Wangding and Yu, Xingkai and Wu, Y. and others},
  journal={arXiv preprint arXiv:2401.06066},
  year={2024},
  url={https://arxiv.org/abs/2401.06066}
}

@article{wang2024auxlossfree,
  title={Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts},
  author={Wang, Lean and Gao, Huazuo and Zhao, Chenggang and Sun, Xu and Dai, Damai},
  journal={arXiv preprint arXiv:2408.15664},
  year={2024},
  url={https://arxiv.org/abs/2408.15664}
}
```

---

### 1.3 Looped Recursive Transformers & Latent Depth Recurrence
Used in [`SparkXLoop`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L556-L830) and [`_VisitCache`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L66-L85) for progressive Prelude/Recurrent-Core/Coda depth expansion (`warmup -> R8 -> R12 -> R24`, 36 physical -> 60 effective layers) and per-visit KV-cached looped decoding:

1. **Reasoning with Latent Thoughts: On the Power of Looped Transformers**
   - **Authors:** Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi (*Google Research*)
   - **Preprint:** [arXiv:2502.17416](https://arxiv.org/abs/2502.17416) (2025)
2. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**
   - **Authors:** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein
   - **Preprint:** [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) (2025)
3. **Looped Transformers as Programmable Computers**
   - **Authors:** Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos
   - **Preprint:** [arXiv:2301.13196](https://arxiv.org/abs/2301.13196) (ICML 2023)
4. **SenseNova-Loop-9B-Preview**
   - **Authors:** SenseTime Research
   - **Model Repository:** [https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview) (2026)

```bibtex
@article{saunshi2025looped,
  title={Reasoning with Latent Thoughts: On the Power of Looped Transformers},
  author={Saunshi, Nikunj and Dikkala, Nishanth and Li, Zhiyuan and Kumar, Sanjiv and Reddi, Sashank J.},
  journal={arXiv preprint arXiv:2502.17416},
  year={2025},
  url={https://arxiv.org/abs/2502.17416}
}

@article{geiping2025recurrent,
  title={Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach},
  author={Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and Singh, Siddharth and Bartoldson, Brian R. and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom},
  journal={arXiv preprint arXiv:2502.05171},
  year={2025},
  url={https://arxiv.org/abs/2502.05171}
}

@inproceedings{giannou2023looped,
  title={Looped Transformers as Programmable Computers},
  author={Giannou, Angeliki and Rajput, Shashank and Sohn, Jy-yong and Lee, Kangwook and Lee, Jason D. and Papailiopoulos, Dimitris},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023},
  url={https://arxiv.org/abs/2301.13196}
}
```

---

### 1.4 Long-Context Attention (NoPE + Sliding Window), Chunked CE & Memory-Efficient Training
Used in [`patch_spark_attention_for_long_context`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L92-L162), [`fused_masked_cross_entropy`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L488-L522), and [`build_optimizer`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L868-L884):

1. **Base Model: `XHToken/Spark-X2.6-4B`**
   - **Repository:** [https://huggingface.co/XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) (`3 × sliding_attention w=512 RoPE + 1 × full_attention NoPE` per 4-layer macro-block, 1M token context support)
2. **Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy)**
   - **Authors:** Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, Philipp Krähenbühl (*Apple*)
   - **Preprint:** [arXiv:2411.09009](https://arxiv.org/abs/2411.09009) (2024)
3. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - **Author:** Tri Dao
   - **Preprint:** [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (2023)
4. **8-bit Optimizers via Block-wise Quantization (`PagedAdamW8bit`)**
   - **Authors:** Tim Dettmers, Mike Lewis, Sam Shleifer, Luke Zettlemoyer
   - **Preprint:** [arXiv:2110.02861](https://arxiv.org/abs/2110.02861) (ICLR 2022)
5. **Unsloth Studio**
   - **Authors:** Daniel Han, Michael Han, and Unsloth AI Team
   - **Repository:** [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

---

### 1.5 Latent Sparse Attention (`D-LSA`), Nested Learning, Sleep-Time Memory Compilation & Null-Space Knowledge Editing (`AlphaEdit`)
Used in [`LatentSparseAttentionIndexer`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L756-L1025), [`ZipfianTieredMoEEngram.compile_clusters_alphaedit`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L1867-L2096), [`SparkXLoopV03.run_sleep_time_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2627-L2779), and [`validate_vertical_sleep_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L3909-L4062) to implement horizontal working attention (`D-LSA`) coupled with vertical sleep-time memory consolidation (`LSA -> AlphaEdit Null-Space MoE-Engram`):

1. **Latent Sparse Attention: Extending Transformative Context to 10M Tokens via Semantic Index Clustering (`D-LSA` & `T-LSA`)**
   - **Authors:** Petr Kubicek et al. (*Prefrontal AI*)
   - **Preprint & Code:** `arXiv:2610.42147` (2026), [https://github.com/Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention)
2. **AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models**
   - **Authors:** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, Tat-Seng Chua
   - **Publication:** *International Conference on Learning Representations (ICLR 2025, Outstanding Paper)*, [arXiv:2410.02355](https://arxiv.org/abs/2410.02355)
3. **Nested Learning: The Illusion of Deep Learning Architectures**
   - **Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (*Google Research*)
   - **Publication:** *Advances in Neural Information Processing Systems (NeurIPS 2025)*, [arXiv:2512.24695](https://arxiv.org/abs/2512.24695)
4. **Titans: Learning to Memorize at Test Time**
   - **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (*Google Research*)
   - **Preprint:** [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) (2025)
5. **Sleep-Time Compute: Beyond Inference Scaling at Test-Time**
   - **Authors:** Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez (*Letta & UC Berkeley*)
   - **Preprint:** [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) (2025)
6. **Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study**
   - **Authors:** Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Ré (*Hazy Research, Stanford University*)
   - **Preprint:** [arXiv:2506.06266](https://arxiv.org/abs/2506.06266) (2025)
7. **Doc-to-LoRA (`D2L`): Learning to Instantly Internalize Contexts**
   - **Authors:** Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange (*Sakana AI*)
   - **Preprint:** [arXiv:2602.15902](https://arxiv.org/abs/2602.15902) (2026)
8. **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (`NSA`)**
   - **Authors:** Jingyang Yuan et al. (*DeepSeek-AI & Peking University*)
   - **Preprint:** [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) (2025)

```bibtex
@article{kubicek2026lsa,
  title={Latent Sparse Attention: Extending Transformative Context to 10M Tokens via Semantic Index Clustering},
  author={Kubicek, Petr and others},
  journal={arXiv preprint arXiv:2610.42147},
  year={2026},
  url={https://github.com/Prefrontal-AI/latent-sparse-attention}
}

@inproceedings{fang2025alphaedit,
  title={AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models},
  author={Fang, Junfeng and Jiang, Houcheng and Wang, Kun and Ma, Yunshan and Jie, Shi and Wang, Xiang and He, Xiangnan and Chua, Tat-Seng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://arxiv.org/abs/2410.02355}
}

@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://arxiv.org/abs/2512.24695}
}

@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025},
  url={https://arxiv.org/abs/2501.00663}
}

@article{lin2025sleeptime,
  title={Sleep-Time Compute: Beyond Inference Scaling at Test-Time},
  author={Lin, Kevin and Snell, Charlie and Wang, Yu and Packer, Charles and Wooders, Sarah and Stoica, Ion and Gonzalez, Joseph E.},
  journal={arXiv preprint arXiv:2504.13171},
  year={2025},
  url={https://arxiv.org/abs/2504.13171}
}

@article{eyuboglu2025cartridges,
  title={Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study},
  author={Eyuboglu, Sabri and Ehrlich, Ryan and Arora, Simran and Guha, Neel and Zinsley, Dylan and Liu, Emily and Tennien, Will and Rudra, Atri and Zou, James and Mirhoseini, Azalia and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2506.06266},
  year={2025},
  url={https://arxiv.org/abs/2506.06266}
}

@article{charakorn2026doctolora,
  title={Doc-to-LoRA: Learning to Instantly Internalize Contexts},
  author={Charakorn, Rujikorn and Cetin, Edoardo and Tang, Yujin and Lange, Robert Tjarko},
  journal={arXiv preprint arXiv:2602.15902},
  year={2026},
  url={https://arxiv.org/abs/2602.15902}
}

@article{yuan2025nsa,
  title={Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
  author={Yuan, Jingyang and Gao, Huazuo and Dai, Damai and Luo, Junyu and Zhao, Liang and Zhang, Zhengyan and Xie, Zhenda and Wei, Y. X. and Wang, Lean and Xiao, Zhiping and Wang, Yuqing and Ruan, Chong and Zhang, Ming and Liang, Wenfeng and Zeng, Wangding},
  journal={arXiv preprint arXiv:2502.11089},
  year={2025},
  url={https://arxiv.org/abs/2502.11089}
}
```

---

## 2. Curated Training & Evaluation Datasets (26 Hugging Face Sources)

### 2.1 Scientific Reasoning & PhD/University STEM Chain-of-Thought
| Dataset ID | Partition / Split | Description & Citation |
| :--- | :--- | :--- |
| [`nvidia/OpenScienceReasoning-2`](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) | `default / train` | Multi-domain synthetic & verified scientific reasoning corpus with `<think>...</think>` traces and `\boxed{}` solutions across biology, chemistry, physics, and immunology. |
| [`Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned`](https://huggingface.co/datasets/Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned) | `PHD-Science / train` | Cleaned PhD-level scientific research trajectories with structured `<think>...</think>` reasoning distilled from Kimi-K2.5. |
| [`MegaScience/MegaScience`](https://huggingface.co/datasets/MegaScience/MegaScience) | `default / train` | Large-scale (1.25M) open scientific reasoning post-training mixture (*Fan et al., 2025*, [arXiv:2507.16812](https://arxiv.org/abs/2507.16812)). |
| [`MegaScience/TextbookReasoning`](https://huggingface.co/datasets/MegaScience/TextbookReasoning) | `default / train` | University-level textbook questions, derivations, and reference answers across medicine, physics, chemistry, and biology (*Fan et al., 2025*, [arXiv:2507.16812](https://arxiv.org/abs/2507.16812)). |
| [`OpenDataArena/Spark-234K`](https://huggingface.co/datasets/OpenDataArena/Spark-234K) | `default / train` | Graduate-level physics, chemistry, and quantitative STEM derivations curated by OpenDataArena. |
| [`HuggingFaceTB/smoltalk2`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | `Mid / OpenThoughts3_1.2M` | Curated STEM, science, and algorithmic reasoning subset from SmolTalk2 / OpenThoughts3 (*Guha et al., 2025*). |

```bibtex
@article{fan2025megascience,
  title={MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning},
  author={Fan, Run-Ze and Wang, Zengzhi and Liu, Pengfei},
  journal={arXiv preprint arXiv:2507.16812},
  year={2025},
  url={https://arxiv.org/abs/2507.16812}
}

@misc{nvidia2025opensciencereasoning2,
  title={OpenScienceReasoning-2 Dataset},
  author={{NVIDIA}},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2}}
}
```

---

### 2.2 High-Quality Multi-Turn Conversation & Instruction Following
| Dataset ID | Partition / Split | Description & Citation |
| :--- | :--- | :--- |
| [`nvidia/Nemotron-Instruction-Following-Chat-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | `default / chat_if` | Multi-turn conversational and instruction-following dataset developed for NVIDIA Nemotron post-training. |
| [`NousResearch/Hermes-3-Dataset`](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset) | `default / train` | High-quality multi-turn system-prompted assistant dialogues (*Teknium et al., 2024*, [arXiv:2408.11857](https://arxiv.org/abs/2408.11857)). |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `default / train_sft` | Filtered 200k multi-turn dialogue corpus derived from UltraChat (*Ding et al., 2023*, [arXiv:2305.14233](https://arxiv.org/abs/2305.14233); *Tunstall et al., 2023*, [arXiv:2310.16944](https://arxiv.org/abs/2310.16944)). |
| [`DataCreatorAI/Multi-Turn-Conversational-SFT`](https://huggingface.co/datasets/DataCreatorAI/Multi-Turn-Conversational-SFT) | `default / train` | Reviewed multi-turn conversational SFT corpus in OpenAI `messages` format across technical and everyday domains. |

```bibtex
@article{teknium2024hermes3,
  title={Hermes 3 Technical Report},
  author={Teknium, Ryan and Quesnelle, Jeffrey and Guang, Chen},
  journal={arXiv preprint arXiv:2408.11857},
  year={2024},
  url={https://arxiv.org/abs/2408.11857}
}

@article{ding2023ultrachat,
  title={Enhancing Chat Language Models by Scaling High-quality Instructional Conversations},
  author={Ding, Ning and Chen, Yulin and Xu, Bokai and Qin, Yujia and Zheng, Zhi and Hu, Shengding and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  journal={arXiv preprint arXiv:2305.14233},
  year={2023},
  url={https://arxiv.org/abs/2305.14233}
}
```

---

### 2.3 Claude Fable, Claude Opus & GPT Astra / GPT-5 / GPT-6 Reasoning Traces
| Family | Dataset ID | Description |
| :--- | :--- | :--- |
| **Claude Fable** | [`kelexine/fable-5-sft-traces`](https://huggingface.co/datasets/kelexine/fable-5-sft-traces) | Multi-turn SFT & explicit `reasoning_content` CoT traces. |
| **Claude Fable** | [`HelioAI/Claude-Fable-5-5500x`](https://huggingface.co/datasets/HelioAI/Claude-Fable-5-5500x) | 5,500 long-form analytical `<think>...</think>` reasoning traces. |
| **Claude Fable** | [`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x) | 10,000 filtered high-effort reasoning traces. |
| **Claude Fable** | [`saidutta69/fable-5-premium-v2`](https://huggingface.co/datasets/saidutta69/fable-5-premium-v2) | Curated `openai_chat/train.parquet` reasoning & chat split. |
| **Claude Fable** | [`armand0e/Fable-5-Chat`](https://huggingface.co/datasets/armand0e/Fable-5-Chat) | Conversational instruction-following dialogues. |
| **Claude Opus** | [`Roman1111111/claude-opus-4.6-10000x`](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x) | 10,000 Opus 4.6 `<think>...</think>` reasoning traces. |
| **Claude Opus** | [`TeichAI/lordx64-claude-opus-4.7-max-cleaned`](https://huggingface.co/datasets/TeichAI/lordx64-claude-opus-4.7-max-cleaned) | Cleaned Opus 4.7 max-effort reasoning traces. |
| **Claude Opus** | [`angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`](https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k) | 8,700 combined Opus 4.6/4.7 reasoning traces. |
| **Claude Opus** | [`ronaldcmz/claude_opus_4.8_max_thinking_5k_v2`](https://huggingface.co/datasets/ronaldcmz/claude_opus_4.8_max_thinking_5k_v2) | 5,000 Opus 4.8 max-thinking traces. |
| **Claude Opus** | [`TeichAI/Claude-Opus-4.6-Reasoning-887x`](https://huggingface.co/datasets/TeichAI/Claude-Opus-4.6-Reasoning-887x) | Curated high-density Opus 4.6 reasoning traces. |
| **GPT Astra / 6** | [`FlaneAI/GPT-6Astra-scientific-research`](https://huggingface.co/datasets/FlaneAI/GPT-6Astra-scientific-research) | Scientific research trajectories (`step_by_step_trace` + `final_solution`). |
| **GPT-5 Series** | [`Roman1111111/gpt-5.4-step-by-step-reasoning`](https://huggingface.co/datasets/Roman1111111/gpt-5.4-step-by-step-reasoning) | Step-by-step analytical derivations. |
| **GPT-5 Series** | [`vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`](https://huggingface.co/datasets/vanty120/Gpt-5.4-Xhigh-Reasoning-2000x) | 2,000 `xhigh`-effort `<think>...</think>` reasoning traces. |
| **GPT-5 Series** | [`TeichAI/gpt-5.1-high-reasoning-1000x`](https://huggingface.co/datasets/TeichAI/gpt-5.1-high-reasoning-1000x) | 1,000 high-effort reasoning traces. |
| **GPT-5.5** | [`Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset`](https://huggingface.co/datasets/Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset) | `data/distilled/gpt55_thinking/shard-00000.jsonl` reasoning shard. |
| **Math Benchmark** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | Grade School Math 8K step-by-step verification (*Cobbe et al., 2021*, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)). |

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021},
  url={https://arxiv.org/abs/2110.14168}
}
```

---

### 2.4 Natural Human Language, Literary Prose & Anti-Slop ("De-Claudification") Datasets (`SparkX-Loop v0.3 / En4B`)
Curated specifically to counteract synthetic "AI slop" / "Claudish" phrasing (formulaic transitions, excessive bulleting, repetitive moralizing, and adjective bloat) and ground the model in authentic human prose and multilingual expression:

| Dataset ID | Partition / Split | Description & Citation |
| :--- | :--- | :--- |
| [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | `default / train` | 10,000 instructions, essays, summaries, and creative texts written **100% by skilled human annotators** with zero synthetic LLM generation (*Rajani et al., 2023*). |
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) | `default / train` | Crowdsourced human-written multilingual conversation trees across 35+ languages (*Köpf et al., NeurIPS 2023*, [arXiv:2304.07327](https://arxiv.org/abs/2304.07327)). |
| [`SAA-Lab/LitBench`](https://huggingface.co/datasets/SAA-Lab/LitBench) | `default / train` | Human-authored literary storytelling and debiased human preference corpus designed to eliminate synthetic "purple prose" and formulaic AI narrative tropes. |
| [`ChaoticNeutrals/Creative_Writing-ShareGPT`](https://huggingface.co/datasets/ChaoticNeutrals/Creative_Writing-ShareGPT) | `default / train` | Multi-turn natural prose and literary style-adherence dialogues free of standard corporate assistant phrasing. |
| [`Pageshift-Entertainment/LongPage`](https://huggingface.co/datasets/Pageshift-Entertainment/LongPage) | `default / train` | Long-form human narrative and structural pacing corpus for natural multi-page discourse. |
| [`allenai/tulu-v2-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) | `LIMA / human splits` | Curated human-authored instruction-response pairs from LIMA (*Zhou et al., NeurIPS 2023*, [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)) and OpenAssistant. |

```bibtex
@misc{rajani2023norobots,
  title={No Robots: 10,000 High-Quality Human-Written Instructions},
  author={Rajani, Nazneen and Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rush, Alexander M. and Wolf, Thomas},
  year={2023},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/HuggingFaceH4/no_robots}}
}

@article{kopf2023openassistant,
  title={OpenAssistant Conversations -- Democratizing Large Language Model Alignment},
  author={K{\"o}pf, Andreas and Kilcher, Yannic and von R{\"u}tte, Dimitri and Anagnostidis, Sotiris and Tam, Zhi-Rui and Stevens, Keith and Barhoum, Abdullah and Duc, Nguyen Minh and Stanley, Oliver and Nagyfi, Rich{\'a}rd and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023},
  url={https://arxiv.org/abs/2304.07327}
}

@article{zhou2023lima,
  title={LIMA: Less Is More for Alignment},
  author={Zhou, Chunting and Liu, Pengfei and Xu, Puxin and Iyer, Srini and Sun, Jiao and Mao, Yuning and Ma, Xuezhe and Efrat, Avia and Yu, Ping and Yu, Lili and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023},
  url={https://arxiv.org/abs/2305.11206}
}
```

---

### 2.5 Agentic Tool Calling, Model Context Protocol (MCP) & Interleaved Thinking Datasets (`SparkX-Loop v0.3 / En4B-A1B`)
Curated to train mid-reasoning tool invocation (`<think>...</think><tool_call>...</tool_call>` $\to$ `<tool_response>...</tool_response>` $\to$ `<think>...</think>`), strict JSON schema adherence, parallel/dependent function calling, and real-world MCP tool use:

| Dataset ID | Partition / Split | Description & Citation |
| :--- | :--- | :--- |
| [`Agent-Ark/Toucan-1.5M`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | `default / train` | **1.5M multi-turn, multi-tool agent trajectories** synthesized from ~500 real-world **Model Context Protocol (MCP)** environments spanning 2,000+ executable tools ([arXiv:2510.01179](https://arxiv.org/abs/2510.01179), Apache 2.0). |
| [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | `default / train` | BFCL-leading function-calling dataset across **26,507 diverse APIs**, covering single, parallel, nested, and multi-turn dependent tool invocations (*Liu et al., ICLR 2025*, [arXiv:2409.00920](https://arxiv.org/abs/2409.00920), Apache 2.0). |
| [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | `default / train` | 60,000 function-calling trajectories generated via **APIGen** and verified by a 3-tier pipeline (format check, live execution, semantic verification; *Liu et al., 2024*, [arXiv:2406.18518](https://arxiv.org/abs/2406.18518)). |
| [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | `default / train` | 5,000 multi-turn agent-human interplay trajectories with executable ground-truth verification (*Salesforce AI Research, 2025*, [arXiv:2504.03601](https://arxiv.org/abs/2504.03601)). |
| [`hxssgaa/xlam-interleave-thinking-40k`](https://huggingface.co/datasets/hxssgaa/xlam-interleave-thinking-40k) | `default / train` | 40,000 interleaved thinking trajectories where the assistant explicitly plans tool selection and argument deduction inside `<think>...</think>` prior to emitting `<tool_call>`. |
| [`pyromind/agentic-tool-call-dataset-12k`](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k) | `default / train` | 12,000 multi-step agentic tool-use traces pairing `<think>...</think>` internal deliberation with structured JSON `<tool_call>` blocks. |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | `default / train` | Multi-turn XML/JSON `<tools>`, `<tool_call>`, and `<tool_response>` structured function calling and JSON-schema extraction dataset. |
| [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | `SWE-Gym-Lite` | Real-world executable Python software engineering (SWE) agent tasks and unit-test verification trajectories (*Pan et al., ICML 2025*, [arXiv:2412.21139](https://arxiv.org/abs/2412.21139)). |

```bibtex
@article{toucan2025mcp,
  title={TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments},
  author={{Agent-Ark Team}},
  journal={arXiv preprint arXiv:2510.01179},
  year={2025},
  url={https://arxiv.org/abs/2510.01179}
}

@inproceedings{liu2025toolace,
  title={ToolACE: Winning the Points of LLM Function Calling},
  author={Liu, Weiwen and Huang, Xu and Zeng, Xingshan and Hao, Xinlong and Yu, Shuai and Li, Dexun and Wang, Shuai and Gan, Weinan and Liu, Zhengying and Yu, Yuanqing and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://arxiv.org/abs/2409.00920}
}

@article{liu2024apigen,
  title={APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author={Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Kokane, Shirley and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and others},
  journal={arXiv preprint arXiv:2406.18518},
  year={2024},
  url={https://arxiv.org/abs/2406.18518}
}

@article{apigenmt2025,
  title={APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay},
  author={{Salesforce AI Research}},
  journal={arXiv preprint arXiv:2504.03601},
  year={2025},
  url={https://arxiv.org/abs/2504.03601}
}
```

---

### 2.6 Verifiable Math, Code, World-Knowledge & Post-Training Datasets (`SparkX-Loop v0.3 / En4B-A1B`)
Additional gold-standard 2024–2026 post-training corpora selected for **verifier-checked math/code execution**, **broad encyclopedic world knowledge** (to populate the `4.0B-A1B` MoE-Engram), and **decontaminated human preference alignment**:

| Dataset ID | Partition / Split | Description & Citation |
| :--- | :--- | :--- |
| [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | `default / train` | 220,000 competition & university math problems (`NuminaMath 1.5`) with step-by-step `<think>` traces verified via `Math-Verify` and `Llama-3.3-70B-Instruct` (*Hugging Face Open-R1, 2025*, Apache 2.0). |
| [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | `default / train` | 114,000 verifier-checked reasoning traces across math, science, algorithmic coding, and logic puzzles (*Bespoke Labs / OpenThoughts Team, 2025*, Apache 2.0). |
| [`nvidia/Llama-Nemotron-Post-Training-Dataset`](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | `SFT` | NVIDIA's flagship post-training SFT dataset spanning verified math, software engineering, STEM reasoning, and tool calling (*Bercovich et al., NVIDIA, 2025*, CC-BY-4.0). |
| [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `default / train` | AI2's decontaminated open-science SFT mixture covering precise constraint following, humanities, coding, and factual QA (*Lambert et al., 2024*, [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)). |
| [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | `default / train` | Human-annotated multi-attribute preference dataset filtered for top-scoring (`helpfulness >= 4`, `correctness >= 4`, `coherence >= 4`) natural responses (*Wang et al., NeurIPS 2024*, [arXiv:2406.08673](https://arxiv.org/abs/2406.08673)). |
| [`HuggingFaceTB/Cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/Cosmopedia-v2) | `default / train` | Structured textbook chapters and conceptual world-knowledge syntheses across humanities, history, biology, chemistry, and engineering to populate the `En4B-A1B` memory experts. |

```bibtex
@article{lambert2024tulu3,
  title={T{\"u}lu 3: Pushing Frontiers in Open Language Model Post-Training},
  author={Lambert, Nathan and Morrison, Jacob and Pyatkin, Valentina andyi, Shengyi and Ivison, Hamish and Brahman, Faeze and Miranda, Lester James V. and Liu, Alisa and Dziri, Nouha and Lyu, Shane and others},
  journal={arXiv preprint arXiv:2411.15124},
  year={2024},
  url={https://arxiv.org/abs/2411.15124}
}

@inproceedings{wang2024helpsteer2,
  title={HelpSteer2: Open-source dataset for training top-performing reward models},
  author={Wang, Zhilin and Dong, Yi and Olivier, Olivier and Zeng, Jiaqi and Rreed, Gerald and Shen, Makesh Narsimhan and Egert, Daniel and Zhang, Yang and Bukharin, Alexander and Delalleau, Olivier and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://arxiv.org/abs/2406.08673}
}
```

---

## 3. Research References for `Spark-X2.6-4B-En4B-A1B` (8.2B Total / 5.2B Active / 72-Layer Loop / $O(n \log n)$ KV Cache)

### 3.1 MoE Routing Collapse & Representation Collapse Prevention
1. **On the Representation Collapse of Sparse Mixture of Experts (`X-MoE` Hypersphere Cosine Routing)**
   - **Authors:** Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, Heyan Huang, Furu Wei (*Microsoft Research*)
   - **Publication:** *NeurIPS 2022* ([arXiv:2204.09179](https://arxiv.org/abs/2204.09179))
   - **Relevance:** Proves that linear dot-product MoE routers cause hidden states to collapse onto expert centroids; solved by projecting token hidden states and expert keys onto a low-dimensional $\ell_2$-normalized hypersphere (**Cosine Router**) with learnable temperature $\tau$.
2. **ST-MoE: Designing Stable and Transferable Sparse Expert Models (`Router Z-Loss`)**
   - **Authors:** Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus (*Google Brain*)
   - **Preprint:** [arXiv:2202.08906](https://arxiv.org/abs/2202.08906) (2022)
   - **Relevance:** Introduces **Router Z-Loss** $\mathcal{L}_z = \frac{1}{B}\sum_{t} \left(\log \sum_{i=1}^E \exp(z_{t,i})\right)^2$ to bound router logit magnitudes in `bfloat16` and prevent premature hard saturation into a few dominant experts.
3. **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (`Sigmoid Affinity + Dynamic Bias`)**
   - **Authors:** Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai (*DeepSeek-AI*)
   - **Preprint:** [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) (2024)
   - **Relevance:** Replaces softmax competition with independent **Sigmoid Affinity Scores** $s_{t,i} = \sigma(\hat{h}_t^\top e_i)$ and updates per-expert routing bias $b_i$ via sign-EMA load balancing without polluting language modeling gradients.
4. **Zipfian Memory Hierarchy & Host DRAM / NVMe SSD Offloading in DeepSeek Engram**
   - **Authors:** Xin Cheng et al. (*DeepSeek-AI & Peking University*)
   - **Preprint:** [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) (2026)
   - **Relevance:** Exploits the Zipfian distribution of N-gram accesses and deterministic token-ID addressing to stratify multi-billion-parameter Engram tables across **GPU HBM (Hot Tier)**, **Host CPU DRAM (Warm Tier)**, and **NVMe SSD `mmap` (Cold Tier)** with asynchronous ahead-of-time PCIe prefetching during Prelude layers (`<3%` throughput overhead).

```bibtex
@inproceedings{chi2022representation,
  title={On the Representation Collapse of Sparse Mixture of Experts},
  author={Chi, Zewen and Dong, Li and Huang, Shaohan and Dai, Damai and Ma, Shuming and Patra, Barun and Singhal, Saksham and Bajaj, Payal and Song, Xia and Mao, Xian-Ling and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022},
  url={https://arxiv.org/abs/2204.09179}
}

@article{zoph2022stmoe,
  title={ST-MoE: Designing Stable and Transferable Sparse Expert Models},
  author={Zoph, Barret and Bello, Irwan and Kumar, Sameer and Du, Nan and Huang, Yanping and Dean, Jeff and Shazeer, Noam and Fedus, William},
  journal={arXiv preprint arXiv:2202.08906},
  year={2022},
  url={https://arxiv.org/abs/2202.08906}
}

@article{wang2024auxiliary,
  title={Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts},
  author={Wang, Lean and Gao, Huazuo and Zhao, Chenggang and Sun, Xu and Dai, Damai},
  journal={arXiv preprint arXiv:2408.15664},
  year={2024},
  url={https://arxiv.org/abs/2408.15664}
}
```

---

### 3.2 $O(n \log n)$ Hierarchical Attention & Sub-Quadratic KV-Cache Compression
1. **Log-Linear Attention**
   - **Authors:** Han Guo, Yoon Kim, et al. (*MIT*)
   - **Preprint:** [arXiv:2506.04761](https://arxiv.org/abs/2506.04761) (2025/2026) & *Adaptive Memory Decay for Log-Linear Attention* ([arXiv:2605.06946](https://arxiv.org/abs/2605.06946), 2026)
   - **Relevance:** Replaces the $O(T)$ flat KV cache with a **Fenwick-tree (binary indexed tree) dyadic hierarchy** of $\lceil \log_2 T \rceil$ multiscale summary states per token, reducing full-attention time complexity from $O(T^2)$ to **$O(T \log T)$** and active KV footprint per query to **$O(\log T)$**.
2. **HiP: Hierarchically Pruned Attention ($O(T \log T)$ Attention & KV Offloading)**
   - **Authors:** Heejun Lee et al.
   - **Preprint:** [arXiv:2406.09827](https://arxiv.org/abs/2406.09827) (2024/2025)
   - **Relevance:** Uses hierarchical binary-tree block scoring to select top-$k$ KV blocks in $O(T \log T)$ prefill / $O(\log T)$ decode time while coordinating GPU-to-Host KV cache tiering.
3. **HGA: Hierarchical Global Attention & HiKV: Hierarchical Importance-Aware KV Cache**
   - **Preprints:** [arXiv:2606.30709](https://arxiv.org/abs/2606.30709) (2026) and [arXiv:2607.22389](https://arxiv.org/abs/2607.22389) (2026)
   - **Relevance:** Two-level chunk-summary + fine-token KV routing for long-context NoPE global attention layers.

```bibtex
@article{guo2025loglinear,
  title={Log-Linear Attention},
  author={Guo, Han and Kim, Yoon and others},
  journal={arXiv preprint arXiv:2506.04761},
  year={2025},
  url={https://arxiv.org/abs/2506.04761}
}

@article{lee2024hip,
  title={HiP: Hierarchically Pruned Attention for Long-Context Large Language Models},
  author={Lee, Heejun and others},
  journal={arXiv preprint arXiv:2406.09827},
  year={2024},
  url={https://arxiv.org/abs/2406.09827}
}
```

---

### 3.3 Long-Term Horizon (`Spark-X2.6-4B-En36B-A1B`): Ultra-Scale World-Knowledge Memory Networks & Parametric Knowledge Injection
1. **Memory Layers at Scale (`128B` Product-Key Memory Layers)**
   - **Authors:** Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Ghosh (*Meta FAIR*)
   - **Preprint:** [arXiv:2412.09764](https://arxiv.org/abs/2412.09764) (December 2024) | Code: [`facebookresearch/memory`](https://github.com/facebookresearch/memory)
   - **Relevance:** Demonstrates that scaling product-quantized key-value memory layers up to **128B parameters** on smaller (`1B–8B`) backbones outperforms dense models with **$>2\times$ the compute FLOPs** and fine-grained MoEs, with the largest gains on **factual world-knowledge benchmarks**.
2. **UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning**
   - **Authors:** ByteDance Seed-Foundation-Model Team
   - **Preprint:** [arXiv:2508.18756](https://arxiv.org/abs/2508.18756) (August 2025) & *UltraMem: Ultra-Sparse Memory Network* (*ICLR 2025*, [arXiv:2411.12364](https://arxiv.org/abs/2411.12364))
   - **Relevance:** Scales ultra-sparse Tucker-decomposed product-key memory networks to **120B total parameters with only 2.5B active parameters (`2.1%` active sparsity)** across every transformer block, cutting memory-access cost (MAC) by **83%** vs. standard MoE while improving multi-round memorization (`+6.2 pt`) and in-context learning (`+7.9 pt`).
3. **KBLaM: Knowledge Base Augmented Language Model (Direct Parametric Knowledge Injection)**
   - **Authors:** Microsoft Research
   - **Preprint:** [arXiv:2410.10450](https://arxiv.org/abs/2410.10450) (October 2024)
   - **Relevance:** Encodes external world-knowledge corpora (encyclopedic triples, textbooks, scientific databases) directly into continuous key-value memory tables via frozen sentence encoders + lightweight linear adapters, enabling a `36B` Engram to be **pre-populated with world knowledge offline** without requiring trillions of tokens of from-scratch pretraining.
4. **MemoryLLM & M+: Self-Updatable Long-Term Latent Memory Pools**
   - **Authors:** Yu Wang et al. (*UCSD / MIT / IBM*)
   - **Preprints:** *MemoryLLM* (*ICML 2024*, [arXiv:2402.04624](https://arxiv.org/abs/2402.04624)) & *M+: Extending MemoryLLM with Scalable Long-Term Memory* ([arXiv:2502.00592](https://arxiv.org/abs/2502.00592), February 2025)
   - **Relevance:** Couples a compact Transformer backbone with a self-updatable latent memory pool and co-trained retriever so new world facts can be written into the memory bank continuously without catastrophic forgetting.

```bibtex
@article{berges2024memory,
  title={Memory Layers at Scale},
  author={Berges, Vincent-Pierre and O{\u{g}}uz, Barlas and Haziza, Daniel and Yih, Wen-tau and Zettlemoyer, Luke and Ghosh, Gargi},
  journal={arXiv preprint arXiv:2412.09764},
  year={2024},
  url={https://arxiv.org/abs/2412.09764}
}

@article{bytedance2025ultramemv2,
  title={UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning},
  author={{ByteDance Seed Team}},
  journal={arXiv preprint arXiv:2508.18756},
  year={2025},
  url={https://arxiv.org/abs/2508.18756}
}

@article{microsoft2024kblam,
  title={KBLaM: Knowledge Base Augmented Language Model},
  author={{Microsoft Research}},
  journal={arXiv preprint arXiv:2410.10450},
  year={2024},
  url={https://arxiv.org/abs/2410.10450}
}

@article{wang2025mplus,
  title={M+: Extending MemoryLLM with Scalable Long-Term Memory},
  author={Wang, Yu and others},
  journal={arXiv preprint arXiv:2502.00592},
  year={2025},
  url={https://arxiv.org/abs/2502.00592}
}
```


