"""
Bootstrap corpus: 64 papers on parameter-efficient fine-tuning (PEFT) of language models.

PROVENANCE - READ THIS
  This snapshot was written from the author's knowledge of the literature because the
  environment it was built in could not reach api.semanticscholar.org. Therefore:
    * `summary` is a short PARAPHRASE written for this project, NOT the published abstract.
    * `references` (in-corpus citation edges) are best-effort and have NOT been verified
      against the Semantic Scholar reference lists.
    * Lead-author names and years are from memory (year = first arXiv appearance).
  `fetch_papers.py` regenerates the same file from the real API (real abstracts, real
  reference lists) using data/seed_titles.tsv. Run it, then `python -m pefkg build`.

Running this file writes data/papers_raw.json and data/seed_titles.tsv.
"""
import json, os

P = []
def paper(id, short, title, year, author, summary, refs=()):
    P.append(dict(id=id, short=short, title=title, year=year, authors=[author], summary=summary, references=list(refs)))

# ------------------------------------------------------------------ foundations
paper("transformer", "Transformer", "Attention Is All You Need", 2017, "Ashish Vaswani",
 "Proposes the Transformer, a network architecture built entirely on self-attention that dispenses with recurrence and convolutions. It trains faster than recurrent models and sets new quality records on machine translation.")
paper("ulmfit", "ULMFiT", "Universal Language Model Fine-tuning for Text Classification", 2018, "Jeremy Howard",
 "Proposes ULMFiT, a transfer learning method for NLP that fine-tunes a pretrained language model with discriminative fine-tuning, slanted triangular learning rates and gradual unfreezing. It cuts error on text classification with far less labeled data and limits catastrophic forgetting during fine-tuning.")
paper("bert", "BERT", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018, "Jacob Devlin",
 "Introduces BERT, a language representation model pre-trained with a masked language modeling objective that conditions on both left and right context. A pre-trained BERT fine-tuned with one extra output layer sets new results on GLUE and other language understanding tasks.",
 ["transformer", "ulmfit"])
paper("glue", "GLUE", "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2018, "Alex Wang",
 "Introduces GLUE, a collection of nine sentence-level language understanding tasks with a diagnostic dataset for analyzing models across tasks. Baselines trained with multi-task learning and transfer from pretrained representations show that existing methods still struggle to share knowledge across tasks.")
paper("gpt2", "GPT-2", "Language Models are Unsupervised Multitask Learners", 2019, "Alec Radford",
 "Presents GPT-2, a 1.5 billion parameter Transformer language model trained on a large web text corpus. It performs many tasks in a zero-shot setting without task-specific training, suggesting that scaling language models is a path toward general multitask systems.",
 ["transformer"])
paper("t5", "T5", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2019, "Colin Raffel",
 "Introduces a text-to-text framework that casts every language task as generating text, and systematically compares pre-training objectives, architectures and data. The resulting T5 models, scaled to 11 billion parameters and pre-trained on a large cleaned web corpus, reach strong results on GLUE and SuperGLUE.",
 ["transformer", "bert", "gpt2", "glue", "superglue", "ulmfit"])
paper("superglue", "SuperGLUE", "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems", 2019, "Alex Wang",
 "Introduces SuperGLUE, a harder successor to GLUE with more difficult language understanding tasks, a public leaderboard and human performance baselines, created after models surpassed human-level scores on GLUE.",
 ["glue", "bert"])
paper("scaling_laws", "Scaling laws", "Scaling Laws for Neural Language Models", 2020, "Jared Kaplan",
 "Shows that language model loss follows power-law relationships with model size, dataset size and training compute, and that larger models are more sample-efficient. Derives how to split a fixed compute allowance across these factors.",
 ["transformer", "gpt2"])
paper("gpt3", "GPT-3", "Language Models are Few-Shot Learners", 2020, "Tom B. Brown",
 "Trains GPT-3, an autoregressive language model with 175 billion parameters, and evaluates it in the few-shot setting where tasks are specified through text demonstrations with no gradient updates. It performs strongly on many tasks but still falls short on some, showing the promise of in-context learning.",
 ["transformer", "bert", "gpt2", "t5", "scaling_laws", "glue", "superglue"])
paper("mmlu", "MMLU", "Measuring Massive Multitask Language Understanding", 2020, "Dan Hendrycks",
 "Introduces MMLU, a benchmark of multiple-choice questions across 57 subjects spanning STEM, humanities and social sciences, used to test knowledge acquired during pretraining in zero-shot and few-shot settings. The largest models of the time scored only modestly above chance.",
 ["glue", "superglue", "gpt3", "bert", "t5"])
paper("flan", "FLAN", "Finetuned Language Models Are Zero-Shot Learners", 2021, "Jason Wei",
 "Introduces FLAN, a 137B model produced by instruction tuning on a collection of datasets phrased as natural language instructions. It substantially improves zero-shot performance on unseen task types, with gains that depend on the number of tuning tasks and on model scale.",
 ["gpt3", "t5", "bert"])
paper("instructgpt", "InstructGPT", "Training Language Models to Follow Instructions with Human Feedback", 2022, "Long Ouyang",
 "Introduces InstructGPT, models obtained by fine-tuning GPT-3 with supervised examples and reinforcement learning from human feedback so that outputs follow user intent. Human evaluators prefer outputs from a 1.3B InstructGPT model over the 175B GPT-3, with improvements in truthfulness and fewer toxic outputs.",
 ["gpt3", "flan", "t5"])
paper("flan_scaling", "Flan-T5 scaling", "Scaling Instruction-Finetuned Language Models", 2022, "Hyung Won Chung",
 "Introduces Flan-T5 and Flan-PaLM by scaling instruction finetuning along three axes: the number of tasks, the model size, and finetuning on chain-of-thought data. Flan-T5 and Flan-PaLM improve over their base models on held-out benchmarks such as MMLU and BIG-Bench Hard.",
 ["flan", "t5", "instructgpt", "mmlu"])
paper("chinchilla", "Chinchilla", "Training Compute-Optimal Large Language Models", 2022, "Jordan Hoffmann",
 "Finds that model size and training tokens should be scaled in equal proportion for a given compute allowance, and shows that a 70B model trained on more data outperforms much larger models trained on fewer tokens.",
 ["scaling_laws", "gpt3", "transformer"])
paper("llama", "LLaMA", "LLaMA: Open and Efficient Foundation Language Models", 2023, "Hugo Touvron",
 "Introduces LLaMA, a collection of foundation language models from 7B to 65B parameters trained on publicly available data only. The 13B model outperforms GPT-3 on most benchmarks, and the 65B model is competitive with the best models of its time.",
 ["chinchilla", "scaling_laws", "gpt3", "transformer", "mmlu", "flan_scaling"])
paper("llama2", "Llama 2", "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023, "Hugo Touvron",
 "Releases Llama 2, a family of pretrained and dialogue-tuned language models from 7B to 70B parameters, with a detailed description of the safety-oriented tuning approach. The chat models outperform most open models on helpfulness and safety evaluations.",
 ["llama", "instructgpt", "mmlu", "flan_scaling", "scaling_laws"])
paper("self_instruct", "Self-Instruct", "Self-Instruct: Aligning Language Models with Self-Generated Instructions", 2022, "Yizhong Wang",
 "Proposes a framework in which a pretrained model bootstraps its own instruction data by generating instructions, inputs and outputs, filtering low-quality samples and fine-tuning on the result. Applied to GPT-3, it yields a large improvement on unseen tasks without relying on costly human-written instructions.",
 ["gpt3", "instructgpt", "flan", "t5"])
paper("lima", "LIMA", "LIMA: Less Is More for Alignment", 2023, "Chunting Zhou",
 "Introduces LIMA, a 65B LLaMA model fine-tuned on only 1,000 carefully curated prompts and responses, which obtains strong chat behavior with no further alignment stage. Suggests that almost all knowledge is learned during pretraining and only limited instruction data is needed to teach output style.",
 ["llama", "instructgpt", "self_instruct", "flan_scaling", "flan"])

# ------------------------------------------------------------------ early PEFT
paper("adapters", "Adapters", "Parameter-Efficient Transfer Learning for NLP", 2019, "Neil Houlsby",
 "Proposes adapter modules, small bottleneck layers inserted between the layers of a pretrained BERT model, so that only the adapters are trained while the original weights stay frozen. Tasks can be added sequentially, each costing a tiny fraction of the parameters, and the approach nearly matches full fine-tuning on 26 text classification tasks including GLUE.",
 ["bert", "transformer", "ulmfit", "glue"])
paper("adapterfusion", "AdapterFusion", "AdapterFusion: Non-Destructive Task Composition for Transfer Learning", 2020, "Jonas Pfeiffer",
 "Proposes a two-stage method that first learns adapters for individual tasks and then trains a fusion layer that combines the stored adapters with attention, sharing knowledge across tasks without forgetting earlier ones. It improves over full fine-tuning and single-task adapters on several language understanding benchmarks.",
 ["adapters", "bert", "glue", "superglue", "ewc"])
paper("madx", "MAD-X", "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer", 2020, "Jonas Pfeiffer",
 "Proposes a modular framework that learns language-specific and task-specific adapters plus invertible adapters, which can be stacked to transfer to languages unseen during pretraining. It gains parameter efficiency and outperforms standard transfer on cross-lingual named entity recognition and question answering.",
 ["adapters", "bert"])
paper("intrinsic_dim", "Intrinsic dimension", "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", 2020, "Armen Aghajanyan",
 "Shows that pretrained language models can be fine-tuned effectively inside a random low-dimensional subspace using a very small number of parameters, and that more pretraining lowers the intrinsic dimension needed for a task. Offers an explanation for why fine-tuning with so few parameters generalizes.",
 ["bert", "glue"])
paper("diff_pruning", "Diff pruning", "Parameter-Efficient Transfer Learning with Diff Pruning", 2020, "Demi Guo",
 "Proposes diff pruning, which learns a task-specific sparse difference vector added to fixed pretrained weights, kept sparse with a differentiable L0 penalty. It modifies only about 0.5% of the parameters per task while matching full fine-tuning on the GLUE benchmark.",
 ["adapters", "bert", "glue"])
paper("prefix_tuning", "Prefix-tuning", "Prefix-Tuning: Optimizing Continuous Prompts for Generation", 2021, "Xiang Lisa Li",
 "Proposes prefix-tuning, which keeps the language model frozen and optimizes a small continuous task-specific vector, the prefix, that later tokens attend to like virtual tokens. Tuning 0.1% of the parameters matches full fine-tuning on E2E table-to-text generation with GPT-2 and BART summarization, and does better in low-data settings.",
 ["gpt2", "gpt3", "adapters", "t5", "intrinsic_dim", "bert"])
paper("prompt_tuning", "Prompt tuning", "The Power of Scale for Parameter-Efficient Prompt Tuning", 2021, "Brian Lester",
 "Proposes prompt tuning, which learns soft prompts as extra input embeddings for a frozen T5 model. As models grow past billions of parameters it closes the gap to full fine-tuning, outperforming GPT-3 few-shot prompting on SuperGLUE and letting one frozen model serve many tasks.",
 ["prefix_tuning", "t5", "gpt3", "adapters", "superglue"])
paper("ptuning", "P-tuning", "GPT Understands, Too", 2021, "Xiao Liu",
 "Proposes P-tuning, which trains continuous prompt embeddings produced by a small prompt encoder and inserts them among discrete prompts, since discrete prompts are brittle and unstable. It improves GPT-style models on knowledge probing and SuperGLUE in fully supervised and few-shot settings, narrowing the gap with similarly sized BERT models.",
 ["gpt3", "gpt2", "bert", "prefix_tuning", "superglue"])
paper("ptuning_v2", "P-tuning v2", "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks", 2021, "Xiao Liu",
 "Proposes P-Tuning v2, which applies deep prompt tuning, continuous prompts at every layer instead of only the input, and matches full fine-tuning from 300M to 10B parameters and on hard sequence-labeling tasks, where earlier prompt tuning lagged behind full fine-tuning.",
 ["prefix_tuning", "prompt_tuning", "ptuning", "bert", "superglue"])
paper("bitfit", "BitFit", "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models", 2021, "Elad Ben Zaken",
 "Proposes BitFit, which updates only the bias terms of a pretrained BERT model, roughly 0.1% of its parameters. It matches or comes close to full fine-tuning on GLUE for small and medium data and is competitive with other sparse approaches, suggesting that fine-tuning mainly exposes existing knowledge.",
 ["bert", "adapters", "diff_pruning", "intrinsic_dim", "glue"])
paper("compacter", "Compacter", "Compacter: Efficient Low-Rank Hypercomplex Adapter Layers", 2021, "Rabeeh Karimi Mahabadi",
 "Proposes Compacter, which builds adapter layers from Kronecker products of shared slow weights and low-rank fast weights using hypercomplex multiplication. It trains only about 0.05% of the parameters of a T5 model while outperforming standard adapters on GLUE and SuperGLUE.",
 ["adapters", "t5", "glue", "superglue", "intrinsic_dim", "diff_pruning", "prefix_tuning", "adapterfusion"])
paper("lora", "LoRA", "LoRA: Low-Rank Adaptation of Large Language Models", 2021, "Edward J. Hu",
 "Proposes LoRA, which freezes the pretrained weights and injects trainable low-rank decomposition matrices into each Transformer layer. The design rests on the observation that weight updates during adaptation have a low intrinsic rank. Compared to fine-tuning GPT-3 175B with Adam it cuts trainable parameters by 10,000 times and GPU memory by three times, unlike adapter layers it adds no inference latency, and it performs on par with or better than full fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3.",
 ["adapters", "prefix_tuning", "intrinsic_dim", "gpt3", "gpt2", "bert", "transformer", "glue", "bitfit"])
paper("unified_view", "Unified view of PETL", "Towards a Unified View of Parameter-Efficient Transfer Learning", 2021, "Junxian He",
 "Proposes a unified framework that casts adapters, prefix tuning and LoRA as modifications to specific hidden states of a frozen model, revealing design dimensions such as the functional form, the insertion position and the composition. It uses the framework to derive new variants that match full fine-tuning with 0.5% of the parameters while outperforming existing methods on translation, summarization and understanding tasks.",
 ["adapters", "prefix_tuning", "lora", "bitfit", "prompt_tuning", "t5", "bert", "compacter", "intrinsic_dim", "diff_pruning"])
paper("unipelt", "UniPELT", "UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning", 2021, "Yuning Mao",
 "Proposes a unified framework that incorporates adapters, prefix tuning and LoRA as submodules, activated through a gating mechanism that learns which to use for a given task and setup. It outperforms its best individual submodule and full fine-tuning on GLUE, particularly with little training data.",
 ["adapters", "prefix_tuning", "lora", "bitfit", "bert", "glue", "adapterfusion"])
paper("spot", "SPoT", "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer", 2021, "Tu Vu",
 "Proposes soft prompt transfer, which first learns a prompt on source tasks and uses it to initialize the prompt for a target task on a frozen T5 model. It substantially improves prompt tuning on SuperGLUE, matches or beats full fine-tuning across model sizes, and allows predicting which source tasks transfer well.",
 ["prompt_tuning", "t5", "superglue", "prefix_tuning", "glue"])
paper("t_few", "T-Few", "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022, "Haokun Liu",
 "Proposes T-Few, a recipe built on (IA)3, which rescales inner activations with learned vectors and adds only a tiny number of new parameters, together with new loss terms and no task-specific tuning. T-Few beats in-context learning with GPT-3 and reaches super-human performance on a few-shot benchmark at far lower inference cost.",
 ["t5", "gpt3", "lora", "prefix_tuning", "prompt_tuning", "bitfit", "adapters", "flan", "superglue", "compacter"])
paper("delta_tuning", "Delta tuning", "Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models", 2022, "Ning Ding",
 "Surveys parameter-efficient methods under the name delta tuning and groups them into addition-based, specification-based and reparameterization-based families, covering adapters, prompt tuning, prefix-tuning, BitFit and LoRA. It offers theoretical views through optimization and optimal control and evaluates over 100 NLP tasks.",
 ["adapters", "prefix_tuning", "prompt_tuning", "lora", "bitfit", "intrinsic_dim", "unified_view", "compacter", "diff_pruning", "bert", "t5", "gpt3", "adapterfusion", "unipelt", "ptuning_v2", "spot"])
paper("adamix", "AdaMix", "AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning", 2022, "Yaqing Wang",
 "Proposes mixture-of-adaptations, in which several adaptation modules are placed in each layer and randomly routed during training, then merged by weight averaging for inference so no extra cost is paid. The modules can be adapters or LoRA. It outperforms full fine-tuning on GLUE while tuning 0.1-0.2% of parameters.",
 ["adapters", "lora", "prefix_tuning", "bitfit", "bert", "glue", "compacter", "unified_view", "unipelt", "prompt_tuning"])
paper("dylora", "DyLoRA", "DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation", 2022, "Mojtaba Valipour",
 "Proposes training LoRA blocks for a range of ranks instead of a single fixed rank, by sorting the representation learned at different ranks during training. It removes the need to search for the best rank and runs 4 to 7 times faster than LoRA while keeping competitive quality.",
 ["lora", "adapters", "bitfit", "prefix_tuning", "glue", "unipelt"])
paper("adalora", "AdaLoRA", "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", 2023, "Qingru Zhang",
 "Proposes AdaLoRA, which parameterizes incremental updates in singular value decomposition form and prunes the singular values of unimportant updates, so the rank budget is allocated adaptively across weight matrices according to an importance score. It gives gains over LoRA and other baselines under low parameter budgets on GLUE, question answering and summarization.",
 ["lora", "adapters", "bitfit", "diff_pruning", "compacter", "unipelt", "bert", "glue", "intrinsic_dim"])
paper("llama_adapter", "LLaMA-Adapter", "LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention", 2023, "Renrui Zhang",
 "Proposes a method that prepends a set of learnable adaptation prompts to the upper Transformer layers of a frozen LLaMA 7B and uses zero-init attention to inject instruction signals stably. It turns LLaMA into an instruction-following model with 1.2M trainable parameters in under one hour.",
 ["llama", "adapters", "lora", "prefix_tuning", "prompt_tuning", "self_instruct", "instructgpt", "mmlu"])
paper("scaling_down", "Scaling Down to Scale Up", "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning", 2023, "Vladislav Lialin",
 "Surveys more than forty parameter-efficient fine-tuning methods and organizes them into additive, selective, reparameterization-based and hybrid families, with a comparison of their storage, memory and computation costs. It reports experiments on models up to 11B parameters and discusses open challenges.",
 ["adapters", "prefix_tuning", "prompt_tuning", "lora", "bitfit", "adalora", "unified_view", "unipelt", "t_few", "compacter", "ptuning", "ptuning_v2", "intrinsic_dim", "madx", "adapterfusion", "diff_pruning", "delta_tuning", "llama", "llama_adapter", "mixed_precision", "grad_checkpoint", "eightbit_optim", "llm_int8", "gptq", "kbit_scaling", "adamix", "spot", "dylora", "flan", "instructgpt", "gpt3", "scaling_laws"])
paper("peft_survey", "PEFT survey (2024)", "Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey", 2024, "Zeyu Han",
 "Surveys PEFT algorithms for large models, covering additive, selective, reparameterized and hybrid methods, plus systems and applications in scenarios such as multi-task serving and distributed tuning. It discusses the computation and memory costs of adapting very large models.",
 ["adapters", "prefix_tuning", "prompt_tuning", "lora", "bitfit", "adalora", "unipelt", "unified_view", "t_few", "compacter", "delta_tuning", "scaling_down", "qlora", "dora", "vera", "lora_fa", "longlora", "lorahub", "loramoe", "s_lora", "relora", "galore", "llama_adapter", "adamix", "dylora", "madx", "adapterfusion", "gptq", "llm_int8", "loftq", "qa_lora"])

# ------------------------------------------------------------------ LoRA family (2023-24)
paper("vera", "VeRA", "VeRA: Vector-based Random Matrix Adaptation", 2023, "Dawid Kopiczko",
 "Proposes VeRA, which freezes a single pair of random low-rank matrices shared across all layers and trains only small scaling vectors per layer. It cuts trainable parameters by 10 times compared to LoRA. Performance is kept on GLUE, E2E and instruction tuning of LLaMA models.",
 ["lora", "adalora", "adapters", "bitfit", "intrinsic_dim", "llama", "llama2", "glue"])
paper("lora_fa", "LoRA-FA", "LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning", 2023, "Longteng Zhang",
 "Proposes LoRA-FA, which freezes the down-projection matrix of each LoRA layer and updates only the up-projection, so the activations needed for backpropagation no longer need to be stored in full. It reduces activation memory by up to 1.4 times with no change in accuracy and without expensive recomputation or quantization.",
 ["lora", "adalora", "adapters", "llama", "grad_checkpoint", "mixed_precision"])
paper("longlora", "LongLoRA", "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models", 2023, "Yukang Chen",
 "Proposes shifted sparse attention, which approximates full attention during fine-tuning with sparse local attention over shifted groups, together with an improved LoRA that also trains embeddings and normalization layers. Full fine-tuning at long context lengths is prohibitively expensive, and this extends LLaMA 2 7B to 100k tokens and the 70B model to 32k on a single machine.",
 ["lora", "llama", "llama2", "transformer", "gpt3"])
paper("lorahub", "LoraHub", "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition", 2023, "Chengsong Huang",
 "Proposes LoraHub, a framework that composes LoRA modules trained on different upstream tasks using a gradient-free search over a few examples of an unseen task, with no extra parameters or gradient steps. On Big-Bench Hard it approaches few-shot in-context learning at lower inference cost.",
 ["lora", "flan", "flan_scaling", "t5", "t_few", "gpt3", "task_arithmetic", "adapterfusion", "adapters"])
paper("loramoe", "LoRAMoE", "LoRAMoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin", 2023, "Shihan Dou",
 "Proposes LoRAMoE, which integrates several LoRA experts with a router network and a localized balancing constraint so that some experts preserve world knowledge while others handle downstream tasks. It reduces forgetting of world knowledge when scaling up instruction data in supervised fine-tuning of LLaMA-2 models.",
 ["lora", "llama2", "llama", "instructgpt", "adamix", "flan_scaling"])
paper("s_lora", "S-LoRA", "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", 2023, "Ying Sheng",
 "Presents a system for serving many LoRA adapters that keeps all of them in main memory, fetches those needed for active queries into GPU memory, and uses unified paging and custom kernels for heterogeneous batching. It improves throughput by up to 4 times over naive serving of LoRA weights and increases the number of concurrently served LoRA models by orders of magnitude.",
 ["lora", "llama", "llama2", "transformer", "gpt3", "adapters"])
paper("relora", "ReLoRA", "ReLoRA: High-Rank Training Through Low-Rank Updates", 2023, "Vladislav Lialin",
 "Proposes ReLoRA, which trains a high-rank network through repeated low-rank updates, merging them into the weights and restarting the optimizer and learning-rate schedule. Pre-training large networks from scratch is expensive, and applied to Transformers up to 1.3B parameters it matches regular training while saving memory.",
 ["lora", "intrinsic_dim", "scaling_laws", "chinchilla", "llama", "transformer", "gpt3", "adapters", "scaling_down"])
paper("galore", "GaLore", "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", 2024, "Jiawei Zhao",
 "Proposes GaLore, a training strategy that projects gradients onto a low-rank subspace so that optimizer states take far less memory while all weights are still learned. Gradients become low-rank during training, which makes this possible. It cuts optimizer-state memory by up to 65%, allows memory-efficient pre-training from scratch of a 7B model on a single consumer 24GB GPU, and works with 8-bit optimizers. Unlike LoRA, it does not restrict the weights to a low-rank parameterization.",
 ["lora", "relora", "llama", "qlora", "eightbit_optim", "glue", "t5", "bert", "transformer", "intrinsic_dim", "adalora", "lora_fa", "grad_checkpoint"])
paper("dora", "DoRA", "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024, "Shih-Yang Liu",
 "Proposes DoRA, which decomposes pretrained weights into magnitude and direction and fine-tunes both, applying LoRA only to the directional component. It narrows the gap between LoRA and full fine-tuning and consistently outperforms LoRA on commonsense reasoning and LLaMA models without adding inference overhead.",
 ["lora", "vera", "adalora", "adapters", "prefix_tuning", "bitfit", "llama", "llama2", "qlora"])
paper("lora_plus", "LoRA+", "LoRA+: Efficient Low Rank Adaptation of Large Models", 2024, "Soufiane Hayou",
 "Shows that using the same learning rate for the two LoRA matrices is suboptimal for wide models and proposes setting different learning rates for them, which improves feature learning. It gains up to 2% accuracy and about 2x speedup over standard LoRA at the same computational cost.",
 ["lora", "adalora", "glue", "llama", "gpt3", "bert", "transformer"])
paper("pissa", "PiSSA", "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models", 2024, "Fanxu Meng",
 "Proposes PiSSA, which initializes the low-rank matrices with the principal singular values and vectors of the pretrained weight matrix and freezes the residual components, so training converges faster and fits better than LoRA. It supports quantized backbones with lower quantization error than QLoRA.",
 ["lora", "qlora", "loftq", "adalora", "vera", "dora", "llama2", "llama", "intrinsic_dim"])
paper("lora_less_forgets", "LoRA Learns Less, Forgets Less", "LoRA Learns Less and Forgets Less", 2024, "Dan Biderman",
 "Studies how LoRA compares with full fine-tuning on code and mathematics for continued pretraining and instruction fine-tuning of Llama-2 models. LoRA substantially underperforms full fine-tuning but also forgets less of the source domain and keeps more diverse generations, and full fine-tuning learns perturbations with a rank 10 to 100 times greater than typical LoRA configurations.",
 ["lora", "qlora", "dora", "galore", "lora_plus", "vera", "relora", "adalora", "llama2", "intrinsic_dim", "ewc"])

# ------------------------------------------------------------------ quantization / memory
paper("ewc", "EWC", "Overcoming Catastrophic Forgetting in Neural Networks", 2016, "James Kirkpatrick",
 "Proposes elastic weight consolidation, which slows learning on weights that were important for earlier tasks through a quadratic penalty, allowing a network to learn tasks sequentially without forgetting old ones. It is shown on supervised MNIST tasks and sequential Atari 2600 games.")
paper("grad_checkpoint", "Gradient checkpointing", "Training Deep Nets with Sublinear Memory Cost", 2016, "Tianqi Chen",
 "Proposes a scheme that drops intermediate activations during the forward pass and recomputes them during backpropagation, reducing memory for an n-layer network to O(sqrt(n)) at the price of one extra forward pass.")
paper("mixed_precision", "Mixed precision", "Mixed Precision Training", 2017, "Paulius Micikevicius",
 "Presents a methodology for training deep networks with half-precision floating point numbers by keeping a single-precision master copy of weights and applying loss scaling to preserve small gradient values. It halves memory use with no loss of accuracy across convolutional, recurrent and generative models.")
paper("eightbit_optim", "8-bit optimizers", "8-bit Optimizers via Block-wise Quantization", 2021, "Tim Dettmers",
 "Introduces 8-bit optimizers that quantize optimizer states block by block with a dynamic quantization scheme and a stable embedding layer, maintaining 32-bit performance on language modeling, translation and pretraining while cutting optimizer memory by 75%.",
 ["mixed_precision", "transformer", "bert", "glue"])
paper("llm_int8", "LLM.int8()", "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022, "Tim Dettmers",
 "Introduces a two-part 8-bit quantization procedure with vector-wise quantization and a mixed-precision decomposition that isolates emergent outlier feature dimensions in 16-bit. It enables inference of models up to 175B parameters, such as OPT, with no loss of performance and halves the memory required.",
 ["transformer", "bert", "gpt3", "gpt2", "t5", "eightbit_optim", "scaling_laws"])
paper("kbit_scaling", "k-bit scaling laws", "The Case for 4-bit Precision: k-bit Inference Scaling Laws", 2022, "Tim Dettmers",
 "Studies the trade-off between model size and bit precision across thousands of inference experiments with OPT, BLOOM, GPT-2 and Pythia, and finds that 4-bit precision is almost universally optimal for total model bits and accuracy. It examines data types and block sizes that improve scaling.",
 ["llm_int8", "gptq", "eightbit_optim", "scaling_laws", "gpt2", "gpt3", "chinchilla", "t5", "transformer"])
paper("gptq", "GPTQ", "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022, "Elias Frantar",
 "Proposes GPTQ, a one-shot weight quantization method based on approximate second-order information that compresses GPT models with hundreds of billions of parameters to 3 or 4 bits per weight with negligible accuracy loss in a few GPU hours. It enables running a 175B-parameter model on a single GPU.",
 ["llm_int8", "transformer", "gpt3", "gpt2", "bert"])
paper("awq", "AWQ", "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023, "Ji Lin",
 "Proposes activation-aware weight quantization, which protects the small fraction of salient weights identified from activation statistics by scaling channels before quantizing, without backpropagation or reconstruction. It keeps small accuracy loss across language and multimodal models and ships with an inference framework that reduces inference latency on edge devices.",
 ["gptq", "llm_int8", "llama", "transformer", "gpt3"])
paper("qlora", "QLoRA", "QLoRA: Efficient Finetuning of Quantized LLMs", 2023, "Tim Dettmers",
 "Introduces QLoRA, which combines a 4-bit NormalFloat data type, double quantization of the quantization constants and paged optimizers to fine-tune large models on a single GPU. Gradients flow through the frozen quantized weights into LoRA adapters, so a 65B LLaMA model can be fine-tuned on one 48GB GPU while preserving 16-bit fine-tuning quality, and the resulting Guanaco models approach ChatGPT on the Vicuna benchmark.",
 ["lora", "llm_int8", "kbit_scaling", "gptq", "eightbit_optim", "llama", "mmlu", "lima", "self_instruct", "flan", "flan_scaling", "instructgpt", "grad_checkpoint", "adapters", "prefix_tuning", "prompt_tuning", "gpt3", "t5", "intrinsic_dim"])
paper("qa_lora", "QA-LoRA", "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models", 2023, "Yuhui Xu",
 "Proposes quantization-aware low-rank adaptation, using group-wise operators so the quantized weights and the low-rank update share a form that can be merged after fine-tuning without falling back to 16-bit precision. It keeps the model low-bit, lowers inference latency compared with QLoRA, and reaches comparable or better accuracy on LLaMA and LLaMA2 without the accuracy loss of post-training quantization.",
 ["qlora", "lora", "gptq", "awq", "llm_int8", "llama", "llama2", "mmlu", "flan_scaling", "self_instruct", "kbit_scaling"])
paper("loftq", "LoftQ", "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models", 2023, "Yixiao Li",
 "Proposes LoftQ, which quantizes a pretrained model while simultaneously finding a good low-rank initialization of the LoRA matrices that compensates for the quantization error. It improves downstream results over QLoRA, especially in low-bit regimes such as 2-bit where the zero initialization of QLoRA fails.",
 ["lora", "qlora", "gptq", "awq", "llm_int8", "kbit_scaling", "adalora", "llama2", "llama", "glue", "bert"])
paper("task_arithmetic", "Task arithmetic", "Editing Models with Task Arithmetic", 2022, "Gabriel Ilharco",
 "Proposes task vectors, obtained by subtracting pretrained weights from fine-tuned weights, and shows that adding, negating or combining them steers model behavior, for example removing a skill or building multi-task models, without further training. It works across vision and language models such as GPT-2 and T5.",
 ["gpt2", "t5", "glue", "gpt3", "bert"])

# ------------------------------------------------------------------ write
ids = {p["id"] for p in P}
assert len(ids) == len(P), "duplicate ids"
byid = {p["id"]: p for p in P}
for p in P:
    for r in p["references"]:
        assert r in ids, (p["id"], "unknown ref", r)
        assert byid[r]["year"] <= p["year"], (p["id"], p["year"], "cites later paper", r, byid[r]["year"])
    p["abstract"] = p.pop("summary")
    p["abstract_source"] = "hand_written_paraphrase"
    p["reference_source"] = "curated_from_memory_unverified"

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "papers_raw.json"), "w") as f:
    json.dump({"topic": "Parameter-efficient fine-tuning (PEFT) of language models",
               "source": "bootstrap_corpus.py (hand-curated; regenerate with fetch_papers.py)",
               "papers": P}, f, indent=1)
with open(os.path.join(here, "seed_titles.tsv"), "w") as f:
    f.write("id\tshort\tyear\ttitle\n")
    for p in P:
        f.write(f"{p['id']}\t{p['short']}\t{p['year']}\t{p['title']}\n")
print(len(P), "papers;", sum(len(p["references"]) for p in P), "in-corpus reference edges")
