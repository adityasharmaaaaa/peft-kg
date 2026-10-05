"""
Ontology for the PEFT (parameter-efficient fine-tuning) research-onboarding knowledge graph.

Everything in this file is hand-authored: the entity types, the relation schema, the
mechanism taxonomy and the lexicon that maps raw text to entities. There is no library
that extracts entities or relations automatically; `extract.py` simply applies the
patterns below and `build.py` applies the relation rules.

Design principle: a paper is not a bag of words, it is a *signature* over a small number
of orthogonal dimensions:

    MECHANISM  - what technical move the paper makes (taxonomy, 2 levels)
    PROBLEM    - which obstacle it says it is attacking
    CONCEPT    - which idea it rests on / establishes
    MODEL      - which backbone family it is built or evaluated on
    BENCHMARK  - what it is evaluated on

Papers that share a signature are related in a way citations alone do not tell you.
"""

SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------------------
# Entity types
# --------------------------------------------------------------------------------------
ENTITY_TYPES = {
    "Paper": "A publication in the corpus. Carries year, lead author, contribution type and derived scores.",
    "Mechanism": "A concrete technical move (e.g. low-rank weight update, soft prompt, 4-bit quantization).",
    "MechanismFamily": "Coarser grouping of mechanisms by *where in the model / training stack* they act.",
    "Problem": "An obstacle a paper says it addresses (e.g. training memory, inference latency, forgetting).",
    "Concept": "An idea that other work rests on (e.g. intrinsic dimensionality, scaling laws).",
    "Model": "A backbone family used as the thing being adapted (BERT, T5, GPT-3, LLaMA ...).",
    "Benchmark": "A dataset or suite on which methods are evaluated.",
    "Author": "Lead author of a paper (the bootstrap corpus stores lead authors only).",
}

# --------------------------------------------------------------------------------------
# Relation types (with the rule that creates each one - documented in approach.md)
# --------------------------------------------------------------------------------------
RELATION_TYPES = {
    # paper -> entity
    "PROPOSES": "Paper -> Mechanism. The mechanism is part of the paper's own contribution "
                "(named in the title or in the first 'we propose/introduce' sentence).",
    "USES": "Paper -> Mechanism. Mechanism is used as a component / substrate but is not the contribution.",
    "COVERS": "Survey -> Mechanism. A survey describes the mechanism; it neither proposes nor uses it.",
    "COMPARES_WITH": "Paper -> Mechanism. Mechanism appears only in a contrastive sentence "
                     "(unlike / compared with / outperforms / without ...), i.e. as a baseline or foil.",
    "ADDRESSES": "Paper -> Problem. The problem is named as motivation or as a measured outcome.",
    "ESTABLISHES": "Paper -> Concept. The paper introduces or demonstrates the concept.",
    "RELIES_ON": "Paper -> Concept. The paper assumes or motivates itself with the concept.",
    "INTRODUCES_MODEL": "Paper -> Model. Foundation-model paper that introduces the model family.",
    "USES_MODEL": "Paper -> Model. Model family is adapted or evaluated in the paper.",
    "INTRODUCES_BENCHMARK": "Paper -> Benchmark. Benchmark paper.",
    "EVALUATED_ON": "Paper -> Benchmark. Paper reports results on the benchmark.",
    "AUTHORED_BY": "Paper -> Author (lead author).",
    # paper -> paper (typed citation; direction = newer -> older)
    "EXTENDS": "A refines a mechanism that B also proposes (shared PROPOSES). Strong prerequisite.",
    "BUILDS_ON": "A uses a mechanism that B proposes as a component/substrate. Strong prerequisite.",
    "COMPARES_WITH_PAPER": "A positions itself against the mechanism B proposes (baseline / alternative). "
                           "Not a prerequisite; a 'must compare' pointer.",
    "MOTIVATED_BY": "A rests on a concept B establishes, or on B's empirical analysis of a mechanism A uses. Prerequisite.",
    "USES_BACKBONE": "A adapts / evaluates on a model family B introduces. Prerequisite for beginners only.",
    "USES_BENCHMARK": "A evaluates on a benchmark B introduces. Reference only.",
    "SURVEYS": "A is a survey and B is one of the works it covers. Never a prerequisite.",
    "BACKGROUND": "Citation with no structural link in the schema. Weakest edge.",
    # entity -> entity (taxonomy)
    "SUBTYPE_OF": "Mechanism -> MechanismFamily.",
    "UNDERPINS": "Concept -> Mechanism. Hand-authored background knowledge (see CONCEPT_UNDERPINS).",
}

# Paper->paper relations that mean "you should understand B before A"
PREREQ_STRONG = {"EXTENDS", "BUILDS_ON", "MOTIVATED_BY"}
PREREQ_FOUNDATION = {"USES_BACKBONE", "BACKGROUND"}   # only added for beginners, only for foundation-tier B

# --------------------------------------------------------------------------------------
# Mechanism taxonomy
# --------------------------------------------------------------------------------------
MECHANISM_FAMILIES = {
    "ADDITIVE": "Inject new trainable parameters; the backbone stays frozen.",
    "SELECTIVE": "Train only a chosen subset of the existing parameters.",
    "REPARAMETERIZATION": "Express the weight update in a cheaper form (low-rank, shared, decomposed ...).",
    "COMPRESSION": "Shrink the numeric representation of the model.",
    "COMPOSITION": "Combine, route or transfer between separately trained modules.",
    "TRAINING_RECIPE_AND_SYSTEMS": "Change how training / serving is executed (memory, optimizer, schedule, kernels) "
                                   "rather than what is learned.",
    "SUPERVISION": "Change the data or objective used to adapt a model.",
    "HYBRID": "Unify or jointly use several families.",
}

# Each mechanism: label, family, description, core patterns, derived patterns.
#   core    -> counts as the mechanism proper (can be PROPOSES / USES / COMPARES_WITH)
#   derived -> a *named variant* of the mechanism (QLoRA, AdaLoRA, S-LoRA ...). Never
#              counts as PROPOSES: a variant name tells us the paper builds on the base
#              mechanism, not that it introduces it.
# All patterns run on lower-cased text.
MECHANISMS = {
    # ---- ADDITIVE -------------------------------------------------------------------
    "adapter_modules": dict(
        label="Adapter modules", family="ADDITIVE",
        desc="Small bottleneck modules inserted between frozen Transformer layers.",
        core=[r"(?<![\w-])(?<!rank )(?<!lora )adapters?(?!\w)", r"bottleneck (layer|module)s?"]),
    "soft_prompt": dict(
        label="Soft / prefix prompts", family="ADDITIVE",
        desc="Learned continuous vectors prepended to the input or to every layer.",
        core=[r"prefix[- ]tuning", r"prompt[- ]tuning", r"soft prompts?", r"continuous prompts?",
              r"(?<![\w-])p-tuning", r"(learnable|trainable|adaptation) prompts?", r"deep prompt"]),
    "activation_rescaling": dict(
        label="Learned activation rescaling", family="ADDITIVE",
        desc="Learned vectors that elementwise rescale internal activations ((IA)^3).",
        core=[r"\(ia\)\s?\^?3", r"\bia3\b", r"rescal\w+", r"inhibiting and amplifying"]),
    "zero_init_gating": dict(
        label="Zero-initialised gated injection", family="ADDITIVE",
        desc="Inject new signal through a gate initialised at zero so the pretrained behaviour is preserved at step 0.",
        core=[r"zero[- ]init\w* attention", r"zero[- ]gating", r"zero[- ]initiali[sz]ed gat\w+"]),
    # ---- SELECTIVE ------------------------------------------------------------------
    "bias_only": dict(
        label="Bias-only tuning", family="SELECTIVE",
        desc="Update only the bias terms.",
        core=[r"bias[- ]terms?", r"bitfit", r"only (the )?bias"]),
    "sparse_diff": dict(
        label="Sparse difference vector", family="SELECTIVE",
        desc="Learn a sparse task-specific delta over frozen weights.",
        core=[r"diff pruning", r"sparse (difference|diff|task-specific|update)s?", r"\bl0\b"]),
    "gradual_unfreezing": dict(
        label="Gradual unfreezing / discriminative LR", family="SELECTIVE",
        desc="Unfreeze layers progressively with layer-wise learning rates.",
        core=[r"gradual(ly)? unfreez\w+", r"discriminative fine-?tuning", r"slanted triangular"]),
    # ---- REPARAMETERIZATION ---------------------------------------------------------
    "low_rank_update": dict(
        label="Low-rank weight update (LoRA)", family="REPARAMETERIZATION",
        desc="Represent the weight update as a product of two thin matrices.",
        core=[r"(?<![\w-])lora(?!\w)",
              r"low[- ]rank (update|adaptation|decomposition|matrices|matrix|parameteri[sz]ation|structure|adapters?|weight)s?"],
        derived=[r"\b\w+-?lora\b"]),
    "adaptive_rank": dict(
        label="Adaptive rank / budget allocation", family="REPARAMETERIZATION",
        desc="Choose rank or parameter budget per layer/matrix instead of a uniform fixed rank.",
        core=[r"adaptive(ly)? (allocat\w+|rank|budget)", r"allocat\w+ adaptively", r"(per|each)[- ]layer rank", r"layer-?wise rank", r"budget allocation", r"rank (allocation|budget|selection|search)",
              r"dynamic(ally)? (rank|search)", r"range of ranks?", r"search\w* for (the )?(best|optimal) rank",
              r"prun\w+[^.]{0,40}singular values?", r"importance score"]),
    "weight_decomposition": dict(
        label="Magnitude / direction decomposition", family="REPARAMETERIZATION",
        desc="Split pretrained weights into magnitude and direction and adapt them separately.",
        core=[r"weight[- ]decompos\w+", r"magnitude and direction"]),
    "lora_init": dict(
        label="Informed adapter initialisation", family="REPARAMETERIZATION",
        desc="Initialise the low-rank factors from the pretrained weights (SVD) or to absorb quantization error.",
        core=[r"initiali[sz]\w*[^.]{0,80}(singular|svd|quantization error|principal)", r"principal singular"]),
    "parameter_sharing": dict(
        label="Parameter sharing / structured factors", family="REPARAMETERIZATION",
        desc="Share or factorise adapter parameters across layers (Kronecker, hypercomplex, shared random matrices).",
        core=[r"kronecker", r"hypercomplex", r"shared across (all )?layers", r"shared (slow )?weights?"]),
    "lr_asymmetry": dict(
        label="Asymmetric learning rates", family="REPARAMETERIZATION",
        desc="Use different learning rates for the two low-rank factors.",
        core=[r"(different|separate|distinct) learning[- ]rates?", r"learning[- ]rate ratio"]),
    "frozen_projection": dict(
        label="Frozen projection matrix", family="REPARAMETERIZATION",
        desc="Freeze one of the low-rank factors to avoid storing activations.",
        core=[r"freez\w+ (the )?(randomly[- ]initiali[sz]ed )?(down[- ]?projection|projection[- ]down|projection)"]),
    "periodic_merge_restart": dict(
        label="Periodic merge-and-restart", family="REPARAMETERIZATION",
        desc="Merge low-rank updates into the weights and restart, accumulating a high-rank change.",
        core=[r"merg\w+[^.]{0,40}into (the )?(\w+ )?weights", r"restart\w*"]),
    # ---- COMPRESSION ----------------------------------------------------------------
    "quantization": dict(
        label="Quantization", family="COMPRESSION",
        desc="Represent weights / states in 8-, 4- or fewer bits.",
        core=[r"quantiz\w+", r"\b[2348]-bit", r"\bint8\b", r"\bnf4\b", r"\bk-bit"]),
    # ---- COMPOSITION ----------------------------------------------------------------
    "module_composition": dict(
        label="Module composition / merging", family="COMPOSITION",
        desc="Combine separately trained adapters, LoRAs or weights (fusion, task arithmetic, averaging).",
        core=[r"adapter ?fusion", r"lora (composition|merging|fusion|arithmetic)",
              r"(fus\w+|combin\w+|compos\w+|merg\w+)\s+(of\s+)?(the\s+)?(multiple\s+|several\s+|many\s+|different\s+)?"
              r"(trained\s+|pre-?trained\s+|task-specific\s+)?(lora|adapter|module|expert)s?",
              r"task arithmetic", r"task vectors?", r"model merging", r"weight (averaging|interpolation)"]),
    "expert_routing": dict(
        label="Mixture-of-experts routing", family="COMPOSITION",
        desc="Route inputs or training steps across several adaptation modules.",
        core=[r"mixture[- ]of[- ]\w+", r"\brout(er|ing|ed)\b", r"\bmoe\b|\w+moe\b"]),
    "prompt_transfer": dict(
        label="Prompt transfer", family="COMPOSITION",
        desc="Initialise a target-task prompt from one learned on source tasks.",
        core=[r"transfer\w* (soft )?prompts?", r"prompt transfer"]),
    # ---- TRAINING RECIPE & SYSTEMS --------------------------------------------------
    "low_rank_gradient": dict(
        label="Low-rank gradient projection", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Project gradients onto a low-rank subspace to shrink optimizer state while training all weights.",
        core=[r"gradient (low[- ]rank )?projection", r"project\w* (the )?gradients?", r"low[- ]rank (gradient|subspace)s?"]),
    "optimizer_state_compression": dict(
        label="Optimizer-state compression", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Shrink or page the optimizer states (8-bit optimizers, paged optimizers, projected states).",
        core=[r"optimizer states?", r"paged optimizers?", r"\b[48]-bit optimizers?"]),
    "activation_recomputation": dict(
        label="Gradient checkpointing", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Drop activations in the forward pass and recompute them in the backward pass.",
        core=[r"gradient checkpoint\w*", r"recomput\w+", r"sublinear memory"]),
    "mixed_precision": dict(
        label="Mixed-precision training", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Compute in half precision with a full-precision master copy.",
        core=[r"mixed[- ]precision", r"half[- ]precision", r"\bfp16\b", r"\bbf16\b"]),
    "sparse_attention": dict(
        label="Sparse / shifted attention", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Approximate full attention with sparse patterns during adaptation to reach long contexts.",
        core=[r"shifted sparse attention", r"sparse (local )?attention"]),
    "serving_system": dict(
        label="Multi-adapter serving system", family="TRAINING_RECIPE_AND_SYSTEMS",
        desc="Memory management and batching so thousands of adapters can be served from one base model.",
        core=[r"serving (thousands|many|multiple|concurrent)", r"serving system", r"unified paging", r"custom (cuda )?kernels?"]),
    # ---- SUPERVISION ----------------------------------------------------------------
    "instruction_tuning": dict(
        label="Instruction tuning", family="SUPERVISION",
        desc="Fine-tune on tasks phrased as natural-language instructions.",
        core=[r"instruction[- ]?(tun|finetun|fine-tun|follow)\w*", r"follow\w* (user |human |natural language )?instructions?",
              r"supervised fine-?tuning", r"\bsft\b"]),
    "synthetic_instructions": dict(
        label="Self-generated instruction data", family="SUPERVISION",
        desc="Have a model bootstrap its own instruction-tuning data.",
        core=[r"self-?(instruct|generated instructions?)", r"bootstrap\w* its own instruction"]),
    "curated_data": dict(
        label="Small curated data", family="SUPERVISION",
        desc="Replace volume with a few carefully curated examples.",
        core=[r"carefully curated", r"quality over quantity"]),
    "preference_alignment": dict(
        label="Learning from human feedback", family="SUPERVISION",
        desc="Align outputs to human preferences using human feedback / RL.",
        core=[r"human feedback", r"\brlhf\b"]),
    "weight_regularization": dict(
        label="Importance-weighted regularisation", family="SUPERVISION",
        desc="Penalise changes to weights that mattered for earlier tasks (EWC).",
        core=[r"elastic weight consolidation", r"quadratic penalty"]),
    # ---- HYBRID ---------------------------------------------------------------------
    "unified_framework": dict(
        label="Unified framework / hybrid", family="HYBRID",
        desc="Casts several PEFT methods in one framework or gates between them.",
        core=[r"unified (view|framework|formulation|perspective)\b", r"design (space|dimensions?)", r"gating mechanism", r"\bhybrid\b"]),
}

# Mechanisms that *operate over other mechanisms* (compose, route, unify, serve them). A paper that
# proposes one of these does not own the constituents it composes: AdapterFusion does not propose
# adapters, UniPELT does not propose LoRA, S-LoRA does not propose LoRA. See extract.py.
OPERATES_OVER = {"module_composition", "expert_routing", "prompt_transfer", "unified_framework", "serving_system"}

# --------------------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------------------
PROBLEMS = {
    "full_finetune_cost": dict(label="Cost of full fine-tuning", patterns=[
        r"full[- ]?(model )?(fine-?tun\w+|finetun\w+)",
        r"(updat\w+|fine-?tun\w+|train\w*) (all|every|the (entire|whole|full)) (the )?(model )?(parameters|weights|model)"]),
    "training_memory": dict(label="Training memory footprint", patterns=[
        r"\bmemory\b", r"\d+ ?gb\b", r"single (\d+ ?gb )?gpu"]),
    "trainable_param_count": dict(label="Per-task parameters / storage", patterns=[
        r"parameter[- ]efficien\w+", r"(few|fewer|small number of|tiny (fraction|number) of|fraction of the) (new |trainable |task-specific )*parameters",
        r"trainable parameters", r"per[- ]task (copy|copies|checkpoint|storage)", r"\bstorage\b",
        r"\d+(\.\d+)?%? of (the )?(model )?(trainable )?parameters"]),
    "inference_latency": dict(label="Inference latency / cost", patterns=[
        r"inference (latency|overhead|cost|time|speed)", r"(additional|extra) (inference )?(latency|cost|overhead)", r"\blatency\b"]),
    "catastrophic_forgetting": dict(label="Catastrophic forgetting", patterns=[
        r"forget\w*", r"catastrophic", r"(without|avoid\w*) revisiting", r"(retain|preserv)\w* .{0,30}(knowledge|capabilit)"]),
    "scale_gap": dict(label="Gap between small and large models", patterns=[
        r"closes? the gap", r"(smaller|small) models?", r"(as|when) (the )?(model )?(size|scale)s? (grow|increase)s?",
        r"across (model )?(sizes|scales)", r"models grow"]),
    "multitask_transfer": dict(label="Multi-task / cross-task transfer", patterns=[
        r"multi-?task", r"cross-?(task|lingual)", r"(across|between|from) (many |multiple |several |different |other |related |source )?tasks?",
        r"sequential(ly)?\b", r"learning several tasks"]),
    "low_resource": dict(label="Low-resource / few-shot adaptation", patterns=[
        r"few-?shot", r"low-?resource", r"(less|fewer|little|limited|scarce) (labeled |training |annotated )?(data|examples)",
        r"low-?data", r"small (training )?datasets?"]),
    "design_understanding_gap": dict(label="Poor understanding of design choices", patterns=[
        r"\bunclear\b", r"poorly understood", r"design (space|choices|dimensions?)", r"\bexplain\w*", r"\bexplanation\b",
        r"relationship between"]),
    "budget_allocation": dict(label="Where to spend the parameter budget (rank selection)", patterns=[
        r"(parameter|rank|trainable|adapter) budget", r"budget allocation", r"fixed rank", r"(uniform|same) rank",
        r"rank (selection|search|tuning)", r"search\w* for (the )?(best|optimal) rank"]),
    "quantization_error": dict(label="Quantization error / outliers", patterns=[
        r"quantization (error|loss|noise)", r"outlier\w*", r"(accuracy|performance) (loss|drop|degradation)", r"information loss"]),
    "long_context": dict(label="Long-context adaptation", patterns=[
        r"long[- ]context", r"context (length|window|extension)", r"(longer|long) (sequence|input|text)s?", r"sequence length"]),
    "adapter_serving_scale": dict(label="Serving many adapters", patterns=[
        r"\bserving\b", r"(thousands|many) of .{0,20}(adapters?|lora|models)", r"concurrent\w*", r"\bbatching\b", r"multi-?tenant"]),
    "pretraining_cost": dict(label="Cost of pre-training from scratch", patterns=[
        r"from scratch", r"(expensive|costly|large[- ]scale) pre-?training"]),
    "instruction_data_cost": dict(label="Cost / quality of instruction data", patterns=[
        r"annotation (cost|effort)", r"human[- ](written|annotat\w+|labeled)", r"(expensive|costly|time-consuming) .{0,20}(collect|annotat|data)\w*",
        r"limited (diversity|coverage)", r"data (quality|efficien\w+)", r"only limited instruction data"]),
    "training_instability": dict(label="Unstable or slow training / convergence", patterns=[
        r"instabilit\w+", r"\bunstable\b", r"converg\w+", r"sensitiv\w+ to (hyper-?parameter|initiali[sz]ation|learning rate)",
        r"(hard|difficult) to (train|optimi[sz]e|tune)", r"brittle"]),
    "peft_quality_gap": dict(label="Quality gap vs full fine-tuning", patterns=[
        r"(gap|difference) (to|with|between|from) (full )?(fine-?tun\w+|lora)", r"(underperform|lag|fall)\w* (behind )?(full )?fine-?tun\w+",
        r"narrow\w* the gap", r"suboptimal\w*"]),
    "unseen_task_generalization": dict(label="Generalising to unseen tasks", patterns=[
        r"zero-?shot", r"unseen (tasks?|topics?|task types?)", r"generali[sz]\w+ to (new|unseen)"]),
    "alignment_with_intent": dict(label="Models not following user intent", patterns=[
        r"follow\w* (user |human )?(intent|instructions?)", r"align\w+ (with|to) (user|human)\w*", r"truthful\w*", r"toxic\w*"]),
}

# --------------------------------------------------------------------------------------
# Concepts. `generic` concepts are ubiquitous and are never used to infer MOTIVATED_BY.
# --------------------------------------------------------------------------------------
CONCEPTS = {
    "intrinsic_dimensionality": dict(label="Intrinsic dimensionality / low intrinsic rank", patterns=[
        r"intrinsic (dimension\w*|rank)", r"low-dimensional (subspace|manifold|reparameteri[sz]ation)", r"low intrinsic"]),
    "scaling_laws": dict(label="Scaling laws", patterns=[r"scaling laws?", r"power[- ]laws?", r"compute-optimal"]),
    "in_context_learning": dict(label="In-context / few-shot prompting", patterns=[
        r"in-context (learning|examples?)", r"few-shot (prompting|demonstrations)", r"demonstrations"]),
    "outlier_features": dict(label="Emergent outlier features", patterns=[
        r"outlier (features?|dimensions?)", r"emergent (outlier|features)", r"systematic outliers?"]),
    "low_rank_gradients": dict(label="Gradients become low-rank during training", patterns=[
        r"gradients? (becomes?|is|are)? ?low[- ]rank", r"low-rank structure of (the )?gradients?"]),
    "superficial_alignment": dict(label="Knowledge is acquired in pre-training; alignment is superficial", patterns=[
        r"superficial alignment", r"almost all knowledge[^.]{0,40}pre-?training"]),
    "transfer_learning_paradigm": dict(label="Pre-train then fine-tune", generic=True, patterns=[
        r"transfer learning", r"pre-?train(ed|ing)? (then|and) fine-?tun\w+", r"pre-?trained (language )?models?"]),
    "transformer_architecture": dict(label="Transformer / self-attention", generic=True, patterns=[
        r"\btransformers?\b", r"self-?attention"]),
}

# Hand-authored background knowledge: which concept makes which mechanism plausible.
CONCEPT_UNDERPINS = {
    "intrinsic_dimensionality": ["low_rank_update", "adaptive_rank", "parameter_sharing", "soft_prompt", "sparse_diff"],
    "low_rank_gradients": ["low_rank_gradient"],
    "outlier_features": ["quantization"],
    "scaling_laws": ["quantization", "soft_prompt"],
    "superficial_alignment": ["curated_data", "instruction_tuning"],
    "in_context_learning": ["soft_prompt", "activation_rescaling"],
}

# --------------------------------------------------------------------------------------
# Models and benchmarks
# --------------------------------------------------------------------------------------
MODELS = {
    "bert_family": dict(label="BERT / RoBERTa / DeBERTa", patterns=[r"\bbert\b", r"roberta", r"deberta"]),
    "gpt2": dict(label="GPT-2", patterns=[r"gpt-?2\b"]),
    "gpt3": dict(label="GPT-3", patterns=[r"gpt-?3\b", r"\b175b?\b"]),
    "t5_family": dict(label="T5 / BART", patterns=[r"\bt5\b", r"\bbart\b"]),
    "llama_family": dict(label="LLaMA / Llama 2", patterns=[r"\bllama\b", r"llama-?2", r"llama\b"]),
    "opt_bloom": dict(label="OPT / BLOOM", patterns=[r"\bopt\b", r"\bbloom\b"]),
}

BENCHMARKS = {
    "glue": dict(label="GLUE", patterns=[r"\bglue\b"]),
    "superglue": dict(label="SuperGLUE", patterns=[r"\bsuperglue\b"]),
    "table_to_text": dict(label="E2E / WebNLG / DART", patterns=[r"\be2e\b", r"webnlg", r"\bdart\b"]),
    "mmlu": dict(label="MMLU", patterns=[r"\bmmlu\b", r"massive multitask language understanding"]),
    "big_bench": dict(label="BIG-Bench / BBH", patterns=[r"big-?bench", r"\bbbh\b"]),
    "vicuna_bench": dict(label="Vicuna benchmark", patterns=[r"vicuna (benchmark|eval\w*)"]),
}

# --------------------------------------------------------------------------------------
# Linguistic cues used for role assignment (see extract.py)
# --------------------------------------------------------------------------------------
PROPOSE_VERBS = r"(propos|introduc|present|develop|devis|describ|releas)\w*"
PROPOSE_VERBS_START = r"^(trains?|builds?|constructs?)\b"
OBSERVE_VERBS = r"(show|find|found|analy[sz]|examin|stud(y|ies)|measur|establish|demonstrat|investigat|report|observ|explor|survey|review|argue|suggest)\w*"
CONTRAST_CUES = (r"\b(unlike|compared (to|with)|versus|vs\.?|outperform\w*|instead of|rather than|than|without|"
                 r"beats?|improv\w+ over|over (standard|naive|regular)|gains over|surpass\w*)\b")
SURVEY_CUES = r"\b(surveys?|comprehensive (survey|review|overview|study)|systematic (review|overview)|guide to)\b"
