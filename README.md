# Comparing Latent Concept Formation in State Space Models and Transformers via Sparse Autoencoders

This repository contains the code and artifacts for the paper:

> Rithin Nagaraj, Rupa Laalasa Oruganti, Prerna Subhashchandra Kunder, Ashwini M Joshi.
> **"Comparing Latent Concept Formation in State Space Models and Transformers via Sparse Autoencoders."**
> PES University, Bangalore, India. [arXiv:2609.24440](https://arxiv.org/abs/2609.24440)

## Overview

Transformers scale quadratically with sequence length due to self-attention, which has motivated interest in sub-quadratic Selective State Space Models (SSMs) like **Mamba**, which compress all past context into a fixed-size recurrent hidden state. This raises a natural question for mechanistic interpretability: does this recurrent bottleneck force SSMs to learn representations that are fundamentally different from those learned by Transformers?

We investigate this using **Sparse Autoencoders (SAEs)** to perform a large-scale, feature-level correspondence analysis between **Mamba-130m** and **Pythia-70m-deduped** over a shared 10-million-token corpus of English Wikipedia. Concretely, we:

1. Extract mid-layer residual stream activations from both models on identical input sequences.
2. Decompose those activations into sparse, higher-dimensional feature dictionaries using SAEs (a pre-trained SAE for Pythia, a custom-trained SAE for Mamba).
3. Binarize feature activations and compute pairwise **Jaccard similarity** over co-activation patterns to find each Mamba feature's best-matching ("twin") Pythia feature.
4. Analyze the resulting similarity distribution to test the **Universality Hypothesis** — that models trained on similar data converge on similar latent concepts, regardless of architecture.

### Key findings

- **No evidence of architectural "Dark Matter."** 99.98% (12,285 / 12,288) of Mamba's latent features are strongly aligned (Jaccard > 0.35) with a Pythia counterpart, providing feature-level support for the Universality Hypothesis.
- **Sequential landmark features.** Within the tiny diverging fraction, Mamba develops monosemantic "state reset" neurons dedicated to structural boundaries (e.g., document/section footers), compensating for the lack of global positional attention.
- **Architecturally-driven polysemanticity at the margins.** A qualitative "Twin Test" case study shows Mamba superimposing unrelated syntactic anomalies (URL slashes, grade-level hyphens, foreign-language capitals) into a single polysemantic "junk drawer" feature, while Pythia's unconstrained attention lets it keep a cleanly monosemantic equivalent. Representational divergence appears confined to rigid syntactic/structural edge cases rather than broad semantics.

## Repository contents

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Streams and tokenizes English Wikipedia (GPT-NeoX tokenizer) into fixed-length, non-overlapping token chunks for the shared evaluation corpus. |
| `harvest_activations.py` | Runs Mamba-130m and Pythia-70m-deduped over the tokenized corpus, extracts mid-layer residual stream activations (`blocks.12.hook_resid_post` for Mamba, `blocks.3.hook_resid_post` for Pythia), passes them through each model's SAE, and stores binarized activation masks to `activations.h5`. |
| `find_dark_matter.py` | Computes the pairwise Jaccard similarity intersection matrix between Mamba and Pythia binary activation masks (disk-streamed, tensor-chunked for VRAM efficiency) and identifies each Mamba feature's maximum-overlap "twin" Pythia feature. Outputs `all_mapped_features.json`. |
| `get_distribution.py` | Buckets the mapped features by Jaccard score into Aligned / Grey Matter / Buffer / Alien categories and prints the global alignment distribution summary. |
| `explore_semantic_features.py` | Samples features from the Grey Matter zone (J ∈ [0.28, 0.35]) and extracts top activating text contexts to qualitatively characterize borderline-diverging concepts. |
| `read_alien_feataures.py` | Locates the most "alien" (lowest-Jaccard) Mamba features and extracts the exact Wikipedia text snippets that trigger them. |
| `compare_twins.py` | Runs the qualitative "Twin Test": for a target Mamba feature, retrieves its Pythia twin and compares their top activating contexts side by side (used for the Feature 4454 / 4309 case study in the paper). |
| `cfg.json` | SAE architecture/config used to load the trained Mamba SAE (`d_sae=12288`, `d_in=768`, hook `blocks.12.hook_resid_post`). |
| `runner_cfg.json` | Full SAELens training run configuration used to train the custom Mamba-130m SAE (learning rate, batch size, L1 coefficient, dataset, etc.). |
| `pretrained_saes.yaml` | SAELens registry of pre-trained SAEs, including the `pythia-70m-deduped-res-sm` SAE used for Pythia. |
| `sae_weights.safetensors` | Trained weights for the custom Mamba-130m SAE. |
| `sparsity.safetensors` | Per-feature sparsity statistics from SAE training. |
| `activations.h5` | HDF5 store of binarized co-activation masks for both models (generated by `harvest_activations.py`; not populated in this snapshot). |
| `evals.txt` | Console output log from a full pipeline run (feature mapping stats, alien feature search, etc.). |

## Pipeline / reproduction order

1. `prepare_dataset.py` — build the shared 10M-token Wikipedia corpus.
2. `harvest_activations.py` — extract and binarize SAE features from both models over the corpus.
3. `find_dark_matter.py` — compute the Jaccard intersection matrix and twin mapping (`all_mapped_features.json`).
4. `get_distribution.py` — summarize the Aligned / Grey Matter / Alien distribution.
5. `read_alien_feataures.py` / `explore_semantic_features.py` / `compare_twins.py` — qualitative deep-dives into specific diverging features.

## Models and dependencies

- **Mamba-130m** ([`state-spaces/mamba-130m`](https://huggingface.co/state-spaces/mamba-130m)), interpreted via [MambaLens](https://github.com/jenhsisheu/MambaLens).
- **Pythia-70m-deduped** ([`EleutherAI/pythia-70m-deduped`](https://huggingface.co/EleutherAI/pythia-70m-deduped)), interpreted via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
- SAE training/loading via [SAELens](https://github.com/jbloomAus/SAELens).
- Tokenization via the GPT-NeoX tokenizer (`EleutherAI/gpt-neox-20b`).
- Corpus: `wikimedia/wikipedia` (`20231101.en`) via Hugging Face `datasets`.

Core Python dependencies (not pinned here): `torch`, `transformers`, `datasets`, `transformer_lens`, `mamba_lens`, `sae_lens`, `h5py`, `numpy`, `tqdm`.

## Citation

If you use this code or refer to this work, please cite:

```bibtex
@article{nagaraj2026comparing,
  title   = {Comparing Latent Concept Formation in State Space Models and Transformers via Sparse Autoencoders},
  author  = {Nagaraj, Rithin and Oruganti, Rupa Laalasa and Kunder, Prerna Subhashchandra and Joshi, Ashwini M},
  journal = {arXiv preprint arXiv:2609.24440},
  year    = {2026}
}
```
