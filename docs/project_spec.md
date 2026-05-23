# Project Specification

## Project Title

Retrieval-Augmented Active Learning for Sparse Protein Sequence Optimization

## Purpose

The purpose of this project is to test whether protein foundation-model representations and retrieval-augmented acquisition strategies improve active learning performance in sparse protein fitness landscapes compared with standard sequence descriptors and standard acquisition functions.

The broader motivation is that biological design often occurs in low-data regimes where experimental measurements are expensive, labels are sparse, and candidate spaces are large. This project uses retrospective deep mutational scanning datasets as a controlled benchmark for studying these decision-making strategies.

## Scientific Question

In sparse protein fitness landscapes, do foundation-model embeddings and retrieval-augmented acquisition improve active learning compared with standard sequence descriptors and standard Bayesian optimization / active learning strategies?

## Task Definition

Given a deep mutational scanning dataset

\[
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
\]

where \(x_i\) is a protein variant sequence and \(y_i = f(x_i)\) is its measured fitness or variant-effect score, we simulate an active learning process.

At the beginning of each run, only a small subset of labels is revealed. The remaining labels are hidden and treated as an unlabeled candidate pool. At each active learning round, the algorithm selects a batch of variants to acquire. The hidden labels for those variants are then revealed, the surrogate model is updated, and the process repeats.

The goal is to discover high-fitness variants using as few label acquisitions as possible.

## Objective

For the initial single-objective setting, the biological objective is

\[
\max_x f(x)
\]

where \(f(x)\) is the experimentally measured DMS fitness score.

In the retrospective benchmark, the primary performance metric is the best observed score after a fixed acquisition budget:

\[
\max_{x \in L_T} f(x)
\]

where \(L_T\) is the labeled/acquired set after \(T\) active learning rounds.

## Surrogate Model

At each round \(t\), a surrogate model is trained on the currently labeled set \(L_t\). The surrogate predicts

\[
\mu_t(x), \sigma_t(x)
\]

for each candidate \(x\) in the unlabeled pool \(U_t\), where \(\mu_t(x)\) is the predicted fitness and \(\sigma_t(x)\) represents model uncertainty.

Possible surrogate models include:

- random forest ensembles
- neural network ensembles
- Gaussian process regression
- other uncertainty-aware predictors

## Acquisition Functions

The acquisition function determines which unlabeled candidates are selected next. Candidate acquisition functions include:

### Greedy exploitation

\[
a_t(x) = \mu_t(x)
\]

### Uncertainty sampling

\[
a_t(x) = \sigma_t(x)
\]

### Upper confidence bound

\[
a_t(x) = \mu_t(x) + \beta \sigma_t(x)
\]

### Expected improvement

\[
a_t(x) = \mathbb{E}[\max(f(x) - f_{\mathrm{best}}, 0)]
\]

### Retrieval-augmented acquisition

Retrieval-augmented acquisition incorporates information from nearby currently labeled variants:

\[
a_t(x) = \mu_t(x) + \beta \sigma_t(x) + \lambda R_t(x)
\]

where \(R_t(x)\) is a retrieval-derived score computed only from the currently labeled set.

Possible retrieval features include:

\[
R_t(x) =
\left[
\overline{y}_{kNN},
\mathrm{std}(y_{kNN}),
d_{\min},
d_{\mathrm{mean}},
\max(y_{kNN})
\right]
\]

where the nearest neighbors are retrieved from the currently labeled set only.

## Representations to Compare

The project will compare several sequence representations:

1. Mutation descriptors  
   - mutation count
   - mutation position
   - wild-type residue
   - mutant residue
   - physicochemical change upon mutation

2. Physicochemical sequence descriptors  
   - amino acid composition
   - net charge
   - hydropathy
   - aromatic fraction
   - polar/charged residue fractions
   - sequence length

3. Protein language model embeddings  
   - mean pooled embeddings
   - mutation-site embeddings
   - mutant-minus-wild-type delta embeddings

4. Retrieval-augmented representations  
   - protein language model embeddings plus nearest-neighbor label/context features

## Initial Benchmark Setting

The initial benchmark will use ProteinGym deep mutational scanning substitution datasets.

The first version will use pool-based retrospective active learning:

1. Load one DMS landscape.
2. Hide most labels.
3. Initialize with a small labeled subset.
4. Train a surrogate model.
5. Score the unlabeled pool with an acquisition function.
6. Select a batch of variants.
7. Reveal their labels.
8. Update the labeled set.
9. Repeat for a fixed number of rounds.

No de novo sequence generation is required in the initial benchmark. Candidate selection occurs from a fixed pool of experimentally measured variants.

## Primary Metrics

The primary metrics are:

1. Best observed fitness versus acquisition round

\[
\max_{x \in L_t} f(x)
\]

2. Simple regret

\[
f(x^*) - \max_{x \in L_t} f(x)
\]

where \(x^*\) is the best variant in the full dataset.

3. Top-k recall

\[
\frac{|L_t \cap \mathrm{TopK}(\mathcal{D})|}{K}
\]

4. Mean fitness of acquired batch

5. Diversity among selected variants

6. Distance from wild-type sequence

## Leakage Rules

The active learning loop must obey strict leakage rules.

1. The surrogate model can only train on labels from the currently labeled set.
2. Acquisition functions cannot access hidden labels in the unlabeled pool.
3. Retrieval features that use labels can only retrieve from the currently labeled set.
4. Embedding computation must not use fitness labels.
5. The full label array can only be accessed by the reveal function and evaluation metrics.
6. Any feature normalization that depends on labels must be fit only on the labeled set.

## Initial Methods to Compare

The first benchmark should compare:

1. Random acquisition
2. Greedy acquisition using predicted mean
3. Uncertainty sampling
4. UCB acquisition
5. Diversity-penalized UCB
6. Retrieval-augmented UCB

Across representations:

1. Mutation descriptors
2. Physicochemical descriptors
3. Mean pooled protein language model embeddings
4. Delta protein language model embeddings
5. Protein language model embeddings plus retrieval features

## Near-Term Milestone

The first milestone is a working retrospective active learning benchmark on one ProteinGym DMS landscape comparing:

- mutation descriptors
- physicochemical descriptors
- protein language model embeddings
- retrieval-augmented protein language model embeddings

under:

- random acquisition
- greedy acquisition
- UCB acquisition
- retrieval-augmented UCB

The first result should be a learning-curve plot showing best observed fitness versus active learning round.
