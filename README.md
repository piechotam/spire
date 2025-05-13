# Sparse Representations for Information Retrieval Enhancement (SpIRE)
## What is SAE?

**SAEs (Sparse Autoencoders)** disentangle learned representations into a sparse set of human-understandable concepts. This is achieved by decomposing the representations into a sparse linear combination of a set of learned dictionary vectors. They consist of a linear encoder $f(\cdot)$ with weights $W_E \in \mathbb{R}^{d \times h}$, a ReLU non-linearity $\phi$, and a linear decoder $g(\cdot)$ with weights $W_D \in \mathbb{R}^{h \times d}$.

The formula for an SAE is given by:

$$
SAE(a) = (g \circ \phi \circ f)(a) = W_D^{\top}  \phi(W_E^{\top} a)
$$

The decoder matrix $W_D$ acts as a dictionary of interpretable concept vectors.

To assign human-readable names to these concepts, each vector from $W_D$ can be compared (e.g. using cosine similarity) to word embeddings from a language model like CLIP’s text encoder, and matched to the closest word.

## Using CLIP to IR 
When using **CLIP (Contrastive Language–Image Pretraining)**  feature extractors, the dictionary vectors from the SAE decoder can be directly mapped to text by finding the most similar text embeddings in CLIP space—without the need for large language models—resulting in semantically meaningful and human-interpretable concepts.
#
*Description based on the paper "Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery".*

## How similiarity was measured?
- CLIP $\rightarrow$ cosine similarity
- SAE $\rightarrow$ Manhattan distance
