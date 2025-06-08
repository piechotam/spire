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

## How similarity was measured?
- CLIP $\rightarrow$ cosine similarity
- SAE $\rightarrow$ Manhattan distance

## Defined metrics

### Precision@k
- **Description**: Precision for the top-k retrieved item.
- **Calculation**: Measures whether first k results are relevant (i.e., belongs to the set of relevant items for the query).

### Recall@k
- **Description**: Recall for the top-k retrieved item.
- **Calculation**: Measures whether first k results are relevant in the context of all relevant items for the query.

### mAP (Mean Average Precision)
- **Description**: The mean of the Average Precision (AP) across all queries.
- **Calculation**: Average Precision for a query is the average of the precision calculated for each relevant result in the ranking. mAP is the mean of these Average Precisions across all queries.

### MicroAP (Micro-Average Precision)
- **Description**: A metric that evaluates all query-reference pairs jointly.
- **Calculation**: Sorts all pairs by confidence (similarity score) and computes the area under the precision-recall curve for the entire set of pairs.

**Note:** `microAP` is better when evaluating global ranking performance, while `mAP` treats each query independently.

### Running Streamlit app
```streamlit run path_to_app```

#### References
[1] Douze, M., Tolias, G., Pizzi, E., Papakipos, Z., Chanussot, L., Radenovic, F., Jenicek, T., Maximov, M., Leal-Taixé, L., Elezi, I., Chum, O., & Canton Ferrer, C. "The 2021 Image Similarity Dataset and Challenge." arXiv preprint arXiv:2106.09672 (2021). https://arxiv.org/pdf/2106.09672
