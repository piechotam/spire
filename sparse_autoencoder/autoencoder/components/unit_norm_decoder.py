"""
Linear layer with unit norm weights.
Code from https://github.com/neuroexplicit-saar/Discover-then-Name/blob/main/sparse_autoencoder/sparse_autoencoder/autoencoder/components/unit_norm_decoder.py
"""

from typing import final

import einops
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


@final
class UnitNormDecoder(Module):
    r"""Constrained unit norm linear decoder layer.

    Linear layer decoder, where the dictionary vectors (columns of the weight matrix) are
    constrained to have unit norm. This is done by removing the gradient information parallel to the
    dictionary vectors before applying the gradient step, using a backward hook. It also requires
    `constrain_weights_unit_norm` to be called after each gradient step, to prevent drift of the
    dictionary vectors away from unit norm (as optimisers such as Adam don't strictly follow the
    gradient, but instead follow a modified gradient that includes momentum).

    $$ \begin{align*}
        m &= \text{learned features dimension} \\
        n &= \text{input and output dimension} \\
        b &= \text{batch items dimension} \\
        f \in \mathbb{R}^{b \times m} &= \text{encoder output} \\
        W_d \in \mathbb{R}^{n \times m} &= \text{weight matrix} \\
        z \in \mathbb{R}^{b \times m} &= f W_d^T = \text{UnitNormDecoder output (pre-tied bias)}
    \end{align*} $$

    Motivation:
        Normalisation of the columns (dictionary features) prevents the model from reducing the
        sparsity loss term by increasing the size of the feature vectors in $W_d$.

        Note that the *Towards Monosemanticity: Decomposing Language Models With Dictionary
        Learning* paper found that removing the gradient information parallel to the dictionary
        vectors before applying the gradient step, rather than resetting the dictionary vectors to
        unit norm after each gradient step, results in a small but real reduction in total
        loss](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization).
    """

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _decoded_features: int
    """Number of decoded features (outputs from this layer)."""

    _n_components: int | None

    _weight: Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE),
    ]
    """Weight parameter internal state."""

    @property
    def weight(
        self,
    ) -> Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE),
    ]:
        """Weight parameter.

        Each column in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """
        return self._weight

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """
        return [ResetOptimizerParameterDetails(parameter=self.weight, axis=-1)]

    @validate_call
    def __init__(
        self,
        learnt_features: PositiveInt,
        decoded_features: PositiveInt,
        n_components: PositiveInt | None,
        *,
        enable_gradient_hook: bool = True,
    ) -> None:
        """Initialize the constrained unit norm linear layer.

        Args:
            learnt_features: Number of learnt features in the autoencoder.
            decoded_features: Number of decoded (output) features in the autoencoder.
            n_components: Number of source model components the SAE is trained on.
            enable_gradient_hook: Enable the gradient backwards hook (modify the gradient before
                applying the gradient step, to maintain unit norm of the dictionary vectors).
        """
        super().__init__()

        self._learnt_features = learnt_features
        self._decoded_features = decoded_features
        self._n_components = n_components

        # Create the linear layer as per the standard PyTorch linear layer
        self._weight = Parameter(
            torch.empty(
                shape_with_optional_dimensions(n_components, decoded_features, learnt_features),
            )
        )
        self.reset_parameters()

        # Register backward hook to remove any gradient information parallel to the dictionary
        # vectors (columns of the weight matrix) before applying the gradient step.
        if enable_gradient_hook:
            self._weight.register_hook(self._weight_backward_hook)

    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        updated_weights: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE_IDX),
        ],
        component_idx: int | None = None,
    ) -> None:
        """Update decoder dictionary vectors.

        Updates the dictionary vectors (rows in the weight matrix) with the given values. Typically
        this is used when resampling neurons (dictionary vectors) that have died.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_weights: Updated weights for just these dictionary vectors.
            component_idx: Component index to update.

        Raises:
            ValueError: If `component_idx` is not specified when `n_components` is not None.
        """
        if dictionary_vector_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.weight[:, dictionary_vector_indices] = updated_weights
            else:
                self.weight[component_idx, :, dictionary_vector_indices] = updated_weights

    def constrain_weights_unit_norm(self) -> None:
        """Constrain the weights to have unit norm.

        Warning:
            Note this must be called after each gradient step. This is because optimisers such as
            Adam don't strictly follow the gradient, but instead follow a modified gradient that
            includes momentum. This means that the gradient step can change the norm of the
            dictionary vectors, even when the hook `_weight_backward_hook` is applied.

            Note this can't be applied directly in the backward hook, as it would interfere with a
            variety of use cases (e.g. gradient accumulation across mini-batches, concurrency issues
            with asynchronous operations, etc).

        Example:
            >>> import torch
            >>> layer = UnitNormDecoder(3, 3, None)
            >>> layer.weight.data = torch.ones((3, 3)) * 10
            >>> layer.constrain_weights_unit_norm()
            >>> column_norms = torch.sqrt(torch.sum(layer.weight ** 2, dim=0))
            >>> column_norms.round(decimals=3).tolist()
            [1.0, 1.0, 1.0]

        """
        with torch.no_grad():
            torch.nn.functional.normalize(self._weight, dim=-2, out=self._weight)

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters.

        Example:
            >>> import torch
            >>> # Create a layer with 4 columns (learnt features) and 3 rows (decoded features)
            >>> layer = UnitNormDecoder(learnt_features=4, decoded_features=3, n_components=None)
            >>> layer.reset_parameters()
            >>> # Get the norm across the rows (by summing across the columns)
            >>> column_norms = torch.sum(layer.weight ** 2, dim=0)
            >>> column_norms.round(decimals=3).tolist()
            [1.0, 1.0, 1.0, 1.0]

        """
        # Initialize the weights with a normal distribution. Note we don't use e.g. kaiming
        # normalisation here, since we immediately scale the weights to have unit norm (so the
        # initial standard deviation doesn't matter). Note also that `init.normal_` is in place.
        self._weight: Float[
            Parameter,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
        ] = init.normal_(self._weight, mean=0, std=1)  # type: ignore

        # Scale so that each row has unit norm
        self.constrain_weights_unit_norm()

    def _weight_backward_hook(
        self,
        grad: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
        ],
    ) -> Float[
        Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
    ]:
        r"""Unit norm backward hook.

        By subtracting the projection of the gradient onto the dictionary vectors, we remove the
        component of the gradient that is parallel to the dictionary vectors and just keep the
        component that is orthogonal to the dictionary vectors (i.e. moving around the hypersphere).
        The result is that the backward pass does not change the norm of the dictionary vectors.

        $$
        \begin{align*}
            W_d &\in \mathbb{R}^{n \times m} = \text{Decoder weight matrix} \\
            g &\in \mathbb{R}^{n \times m} = \text{Gradient w.r.t. } W_d
                \text{ from the backward pass} \\
            W_{d, \text{norm}} &= \frac{W_d}{\|W_d\|} = \text{Normalized decoder weight matrix
                (over columns)} \\
            g_{\parallel} &\in \mathbb{R}^{n \times m} = \text{Component of } g
                \text{ parallel to } W_{d, \text{norm}} \\
            g_{\perp} &\in \mathbb{R}^{n \times m} = \text{Component of } g \text{ orthogonal to }
                W_{d, \text{norm}} \\
            g_{\parallel} &= W_{d, \text{norm}} \cdot (W_{d, \text{norm}}^\top \cdot g) \\
            g_{\perp} &= g - g_{\parallel} =
                \text{Adjusted gradient with parallel component removed} \\
        \end{align*}
        $$

        Args:
            grad: Gradient with respect to the weights.

        Returns:
            Gradient with respect to the weights, with the component parallel to the dictionary
            vectors removed.
        """
        # Project the gradients onto the dictionary vectors. Intuitively the dictionary vectors can
        # be thought of as vectors that end on the circumference of a hypersphere. The projection of
        # the gradient onto the dictionary vectors is the component of the gradient that is parallel
        # to the dictionary vectors, i.e. the component that moves to or from the center of the
        # hypersphere.
        normalized_weight: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
        ] = self._weight / torch.norm(self._weight, dim=-2, keepdim=True)

        scalar_projections = einops.einsum(
            grad,
            normalized_weight,
            f"... {Axis.LEARNT_FEATURE} {Axis.INPUT_OUTPUT_FEATURE}, \
                ... {Axis.LEARNT_FEATURE} {Axis.INPUT_OUTPUT_FEATURE} \
                -> ... {Axis.INPUT_OUTPUT_FEATURE}",
        )

        projection = einops.einsum(
            scalar_projections,
            normalized_weight,
            f"... {Axis.INPUT_OUTPUT_FEATURE}, \
                ... {Axis.LEARNT_FEATURE} {Axis.INPUT_OUTPUT_FEATURE} \
                -> ... {Axis.LEARNT_FEATURE} {Axis.INPUT_OUTPUT_FEATURE}",
        )

        # Subtracting the parallel component from the gradient leaves only the component that is
        # orthogonal to the dictionary vectors, i.e. the component that moves around the surface of
        # the hypersphere.
        return grad - projection

    def forward(
        self, x: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        return einops.einsum(
            x,
            self.weight,
            f"{Axis.BATCH} ... {Axis.LEARNT_FEATURE}, \
            ... {Axis.INPUT_OUTPUT_FEATURE} {Axis.LEARNT_FEATURE} \
                -> {Axis.BATCH} ... {Axis.INPUT_OUTPUT_FEATURE}",
        )

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return (
            f"learnt_features={self._learnt_features}, "
            f"decoded_features={self._decoded_features}, "
            f"n_components={self._n_components}"
        )