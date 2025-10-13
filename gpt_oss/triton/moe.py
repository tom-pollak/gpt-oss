import torch
from torch.profiler import record_function

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs import matmul_ogs, RoutingData, GatherIndx, ScatterIndx
from triton_kernels.numerics import InFlexData
from triton_kernels.topk import topk
from triton_kernels.tensor import convert_layout, SparseMatrix, make_ragged_tensor_metadata
from triton_kernels.tensor_details.layout import StridedLayout, HopperMXScaleLayout, HopperMXValueLayout
from triton_kernels.tensor import wrap_torch_tensor, FP4


def quantize_mx4(w):
    w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, interleaved: bool = True):
    if interleaved:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def legacy_routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    """
    Legacy routing helper that constructs routing data from a bitmatrix.

    This function provides compatibility with the deprecated triton_kernels.routing API.
    It's a temporary shim while migrating to the new composable primitives.

    Migration path: This function can be eliminated by inlining its logic directly
    into the caller and using the SparseMatrix from topk() directly, avoiding the
    redundant SparseMatrix construction here.

    Args:
        bitmatrix: Bitmatrix mask from topk
        expt_scal: Expert scale values (gate weights)
        expt_indx: Expert indices
        n_expts_tot: Total number of experts
        n_expts_act: Active experts per token

    Returns:
        Tuple of (RoutingData, GatherIndx, ScatterIndx)
    """
    sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
    combine_indx = sparse_logits.mask_metadata.row_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(gate_scal, ragged_batch_metadata.batch_sizes, n_expts_tot, n_expts_act,
                               ragged_batch_metadata)
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx


def legacy_routing(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
    """
    Legacy routing function that provides compatibility with the deprecated triton_kernels.routing API.

    This is a temporary compatibility shim. The modern approach uses the new composable
    primitives directly for better control and potential optimization.

    Args:
        logits: Router logits of shape (n_tokens, n_experts)
        n_expts_act: Number of experts to route each token to
        sm_first: If True, apply softmax before topk
        expt_indx: Optional pre-computed expert indices
        n_rows: Optional number of rows to process

    Returns:
        routing_data: RoutingData containing gate scales, histograms, and metadata
        gather_idx: Indices for gathering tokens by expert
        scatter_idx: Indices for scattering tokens back to original positions

    Modern equivalent:
        sparse_logits = topk(logits, n_expts_act, apply_softmax=True)
        dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
        combine_indx = sparse_logits.mask_metadata.row_sorted_indx
        ragged_batch_metadata = make_ragged_tensor_metadata(
            sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0]
        )
        gate_scal = sparse_logits.vals.flatten()[combine_indx]
        routing_data = RoutingData(gate_scal, ragged_batch_metadata.batch_sizes,
                                   n_expts_tot, n_expts_act, ragged_batch_metadata)
        gather_idx = GatherIndx(combine_indx, dispatch_indx)
        scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    """
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
    return legacy_routing_from_bitmatrix(sparse_logits.mask, sparse_logits.vals, sparse_logits.indx, logits.shape[-1],
                                         n_expts_act)


def moe(x, wg, w1, w1_mx, w2, w2_mx, bg, b1, b2, experts_per_token=4, num_experts=128, swiglu_limit=7.0, fused_act=True, interleaved=True):
    """
    Mixture of Experts layer with MX4 quantization and fused SwiGLU activation.

    This implementation uses the legacy routing API for compatibility with older
    triton_kernels. For a more modern approach with better composability:
    - Use topk() directly to get sparse_logits with bitmatrix metadata
    - Access sparse_logits.mask_metadata for routing indices
    - Build RoutingData, GatherIndx, ScatterIndx explicitly
    See legacy_routing() docstring for the modern equivalent code.

    Args:
        x: Input tensor
        wg: Gate/router weights
        w1: First expert weights
        w1_mx: MX4 scales for w1
        w2: Second expert weights
        w2_mx: MX4 scales for w2
        bg: Gate bias
        b1: First expert bias
        b2: Second expert bias
        experts_per_token: Number of experts to route each token to (top-k)
        num_experts: Total number of experts
        swiglu_limit: Clipping limit for SwiGLU activation
        fused_act: Whether to use fused SwiGLU activation
        interleaved: Whether expert weights are interleaved

    Returns:
        Output tensor after MoE computation
    """
    if x.numel() == 0:
        return x

    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))

    with record_function("wg"):
        logits = matmul_ogs(x, wg, bg, precision_config=pcg)
    with record_function("routing"):
        # Using legacy routing API for now - see function docstring for modern approach
        rdata, gather_indx, scatter_indx = legacy_routing(logits, experts_per_token)

    if fused_act:
        assert interleaved, "Fused activation requires interleaved weights"
        with record_function("w1+swiglu"):
            act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.702, swiglu_limit), 2)
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
    else:
        with record_function("w1"):
            x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        with record_function("swiglu"):
            x = swiglu(x, limit=swiglu_limit, interleaved=interleaved)

    with record_function("w2"):
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, gammas=rdata.gate_scal)
    return x
