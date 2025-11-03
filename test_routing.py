"""Test to verify compute_routing correctness."""
import torch
import sys
sys.path.insert(0, '/lambda/nfs/nethome-us-east-1/tomp/gpt-oss')

from gpt_oss.triton.moe import compute_routing
from triton_kernels.topk import topk
from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata
from triton_kernels.matmul_ogs import RoutingData, GatherIndx, ScatterIndx


def legacy_routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    """Original legacy implementation from commit 67ef07a."""
    sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
    combine_indx = sparse_logits.mask_metadata.row_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
    gate_scal = sparse_logits.vals.flatten()[combine_indx]  # Uses combine_indx
    routing_data = RoutingData(gate_scal, ragged_batch_metadata.batch_sizes, n_expts_tot, n_expts_act,
                               ragged_batch_metadata)
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx


def legacy_routing(logits, n_expts_act):
    """Original legacy implementation."""
    sparse_logits = topk(logits, n_expts_act, apply_softmax=True)
    return legacy_routing_from_bitmatrix(sparse_logits.mask, sparse_logits.vals, sparse_logits.indx, logits.shape[-1],
                                         n_expts_act)


def test_routing_equivalence():
    """Test that compute_routing produces correct results."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test case: 8 tokens, 16 experts, top-4
    n_tokens = 8
    n_experts = 16
    top_k = 4

    logits = torch.randn(n_tokens, n_experts, device=device, dtype=torch.float32)

    # Get legacy routing
    legacy_rdata, legacy_gather, legacy_scatter = legacy_routing(logits, top_k)

    # Get new routing
    new_rdata, new_gather, new_scatter = compute_routing(logits, top_k, n_experts)

    print("Testing routing equivalence...")
    print(f"  n_tokens={n_tokens}, n_experts={n_experts}, top_k={top_k}")

    # Compare batch sizes (should be identical)
    assert torch.equal(legacy_rdata.expt_hist, new_rdata.expt_hist), \
        f"Histogram mismatch:\n  Legacy: {legacy_rdata.expt_hist}\n  New: {new_rdata.expt_hist}"
    print("✓ Histograms match")

    # Compare gather/scatter indices (should be identical)
    assert torch.equal(legacy_gather.src_indx, new_gather.src_indx), "Gather src_indx mismatch"
    assert torch.equal(legacy_gather.dst_indx, new_gather.dst_indx), "Gather dst_indx mismatch"
    assert torch.equal(legacy_scatter.src_indx, new_scatter.src_indx), "Scatter src_indx mismatch"
    assert torch.equal(legacy_scatter.dst_indx, new_scatter.dst_indx), "Scatter dst_indx mismatch"
    print("✓ Indices match")

    # Compare gate scales - THIS IS THE KEY TEST
    if not torch.allclose(legacy_rdata.gate_scal, new_rdata.gate_scal, rtol=1e-5, atol=1e-7):
        print("\n✗ Gate scales MISMATCH!")
        print(f"  Legacy gate_scal: {legacy_rdata.gate_scal[:20]}")
        print(f"  New gate_scal:    {new_rdata.gate_scal[:20]}")
        print(f"  Max diff: {(legacy_rdata.gate_scal - new_rdata.gate_scal).abs().max()}")

        # Show which expert's scales are wrong
        hist = legacy_rdata.expt_hist
        offset = 0
        for expert_id in range(n_experts):
            n_tokens_for_expert = hist[expert_id].item()
            if n_tokens_for_expert > 0:
                legacy_scales = legacy_rdata.gate_scal[offset:offset+n_tokens_for_expert]
                new_scales = new_rdata.gate_scal[offset:offset+n_tokens_for_expert]
                if not torch.allclose(legacy_scales, new_scales, rtol=1e-5):
                    print(f"  Expert {expert_id} (offset {offset}, {n_tokens_for_expert} tokens):")
                    print(f"    Legacy: {legacy_scales}")
                    print(f"    New:    {new_scales}")
                offset += n_tokens_for_expert
        return False
    else:
        print("✓ Gate scales match!")
        return True


if __name__ == "__main__":
    success = test_routing_equivalence()
    sys.exit(0 if success else 1)


