"""Unit tests for CE score↔class mapping utilities."""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-analysis" / "blstm"))

from blstm_ce import (
    score_to_class_idx,
    class_idx_to_score,
    scores_to_class_indices,
    class_indices_to_scores,
    logits_to_scores,
    validate_scores_for_ce,
    VALID_SCORES,
    NUM_CLASSES,
)


class TestScalarMappings:
    """Test scalar score↔class conversions."""
    
    def test_score_to_class_all_valid(self):
        """Test score_to_class_idx for all valid scores."""
        expected = {0: 0, 40: 1, 80: 2, 120: 3, 160: 4, 200: 5}
        for score, expected_class in expected.items():
            assert score_to_class_idx(score) == expected_class
    
    def test_class_to_score_all_valid(self):
        """Test class_idx_to_score for all valid indices."""
        expected = {0: 0, 1: 40, 2: 80, 3: 120, 4: 160, 5: 200}
        for class_idx, expected_score in expected.items():
            assert class_idx_to_score(class_idx) == expected_score
    
    def test_roundtrip_score_to_class_to_score(self):
        """Test score→class→score roundtrip."""
        for score in sorted(VALID_SCORES):
            class_idx = score_to_class_idx(score)
            recovered_score = class_idx_to_score(class_idx)
            assert recovered_score == score, f"Roundtrip failed for score {score}"
    
    def test_roundtrip_class_to_score_to_class(self):
        """Test class→score→class roundtrip."""
        for class_idx in range(NUM_CLASSES):
            score = class_idx_to_score(class_idx)
            recovered_class = score_to_class_idx(score)
            assert recovered_class == class_idx, f"Roundtrip failed for class {class_idx}"
    
    def test_score_to_class_invalid_raises(self):
        """Test that invalid scores raise ValueError."""
        invalid_scores = [-40, 10, 30, 50, 90, 220, 1000]
        for invalid in invalid_scores:
            with pytest.raises(ValueError, match="Invalid score"):
                score_to_class_idx(invalid)
    
    def test_class_to_score_invalid_raises(self):
        """Test that invalid class indices raise ValueError."""
        invalid_indices = [-1, 6, 7, 10, 100]
        for invalid in invalid_indices:
            with pytest.raises(ValueError, match="Invalid class index"):
                class_idx_to_score(invalid)


class TestVectorizedMappings:
    """Test vectorized tensor score↔class conversions."""
    
    def test_scores_to_indices_all_valid(self):
        """Test scores_to_class_indices for all valid scores."""
        scores = torch.tensor([0, 40, 80, 120, 160, 200])
        expected = torch.tensor([0, 1, 2, 3, 4, 5])
        result = scores_to_class_indices(scores)
        assert torch.equal(result, expected)
        assert result.dtype == torch.int64
    
    def test_indices_to_scores_all_valid(self):
        """Test class_indices_to_scores for all valid indices."""
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        expected = torch.tensor([0, 40, 80, 120, 160, 200])
        result = class_indices_to_scores(indices)
        assert torch.equal(result, expected)
        assert result.dtype == torch.int64
    
    def test_roundtrip_scores_to_indices_to_scores(self):
        """Test vectorized score→class→score roundtrip."""
        scores = torch.tensor([0, 40, 80, 120, 160, 200, 0, 120, 200])
        indices = scores_to_class_indices(scores)
        recovered = class_indices_to_scores(indices)
        assert torch.equal(recovered, scores)
    
    def test_roundtrip_indices_to_scores_to_indices(self):
        """Test vectorized class→score→class roundtrip."""
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 0, 3, 5])
        scores = class_indices_to_scores(indices)
        recovered = scores_to_class_indices(scores)
        assert torch.equal(recovered, indices)
    
    def test_scores_to_indices_multidimensional(self):
        """Test scores_to_class_indices with 2D tensor."""
        scores = torch.tensor([[0, 40], [120, 200]])
        expected = torch.tensor([[0, 1], [3, 5]])
        result = scores_to_class_indices(scores)
        assert torch.equal(result, expected)
    
    def test_indices_to_scores_multidimensional(self):
        """Test class_indices_to_scores with 2D tensor."""
        indices = torch.tensor([[0, 1], [3, 5]])
        expected = torch.tensor([[0, 40], [120, 200]])
        result = class_indices_to_scores(indices)
        assert torch.equal(result, expected)
    
    def test_scores_to_indices_invalid_raises(self):
        """Test that invalid scores in tensor raise ValueError."""
        invalid_tensors = [
            torch.tensor([0, 50, 120]),  # 50 is invalid
            torch.tensor([10, 40, 80]),  # 10 is invalid
            torch.tensor([0, 40, 220]),  # 220 is invalid
        ]
        for invalid in invalid_tensors:
            with pytest.raises(ValueError, match="Invalid scores found"):
                scores_to_class_indices(invalid)
    
    def test_indices_to_scores_invalid_raises(self):
        """Test that invalid indices in tensor raise ValueError."""
        invalid_tensors = [
            torch.tensor([0, 6, 3]),  # 6 is invalid
            torch.tensor([-1, 1, 2]),  # -1 is invalid
            torch.tensor([0, 1, 10]),  # 10 is invalid
        ]
        for invalid in invalid_tensors:
            with pytest.raises(ValueError, match="Invalid class indices found"):
                class_indices_to_scores(invalid)
    
    def test_scores_cuda_cpu_consistency(self):
        """Test that CUDA and CPU give same results (if CUDA available)."""
        scores_cpu = torch.tensor([0, 40, 120, 200])
        result_cpu = scores_to_class_indices(scores_cpu)
        
        if torch.cuda.is_available():
            scores_cuda = scores_cpu.cuda()
            result_cuda = scores_to_class_indices(scores_cuda)
            assert torch.equal(result_cpu, result_cuda.cpu())


class TestLogitsToScores:
    """Test logits→scores conversion (inference function)."""
    
    def test_logits_to_scores_basic(self):
        """Test logits_to_scores with simple logits."""
        # Create logits where argmax gives known class indices
        logits = torch.zeros(4, NUM_CLASSES)
        logits[0, 0] = 10.0  # Class 0 → score 0
        logits[1, 1] = 10.0  # Class 1 → score 40
        logits[2, 3] = 10.0  # Class 3 → score 120
        logits[3, 5] = 10.0  # Class 5 → score 200
        
        expected_scores = torch.tensor([0, 40, 120, 200])
        result = logits_to_scores(logits)
        
        assert torch.equal(result, expected_scores)
        assert result.dtype == torch.int64
        assert result.shape == (4,)
    
    def test_logits_to_scores_all_classes(self):
        """Test that all possible classes can be predicted."""
        batch_size = NUM_CLASSES
        logits = torch.zeros(batch_size, NUM_CLASSES)
        
        # Set each batch element to predict different class
        for i in range(NUM_CLASSES):
            logits[i, i] = 10.0
        
        result = logits_to_scores(logits)
        expected = torch.tensor([0, 40, 80, 120, 160, 200])
        
        assert torch.equal(result, expected)
    
    def test_logits_to_scores_domain(self):
        """Test that logits_to_scores always returns valid scores."""
        # Random logits
        torch.manual_seed(42)
        logits = torch.randn(100, NUM_CLASSES)
        
        scores = logits_to_scores(logits)
        
        # All scores should be in valid set
        for score in scores.tolist():
            assert score in VALID_SCORES


class TestValidateScoresForCE:
    """Test score validation utility."""
    
    def test_validate_all_valid_tensor(self):
        """Test validation passes for valid scores in tensor."""
        valid_scores = torch.tensor([0, 40, 80, 120, 160, 200])
        # Should not raise
        validate_scores_for_ce(valid_scores)
    
    def test_validate_all_valid_list(self):
        """Test validation passes for valid scores in list."""
        valid_scores = [0, 40, 80, 120, 160, 200]
        # Should not raise
        validate_scores_for_ce(valid_scores)
    
    def test_validate_invalid_tensor_raises(self):
        """Test validation raises for invalid scores in tensor."""
        invalid_scores = torch.tensor([0, 40, 50, 120])  # 50 is invalid
        with pytest.raises(ValueError, match="Found invalid C1 scores"):
            validate_scores_for_ce(invalid_scores)
    
    def test_validate_invalid_list_raises(self):
        """Test validation raises for invalid scores in list."""
        invalid_scores = [0, 40, 100, 200]  # 100 is invalid
        with pytest.raises(ValueError, match="Found invalid C1 scores"):
            validate_scores_for_ce(invalid_scores)
    
    def test_validate_empty_passes(self):
        """Test validation passes for empty input."""
        validate_scores_for_ce(torch.tensor([]))
        validate_scores_for_ce([])


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_ce_mapping.py -v
    pytest.main([__file__, "-v"])
