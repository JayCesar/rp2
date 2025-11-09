"""Unit tests for BiLSTMClassifier.

Tests the classification variant of BiLSTMRegressor that outputs
6-way logits for CrossEntropyLoss training.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-analysis" / "blstm"))

from blstm import ModelConfig
from blstm_cross_entropy_loss import BiLSTMClassifier, NUM_CLASSES, logits_to_scores


@pytest.fixture
def simple_config():
    """Basic config for testing."""
    return ModelConfig(
        hidden_sizes=[64, 64, 64],
        input_dim=768,
        num_layers=3,
        dropout=0.1,
        aggregation="last",
        mlp_hidden=None,
        use_layer_norm=False,
        token_proj_dim=None,
    )


@pytest.fixture
def config_with_mlp():
    """Config with MLP head."""
    return ModelConfig(
        hidden_sizes=[64, 64, 64],
        input_dim=768,
        dropout=0.1,
        aggregation="mean",
        mlp_hidden=128,
        use_layer_norm=True,
        token_proj_dim=256,
    )


class TestBiLSTMClassifierArchitecture:
    """Test model architecture and initialization."""
    
    def test_initialization(self, simple_config):
        """Model initializes without errors."""
        model = BiLSTMClassifier(simple_config)
        assert model is not None
        assert model.config == simple_config
    
    def test_has_required_components(self, simple_config):
        """Model has all required layers."""
        model = BiLSTMClassifier(simple_config)
        assert hasattr(model, "lstm1")
        assert hasattr(model, "lstm2")
        assert hasattr(model, "lstm3")
        assert hasattr(model, "dropout")
        assert hasattr(model, "head")
        assert hasattr(model, "pre_head_norm")
        assert hasattr(model, "token_proj")
    
    def test_head_output_size(self, simple_config):
        """Classification head outputs 6 classes."""
        model = BiLSTMClassifier(simple_config)
        # Simple head is Linear(hidden, NUM_CLASSES)
        assert model.head.out_features == NUM_CLASSES
    
    def test_mlp_head_structure(self, config_with_mlp):
        """MLP head has correct structure."""
        model = BiLSTMClassifier(config_with_mlp)
        # MLP head is Sequential with final Linear outputting NUM_CLASSES
        assert isinstance(model.head, torch.nn.Sequential)
        # Last layer should output NUM_CLASSES
        last_layer = list(model.head.children())[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == NUM_CLASSES
    
    def test_aggregation_options(self, simple_config):
        """Different aggregation methods initialize correctly."""
        for agg in ["last", "mean", "max", "attn"]:
            config = ModelConfig(
                hidden_sizes=[32, 32, 32],
                aggregation=agg,
            )
            model = BiLSTMClassifier(config)
            if agg == "attn":
                assert model.aggregation is not None
            else:
                assert model.aggregation is None


class TestBiLSTMClassifierForward:
    """Test forward pass behavior."""
    
    def test_forward_output_shape(self, simple_config):
        """Forward returns correct logits shape."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        batch_size = 4
        max_seq_len = 10
        tokens = torch.randn(batch_size, max_seq_len, 768)
        lengths = torch.tensor([10, 8, 6, 5])
        
        logits = model(tokens, lengths)
        
        assert logits.shape == (batch_size, NUM_CLASSES)
    
    def test_forward_dtype(self, simple_config):
        """Forward outputs float tensors."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        tokens = torch.randn(2, 5, 768)
        lengths = torch.tensor([5, 3])
        
        logits = model(tokens, lengths)
        
        assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16]
    
    def test_forward_variable_lengths(self, simple_config):
        """Forward handles variable sequence lengths."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        batch_size = 8
        max_seq_len = 20
        tokens = torch.randn(batch_size, max_seq_len, 768)
        # Different lengths for each sequence
        lengths = torch.tensor([20, 18, 15, 12, 10, 8, 5, 3])
        
        logits = model(tokens, lengths)
        
        assert logits.shape == (batch_size, NUM_CLASSES)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_forward_single_sample(self, simple_config):
        """Forward works with batch size 1."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        tokens = torch.randn(1, 12, 768)
        lengths = torch.tensor([12])
        
        logits = model(tokens, lengths)
        
        assert logits.shape == (1, NUM_CLASSES)
    
    def test_forward_deterministic(self, simple_config):
        """Forward is deterministic in eval mode."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        tokens = torch.randn(3, 8, 768)
        lengths = torch.tensor([8, 6, 4])
        
        with torch.no_grad():
            logits1 = model(tokens, lengths)
            logits2 = model(tokens, lengths)
        
        assert torch.allclose(logits1, logits2)
    
    def test_forward_with_token_projection(self, config_with_mlp):
        """Forward works with token projection."""
        model = BiLSTMClassifier(config_with_mlp)
        model.eval()
        
        tokens = torch.randn(2, 10, 768)
        lengths = torch.tensor([10, 7])
        
        logits = model(tokens, lengths)
        
        assert logits.shape == (2, NUM_CLASSES)
    
    def test_backward_pass(self, simple_config):
        """Backward pass computes gradients."""
        model = BiLSTMClassifier(simple_config)
        model.train()
        
        tokens = torch.randn(2, 5, 768)
        lengths = torch.tensor([5, 4])
        targets = torch.tensor([0, 3])  # Class indices
        
        logits = model(tokens, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        
        # Check gradients exist
        assert any(p.grad is not None for p in model.parameters())


class TestBiLSTMClassifierPredictScores:
    """Test convenience predict_scores method."""
    
    def test_predict_scores_output_shape(self, simple_config):
        """predict_scores returns scores with correct shape."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        batch_size = 4
        tokens = torch.randn(batch_size, 10, 768)
        lengths = torch.tensor([10, 8, 6, 4])
        
        with torch.no_grad():
            scores = model.predict_scores(tokens, lengths)
        
        assert scores.shape == (batch_size,)
    
    def test_predict_scores_valid_values(self, simple_config):
        """predict_scores returns valid C1 scores."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        tokens = torch.randn(5, 8, 768)
        lengths = torch.tensor([8, 7, 6, 5, 4])
        
        with torch.no_grad():
            scores = model.predict_scores(tokens, lengths)
        
        valid_scores = {0, 40, 80, 120, 160, 200}
        assert all(s.item() in valid_scores for s in scores)
    
    def test_predict_scores_uses_logits_to_scores(self, simple_config):
        """predict_scores correctly maps logits to scores."""
        model = BiLSTMClassifier(simple_config)
        model.eval()
        
        tokens = torch.randn(3, 6, 768)
        lengths = torch.tensor([6, 5, 4])
        
        with torch.no_grad():
            logits = model(tokens, lengths)
            scores_direct = logits_to_scores(logits)
            scores_method = model.predict_scores(tokens, lengths)
        
        assert torch.equal(scores_direct, scores_method)


class TestBiLSTMClassifierGradients:
    """Test gradient flow and training behavior."""
    
    def test_all_parameters_trainable(self, simple_config):
        """All parameters require gradients."""
        model = BiLSTMClassifier(simple_config)
        
        trainable = [p for p in model.parameters() if p.requires_grad]
        all_params = list(model.parameters())
        
        assert len(trainable) == len(all_params)
        assert len(trainable) > 0
    
    def test_gradient_flow_through_all_layers(self, simple_config):
        """Gradients flow to all layers."""
        model = BiLSTMClassifier(simple_config)
        model.train()
        
        tokens = torch.randn(2, 5, 768)
        lengths = torch.tensor([5, 4])
        targets = torch.tensor([1, 4])
        
        logits = model(tokens, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        
        # Check LSTM layers have gradients
        assert model.lstm1.weight_ih_l0.grad is not None
        assert model.lstm2.weight_ih_l0.grad is not None
        assert model.lstm3.weight_ih_l0.grad is not None
        
        # Check head has gradients
        if isinstance(model.head, torch.nn.Linear):
            assert model.head.weight.grad is not None
        else:
            # MLP head - check last layer
            last_linear = [m for m in model.head.modules() if isinstance(m, torch.nn.Linear)][-1]
            assert last_linear.weight.grad is not None


class TestBiLSTMClassifierDeviceHandling:
    """Test device compatibility."""
    
    def test_cpu_forward(self, simple_config):
        """Model works on CPU."""
        model = BiLSTMClassifier(simple_config).to("cpu")
        model.eval()
        
        tokens = torch.randn(2, 5, 768, device="cpu")
        lengths = torch.tensor([5, 4], device="cpu")
        
        with torch.no_grad():
            logits = model(tokens, lengths)
        
        assert logits.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, simple_config):
        """Model works on CUDA."""
        model = BiLSTMClassifier(simple_config).to("cuda")
        model.eval()
        
        tokens = torch.randn(2, 5, 768, device="cuda")
        lengths = torch.tensor([5, 4], device="cuda")
        
        with torch.no_grad():
            logits = model(tokens, lengths)
        
        assert logits.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
