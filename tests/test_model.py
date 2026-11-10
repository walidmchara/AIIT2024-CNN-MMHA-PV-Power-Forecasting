import torch
from aiit_cnn_mmha.model import CNNMaskedAttention

def test_published_dimensions_forward():
    net=CNNMaskedAttention().eval()
    assert net(torch.randn(2,9,9)).shape==(2,)
    assert net.cnn[-2].out_channels==10
    assert net.attention.num_heads==20

def test_causality_all_previous_positions_unchanged_by_future_input():
    torch.manual_seed(9)
    net=CNNMaskedAttention(conv_channels=(4,6,2),model_dim=20,heads=4,ff_hidden=10).eval()
    x=torch.randn(1,7,9)
    a=net.encode(x)
    x[:,5:,:]+=torch.randn_like(x[:,5:,:])*100
    b=net.encode(x)
    assert torch.allclose(a[:,:5],b[:,:5],atol=2e-6,rtol=2e-5)

def test_invalid_input_dim_rejected():
    net=CNNMaskedAttention(conv_channels=(4,6,2),model_dim=20,heads=4,ff_hidden=10)
    import pytest
    with pytest.raises(RuntimeError):net(torch.randn(2,4,8))

def test_reference_model_families():
    from aiit_cnn_mmha.baselines import make_baseline
    for name in ['mlp','lstm','bilstm','cnn_lstm','cnn_bilstm']:
        net=make_baseline(name,9,hidden=8)
        assert net(torch.randn(2,10,9)).shape==(2,)
