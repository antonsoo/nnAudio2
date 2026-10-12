import pytest
import torch

from nnAudio2.features.stft import STFT, iSTFT
from nnAudio2.utils import extend_fbins


class InverseSTFT(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, spectrum, length: int):
        return self.layer.inverse(spectrum, length=length)


@pytest.mark.parametrize("n_fft", [255, 256, 257, 513])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("entry", ["inverse", "standalone"])
@pytest.mark.parametrize("scripted", [False, True])
def test_one_sided_inverse_fft_parity(n_fft, device, entry, scripted):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(83)
    waveform = torch.randn(2, 2053, device=device)
    window = torch.hann_window(n_fft, device=device)
    spectrum = torch.stft(
        waveform, n_fft, hop_length=n_fft // 4, window=window, return_complex=True
    )
    expected = torch.istft(
        spectrum, n_fft, hop_length=n_fft // 4, window=window, length=waveform.shape[-1]
    )
    packed = torch.view_as_real(spectrum)
    if entry == "inverse":
        module = STFT(n_fft=n_fft, iSTFT=True, verbose=False).to(device)
    else:
        module = iSTFT(n_fft=n_fft, verbose=False).to(device)
    if scripted:
        module = torch.jit.script(InverseSTFT(module) if entry == "inverse" else module)
    if entry == "inverse":
        actual = (
            module(packed, waveform.shape[-1])
            if scripted
            else module.inverse(packed, length=waveform.shape[-1])
        )
    else:
        actual = module(packed, onesided=True, length=waveform.shape[-1])
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, check_dtype=False
    )


@pytest.mark.parametrize("n_fft", [3, 4, 255, 256, 257])
def test_frequency_extension_matches_full_fft(n_fft):
    torch.manual_seed(52)
    frames = torch.randn(2, 3, n_fft)
    spectrum = torch.view_as_real(torch.fft.rfft(frames).transpose(1, 2))
    actual = extend_fbins(spectrum, n_fft) if n_fft % 2 else extend_fbins(spectrum)
    expected = torch.view_as_real(torch.fft.fft(frames).transpose(1, 2))
    torch.testing.assert_close(actual, expected)
