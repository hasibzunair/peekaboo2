import sys
import torch
import pytest

sys.path.insert(0, "../")
from model import PeekabooModel


# Variables
PRE_TRAINING = "dino"
ARCH = "vit_small"
PATCH_SIZE = 8
FEATS = "k"


@pytest.fixture
def peekaboo_model():
    model = PeekabooModel(
        vit_model=PRE_TRAINING,
        vit_arch=ARCH,
        vit_patch_size=PATCH_SIZE,
        enc_type_feats=FEATS,
    )
    model.decoder_load_weights("data/weights/peekaboo_decoder_weights_niter500.pt")
    return model


def test_model_initialization(peekaboo_model):
    assert isinstance(peekaboo_model, PeekabooModel)
    assert peekaboo_model.vit_patch_size == 8
    assert peekaboo_model.enc_type_feats == "k"


def test_make_input_divisible(peekaboo_model):
    img = torch.rand(1, 3, 224, 224)

    padded_img = peekaboo_model._make_input_divisible(img)

    assert padded_img.shape[-2] % peekaboo_model.vit_patch_size == 0
    assert padded_img.shape[-1] % peekaboo_model.vit_patch_size == 0


def test_model_function(peekaboo_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.rand(1, 3, 224, 224).to(device)
    pred = peekaboo_model(img, for_eval=True)

    assert pred.shape[0] == 1
    assert pred.shape[1] == 1
    assert pred.shape[2] == 224 // peekaboo_model.vit_patch_size
    assert pred.shape[3] == 224 // peekaboo_model.vit_patch_size


if __name__ == "__main__":
    test_model_function()
