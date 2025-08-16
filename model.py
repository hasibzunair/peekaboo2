# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND, see license below.

# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model code for Peekaboo"""

import os
import torch
import torch.nn as nn
import dino.vision_transformer as vits


class PeekabooModel(nn.Module):
    def __init__(
        self,
        vit_model="dino",
        vit_arch="vit_small",
        vit_patch_size=8,
        enc_type_feats="k",
    ):

        super(PeekabooModel, self).__init__()

        ########## Encoder ##########
        self.vit_encoder, self.initial_dim, self.hook_features = get_vit_encoder(
            vit_arch, vit_model, vit_patch_size, enc_type_feats
        )
        self.vit_patch_size = vit_patch_size
        self.enc_type_feats = enc_type_feats

        ########## Decoder ##########
        self.previous_dim = self.initial_dim
        self.decoder = nn.Conv2d(self.previous_dim, 1, (1, 1))

    def _make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        # From selfmask
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size

        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward(self, batch, decoder=None, for_eval=False):

        # Make the image divisible by the patch size
        if for_eval:
            batch = self._make_input_divisible(batch)
            _w, _h = batch.shape[-2:]
            _h, _w = _h // self.vit_patch_size, _w // self.vit_patch_size
        else:
            # Cropping used during training, could be changed to improve
            w, h = (
                batch.shape[-2] - batch.shape[-2] % self.vit_patch_size,
                batch.shape[-1] - batch.shape[-1] % self.vit_patch_size,
            )
            batch = batch[:, :, :w, :h]

        w_featmap = batch.shape[-2] // self.vit_patch_size
        h_featmap = batch.shape[-1] // self.vit_patch_size

        # Forward pass
        with torch.no_grad():
            # Encoder forward pass
            att = self.vit_encoder.get_last_selfattention(batch)

            # Get decoder features
            feats = self.extract_feats(dims=att.shape, type_feats=self.enc_type_feats)
            feats = feats[:, 1:, :, :].reshape(att.shape[0], w_featmap, h_featmap, -1)
            feats = feats.permute(0, 3, 1, 2)

        # Apply decoder
        if decoder is None:
            decoder = self.decoder

        logits = decoder(feats)
        return logits

    @torch.no_grad()
    def decoder_load_weights(self, weights_path):
        print(f"Loading model from weights {weights_path}.")
        # Load states
        if torch.cuda.is_available():
            state_dict = torch.load(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        # Decoder
        self.decoder.load_state_dict(state_dict["decoder"])
        self.decoder.eval()
        self.decoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @torch.no_grad()
    def decoder_save_weights(self, save_dir, n_iter):
        state_dict = {}
        state_dict["decoder"] = self.decoder.state_dict()
        fname = os.path.join(save_dir, f"decoder_weights_niter{n_iter}.pt")
        torch.save(state_dict, fname)
        print(f"\n----" f"\nModel saved at {fname}")

    @torch.no_grad()
    def extract_feats(self, dims, type_feats="k"):

        nb_im, nh, nb_tokens, _ = dims
        qkv = (
            self.hook_features["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)  # 3 corresponding to |qkv|
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")


def get_vit_encoder(vit_arch, vit_model, vit_patch_size, enc_type_feats):
    if vit_arch == "vit_small" and vit_patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_small" and vit_patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_base" and vit_patch_size == 16:
        if vit_model == "clip":
            url = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        elif vit_model == "dino":
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        initial_dim = 768
    elif vit_arch == "vit_base" and vit_patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        initial_dim = 768

    if vit_model == "dino":
        vit_encoder = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0)
        # TODO change if want to have last layer not unfrozen
        for p in vit_encoder.parameters():
            p.requires_grad = False
        vit_encoder.eval().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )  # mode eval
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        vit_encoder.load_state_dict(state_dict, strict=True)

        hook_features = {}
        if enc_type_feats in ["k", "q", "v", "qkv", "mlp"]:
            # Define the hook
            def hook_fn_forward_qkv(module, input, output):
                hook_features["qkv"] = output

            vit_encoder._modules["blocks"][-1]._modules["attn"]._modules[
                "qkv"
            ].register_forward_hook(hook_fn_forward_qkv)
    else:
        raise ValueError("Not implemented.")

    return vit_encoder, initial_dim, hook_features
