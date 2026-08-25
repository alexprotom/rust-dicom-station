#!/usr/bin/env python3
"""Dump a reference SAM 2.1-T @ 512 run: weights in, activations out.

This is the harness the MedSAM2 port is validated against. It builds the
*reference* implementation with **randomly initialized** weights (so nothing
license-encumbered is involved and no 156 MB download is needed), runs every
module the port implements, and writes both the weights and the activations
as safetensors. The Rust side then loads the same weights and must reproduce
the same numbers.

Random weights are as strong a test as trained ones — stronger, if anything,
since a mistake cannot hide behind a plausible-looking mask — and they make
the whole check reproducible from a seed.

The output is large (the weights alone are 156 MB) and is **not** committed;
regenerate it when you need it:

    python3 tools/gen_reference_activations.py /tmp/ref

writes /tmp/ref-weights.safetensors and /tmp/ref-acts.safetensors.
"""

import sys

import torch

from safetensors.torch import save_file
from sam2.build_sam import build_sam2

SEED = 20260825
IMAGE_SIZE = 512


def main(stem):
    torch.manual_seed(SEED)
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=None,
        device="cpu",
        mode="eval",
        hydra_overrides_extra=[
            f"++model.image_size={IMAGE_SIZE}",
            "++model.memory_attention.layer.self_attention.feat_sizes=[32,32]",
            "++model.memory_attention.layer.cross_attention.feat_sizes=[32,32]",
            # MedSAM2 runs through the *video* predictor, whose
            # apply_postprocessing adds this one on top of what the image
            # builder sets. It is what makes a prompted slice's mask hard
            # binarized at zero before it reaches the memory encoder.
            "++model.binarize_mask_from_pts_for_mem_enc=true",
        ],
        apply_postprocessing=True,
    )
    model.eval()

    state = {k: v.detach().clone().float().contiguous() for k, v in model.state_dict().items()}
    save_file(state, f"{stem}-weights.safetensors")
    n = sum(v.numel() for v in state.values())
    print(f"weights: {len(state)} tensors, {n} elements")

    acts = {}

    def put(name, value):
        acts[name] = value.detach().clone().float().contiguous()

    g = torch.Generator().manual_seed(SEED + 1)
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g) * 0.7
    put("img", img)

    with torch.no_grad():
        # ---- trunk, stage by stage --------------------------------------
        trunk_out = model.image_encoder.trunk(img)
        for i, t in enumerate(trunk_out):
            put(f"trunk.{i}", t)

        # ---- neck, before the image encoder's scalp ----------------------
        neck_out, neck_pos = model.image_encoder.neck(trunk_out)
        for i, t in enumerate(neck_out):
            put(f"neck.{i}", t)
        put("neck_pos.2", neck_pos[2])

        # ---- the image encoder as the tracker calls it -------------------
        backbone_out = model.forward_image(img)
        put("vision_features", backbone_out["vision_features"])
        put("high_res_s0", backbone_out["backbone_fpn"][0])
        put("high_res_s1", backbone_out["backbone_fpn"][1])

        _, vision_feats, vision_pos, feat_sizes = model._prepare_backbone_features(
            backbone_out
        )
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        pix_feat = (
            vision_feats[-1]
            .permute(1, 2, 0)
            .view(1, -1, feat_sizes[-1][0], feat_sizes[-1][1])
        )
        put("pix_feat", pix_feat)

        # ---- prompt encoder ---------------------------------------------
        # a box, the way the video predictor passes one: two corner points
        # labelled 2 and 3, and `pad=True` appends a not-a-point.
        box = torch.tensor([[120.0, 200.0], [300.0, 380.0]]).unsqueeze(0)
        labels = torch.tensor([[2, 3]], dtype=torch.int32)
        put("prompt_box_coords", box)
        sparse, dense = model.sam_prompt_encoder(
            points=(box, labels), boxes=None, masks=None
        )
        put("prompt_sparse_box", sparse)
        put("prompt_dense_none", dense)
        put("prompt_dense_pe", model.sam_prompt_encoder.get_dense_pe())

        # no prompt at all — what a propagated slice sends
        empty_coords = torch.zeros(1, 1, 2)
        empty_labels = -torch.ones(1, 1, dtype=torch.int32)
        sparse_empty, _ = model.sam_prompt_encoder(
            points=(empty_coords, empty_labels), boxes=None, masks=None
        )
        put("prompt_sparse_empty", sparse_empty)

        # a mask prompt, downsampled by mask_downsample first
        mask_in = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        put("mask_prompt_in", mask_in)
        mask_small = model.mask_downsample(mask_in)
        put("mask_prompt_downsampled", mask_small)
        _, dense_mask = model.sam_prompt_encoder(
            points=(empty_coords, empty_labels), boxes=None, masks=mask_small
        )
        put("prompt_dense_mask", dense_mask)

        # ---- mask decoder, box prompt (multimask_output = False) ---------
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs={"point_coords": box, "point_labels": labels},
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
        )
        put("sam_box.low_res_multimasks", low_res_multimasks)
        put("sam_box.ious", ious)
        put("sam_box.low_res_masks", low_res_masks)
        put("sam_box.high_res_masks", high_res_masks)
        put("sam_box.obj_ptr", obj_ptr)
        put("sam_box.object_score_logits", object_score_logits)

        # ---- mask decoder, tracking (no prompt, multimask_output = True) --
        (
            low2,
            _,
            ious2,
            low_res_masks2,
            high_res_masks2,
            obj_ptr2,
            obj_score2,
        ) = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=True,
        )
        put("sam_track.low_res_multimasks", low2)
        put("sam_track.ious", ious2)
        put("sam_track.low_res_masks", low_res_masks2)
        put("sam_track.high_res_masks", high_res_masks2)
        put("sam_track.obj_ptr", obj_ptr2)
        put("sam_track.object_score_logits", obj_score2)

        # ---- the mask decoder on its own ---------------------------------
        # `_forward_sam_heads` replaces every logit with NO_OBJ_SCORE when the
        # object-score head is negative, which with random weights it usually
        # is — so the decoder's own output is dumped too, or the whole
        # transformer would go unchecked.
        raw_low, raw_ious, raw_tokens, raw_obj = model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        put("dec_box.masks", raw_low)
        put("dec_box.ious", raw_ious)
        put("dec_box.tokens", raw_tokens)
        put("dec_box.obj_score", raw_obj)
        put("dec_box.obj_ptr", model.obj_ptr_proj(raw_tokens[:, 0]))

        m_low, m_ious, m_tokens, _ = model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_empty,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        put("dec_track.masks", m_low)
        put("dec_track.ious", m_ious)
        put("dec_track.tokens", m_tokens)

        # ---- memory encoder ----------------------------------------------
        maskmem_features, maskmem_pos_enc = model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )
        put("memenc.features", maskmem_features)
        put("memenc.pos", maskmem_pos_enc[0])

        # the same, from a propagated slice: not binarized, and the object
        # may be absent
        maskmem_features2, _ = model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks2,
            object_score_logits=obj_score2,
            is_mask_from_pts=False,
        )
        put("memenc.features_soft", maskmem_features2)

        # ---- memory attention ---------------------------------------------
        tokens = feat_sizes[-1][0] * feat_sizes[-1][1]
        curr = torch.randn(tokens, 1, 256, generator=g) * 0.3
        curr_pos = torch.randn(tokens, 1, 256, generator=g) * 0.3
        n_ptr = 8
        memory = torch.randn(tokens * 3 + n_ptr, 1, 64, generator=g) * 0.3
        memory_pos = torch.randn(tokens * 3 + n_ptr, 1, 64, generator=g) * 0.3
        put("memattn.curr", curr)
        put("memattn.curr_pos", curr_pos)
        put("memattn.memory", memory)
        put("memattn.memory_pos", memory_pos)
        put(
            "memattn.out",
            model.memory_attention(
                curr=curr,
                curr_pos=curr_pos,
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=n_ptr,
            ),
        )
        # one memory frame, no pointers — the first propagated slice
        memory1 = torch.randn(tokens, 1, 64, generator=g) * 0.3
        memory1_pos = torch.randn(tokens, 1, 64, generator=g) * 0.3
        put("memattn1.memory", memory1)
        put("memattn1.memory_pos", memory1_pos)
        put(
            "memattn1.out",
            model.memory_attention(
                curr=curr,
                curr_pos=curr_pos,
                memory=memory1,
                memory_pos=memory1_pos,
                num_obj_ptr_tokens=0,
            ),
        )

        # ---- memory encoder on a mask that is actually varied -------------
        # With random weights the object-score head is usually negative, so
        # the masks above are a constant -1024 and the downsampler would go
        # untested. These two force a present object and a varied mask.
        rand_mask = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        present = torch.tensor([[1.0]])
        put("memenc.mask_rand", rand_mask)
        f_hard, _ = model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=rand_mask,
            object_score_logits=present,
            is_mask_from_pts=True,
        )
        put("memenc.features_rand", f_hard)
        f_soft, _ = model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=rand_mask,
            object_score_logits=present,
            is_mask_from_pts=False,
        )
        put("memenc.features_rand_soft", f_soft)

    save_file(acts, f"{stem}-acts.safetensors")
    print(f"activations: {len(acts)} tensors")
    for k, v in acts.items():
        print(f"  {k:32s} {tuple(v.shape)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref")
