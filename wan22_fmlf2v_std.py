from comfy.sd import VAE
import node_helpers
import comfy.utils
import comfy.clip_vision
import torch
import torch.nn.functional as F
import comfy.model_management


class FossielWanFirstMiddleLastFrameToVideoTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "high_noise_mid_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "low_noise_start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "low_noise_mid_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "low_noise_end_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "structural_repulsion_boost": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(
        self,
        positive,
        negative,
        vae: VAE,
        width,
        height,
        length,
        batch_size,
        tile_size,
        overlap,
        temporal_size=64,
        temporal_overlap=8,
        mode="NORMAL",
        start_image=None,
        middle_image=None,
        end_image=None,
        middle_frame_ratio=0.5,
        high_noise_mid_strength=0.8,
        low_noise_start_strength=1.0,
        low_noise_mid_strength=0.2,
        low_noise_end_strength=1.0,
        structural_repulsion_boost=1.0,
        clip_vision_start_image=None,
        clip_vision_middle_image=None,
        clip_vision_end_image=None,
    ):
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1

        device = comfy.model_management.intermediate_device()

        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale],
            device=device
        )

        # Upscale input images to target resolution
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, height, width), device=device)

        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask_high_noise[:, :, :start_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, :start_image.shape[0] + 3] = 1.0 - low_noise_start_strength

        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image
            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)
            mask_high_noise[:, :, start_range:end_range] = 1.0 - high_noise_mid_strength
            mask_low_noise[:, :, start_range:end_range] = 1.0 - low_noise_mid_strength

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
            mask_low_noise[:, :, -end_image.shape[0]:] = 1.0 - low_noise_end_strength

        # === VAE TILING REPLACEMENT (only change from original) ===
        concat_latent_image = vae.encode_tiled(
            image[:, :, :, :3],
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
            tile_t=temporal_size,
            overlap_t=temporal_overlap
        )

        # Structural repulsion boost (unchanged)
        if structural_repulsion_boost > 1.001 and length > 4:
            mask_h, mask_w = height, width
            boost_factor = structural_repulsion_boost - 1.0

            def create_spatial_gradient(img1, img2):
                if img1 is None or img2 is None:
                    return None
                motion_diff = torch.abs(img2[0] - img1[0]).mean(dim=-1, keepdim=False)
                motion_diff_4d = motion_diff.unsqueeze(0).unsqueeze(0)
                motion_diff_scaled = F.interpolate(motion_diff_4d, size=(mask_h, mask_w), mode='bilinear', align_corners=False)
                motion_normalized = (motion_diff_scaled - motion_diff_scaled.min()) / (motion_diff_scaled.max() - motion_diff_scaled.min() + 1e-8)
                spatial_gradient = 1.0 - motion_normalized * boost_factor * 2.5
                return torch.clamp(spatial_gradient, 0.02, 1.0)[0, 0]

            # ... (exact same logic as original - omitted for brevity but fully preserved below)

            if start_image is not None and middle_image is not None:
                spatial_gradient_1 = create_spatial_gradient(start_image[0:1], middle_image[0:1])
                if spatial_gradient_1 is not None:
                    start_end = start_image.shape[0] + 3
                    mid_protect_start = max(start_end, middle_idx - 4)
                    for f in range(start_end, min(mid_protect_start, length)):
                        mask_high_noise[:, :, f] *= spatial_gradient_1

            if middle_image is not None and end_image is not None:
                spatial_gradient_2 = create_spatial_gradient(middle_image[0:1], end_image[-1:])
                if spatial_gradient_2 is not None:
                    transition_start = middle_idx + 5
                    end_start = length - end_image.shape[0]
                    for f in range(transition_start, end_start):
                        mask_high_noise[:, :, f] *= spatial_gradient_2

            if start_image is not None and end_image is not None and middle_image is None:
                spatial_gradient = create_spatial_gradient(start_image[0:1], end_image[-1:])
                if spatial_gradient is not None:
                    start_end = start_image.shape[0] + 3
                    end_start = length - end_image.shape[0]
                    for f in range(start_end, end_start):
                        mask_high_noise[:, :, f] *= spatial_gradient

        # Low-noise branch
        if mode == "SINGLE_PERSON":
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if start_image is not None:
                image_low_only[:start_image.shape[0]] = start_image
            concat_latent_image_low = vae.encode_tiled(image_low_only[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if start_image is not None and low_noise_start_strength > 0.0:
                image_low_only[:start_image.shape[0]] = start_image
            if middle_image is not None and low_noise_mid_strength > 0.0:
                image_low_only[middle_idx:middle_idx + 1] = middle_image
            if end_image is not None and low_noise_end_strength > 0.0:
                image_low_only[-end_image.shape[0]:] = end_image
            concat_latent_image_low = vae.encode_tiled(image_low_only[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        else:
            concat_latent_image_low = concat_latent_image  # reuse high-noise version

        # Reshape masks to latent temporal dimension
        mask_high_reshaped = mask_high_noise.view(1, latent_t, 4, height, width).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, latent_t, 4, height, width).transpose(1, 2)

        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })

        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
        })

        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })

        # CLIP_VISION merging (unchanged)
        clip_vision_output = self._merge_clip_vision_outputs(clip_vision_start_image, clip_vision_middle_image, clip_vision_end_image)
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})
            negative_out = node_helpers.conditioning_set_values(negative_out, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}

        return (positive_high_noise, positive_low_noise, negative_out, out_latent)

    @staticmethod
    def _calculate_aligned_position(ratio, total_frames):
        desired_idx = int(total_frames * ratio)
        aligned_idx = (desired_idx // 4) * 4
        return max(0, min(aligned_idx, total_frames - 1))

    @staticmethod
    def _merge_clip_vision_outputs(cls, *outputs):
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            return None
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined = torch.cat(all_states, dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined
        return result
