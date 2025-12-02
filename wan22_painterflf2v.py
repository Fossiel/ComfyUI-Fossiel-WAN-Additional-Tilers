import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
import torch.nn.functional as F
import nodes
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


class FossielPainterFLF2VTiled:
    """
    PainterFLF2V boosts first-last frame motion with inverse structural repulsion.
    Dynamically enhances the original first-last frame node, allowing you to customize the dynamic enhancement intensity.
    Now with full VAE tiling support for huge resolutions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                # Modified: restrict range 1.0 - 2.0
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "1.0=official version, 2.0=extreme speed (eliminates slow motion)"}),

                # === VAE Tiling controls – identical to reference ===
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4,
                                          "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4,
                                            "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.0,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                start_image=None, end_image=None,
                clip_vision_start_image=None, clip_vision_end_image=None):

        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1

        # Initialize Latent
        latent = torch.zeros([batch_size, vae.latent_channels, latent_frames, height // spacial_scale, width // spacial_scale],
                             device=mm.intermediate_device())

        # 1. Image preprocessing
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        # 2. Build baseline latent
        # [Official baseline]: middle filled with 0.5 (gray)
        official_image = torch.ones((length, height, width, 3), device=mm.intermediate_device()) * 0.5
        mask = torch.ones((1, 1, latent_frames * 4, height // spacial_scale, width // spacial_scale), device=mm.intermediate_device())

        if start_image is not None:
            official_image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0
        if end_image is not None:
            official_image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        # <<< ONLY CHANGE: replaced vae.encode() with tiled version >>>
        official_latent = vae.encode_tiled(
            official_image[:, :, :, :3],
            tile_x=tile_size, tile_y=tile_size,
            overlap=overlap,
            tile_t=temporal_size, overlap_t=temporal_overlap
        )

        # [Linear baseline]: used to compute "slow motion" features
        if start_image is not None and end_image is not None and length > 2:
            start_l = official_latent[:, :, 0:1]
            end_l   = official_latent[:, :, -1:]
            t = torch.linspace(0.0, 1.0, official_latent.shape[2], device=official_latent.device).view(1, 1, -1, 1, 1)
            linear_latent = start_l * (1 - t) + end_l * t
        else:
            linear_latent = official_latent

        # ==================== Core algorithm: Inverse Structural Repulsion ====================

        # Only trigger enhancement when amplitude > 1.0
        if length > 2 and motion_amplitude > 1.001 and start_image is not None and end_image is not None:

            # A. Compute difference vector (Anti-Ghost Vector)
            # diff = official(gray) - linear(PPT)
            # This vector actually contains the information needed to "remove PPT ghosting"
            diff = official_latent - linear_latent

            # B. Frequency separation (absolute color protection)
            h, w = diff.shape[-2], diff.shape[-1]
            # Extract low frequency (color)
            low_freq_diff = F.interpolate(diff.view(-1, vae.latent_channels, h, w),
                                         size=(h // 8, w // 8), mode='area')
            low_freq_diff = F.interpolate(low_freq_diff, size=(h, w), mode='bilinear')
            low_freq_diff = low_freq_diff.view_as(diff)

            # Extract high frequency (structure/ghosting)
            high_freq_diff = diff - low_freq_diff

            # C. Aggressive boost coefficient calculation
            # Map user input 1.0-2.0 → internal strength 0.0-4.0
            # Previous versions felt weak because the coefficient was too small. Now 2.0 = 4× strength.
            boost_scale = (motion_amplitude - 1.0) * 4.0

            # D. Final synthesis
            # Base: official latent (guarantees consistency at 1.0)
            # Boost: high frequency difference × strength
            # Note: we completely discard enhancement of low_freq_diff → colors never move
            concat_latent_image = official_latent + (high_freq_diff * boost_scale)

        else:
            # 1.0 mode: output official latent directly
            concat_latent_image = official_latent

        # ========================================================================

        # Adjust mask format
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

        # Inject into conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # Clip Vision handling (unchanged)
        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states,
                                   clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return (positive, negative, out_latent)


class FossielPainterFLF2VTiledExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FossielPainterFLF2VTiled]

async def comfy_entrypoint() -> FossielPainterFLF2VTiledExtension:
    return FossielPainterFLF2VTiledExtension()
