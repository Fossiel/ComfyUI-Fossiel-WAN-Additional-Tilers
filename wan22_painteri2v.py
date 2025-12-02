import torch
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


class FossielPainterI2VTiled:
    """
    An enhanced Wan2.2 Image-to-Video node specifically designed to fix the slow-motion issue in 4-step LoRAs (like lightx2v).
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
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),

                # VAE tiling controls – exactly like the reference nodes
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4,
                                          "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4,
                                            "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                start_image=None, clip_vision_output=None):

        # 1. Strict zero latent initialization (lifeline for 4-step LoRAs)
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())

        if start_image is not None:
            # Single frame input handling
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            # Create sequence: first frame real, the rest 0.5 gray
            image = torch.ones((length, height, width, start_image.shape[-1]),
                               device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]

            #TILED ENCODE – only thing changed
            concat_latent_image = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap
            )

            # Single-frame mask: only constrain the first frame
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2],
                               concat_latent_image.shape[-1]),
                              device=start_image.device, dtype=start_image.dtype)
            mask[:, :, 0] = 0.0

            # 2. Motion amplitude enhancement (brightness protection core algorithm)
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]      # first frame
                gray_latent = concat_latent_image[:, :, 1:]       # gray frames

                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean

                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)

            # 3. Inject into conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. Reference frame enhancement – also tiled
            ref_latent = vae.encode_tiled(
                start_image[:, :, :, :3],
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap
            )
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent = {"samples": latent}
        return (positive, negative, out_latent)


class FossielPainterI2VExtensionTiled(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FossielPainterI2VTiled]

def comfy_entrypoint() -> FossielPainterI2VExtensionTiled:
    return FossielPainterI2VExtensionTiled()
