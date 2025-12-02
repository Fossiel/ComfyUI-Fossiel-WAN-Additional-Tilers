# wan22_fun_ctrl.py
import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import comfy.latent_formats


class FossielWan22FunControlToVideoTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64,
                                      "tooltip": "Tile size for VAE encoding (X and Y)."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32,
                                    "tooltip": "Overlap between spatial tiles."}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4,
                                          "tooltip": "Number of frames to encode per temporal tile."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4,
                                              "tooltip": "Overlap between temporal tiles."}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "control_video": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                ref_image=None, control_video=None):

        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_timesteps = ((length - 1) // 4) + 1

        latent = torch.zeros([batch_size, latent_channels, latent_timesteps,
                              height // spacial_scale, width // spacial_scale],
                             device=comfy.model_management.intermediate_device())

        concat_latent = torch.zeros_like(latent)
        if latent_channels == 48:
            concat_latent = comfy.latent_formats.Wan22().process_out(concat_latent)
        else:
            concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        mask = torch.ones((1, 1, latent_timesteps * 4, latent.shape[-2], latent.shape[-1]),
                          device=comfy.model_management.intermediate_device())

        # Note: start_image was never actually connected in your original schema
        # It was referenced in code but not in define_schema() → keeping behavior identical
        # (if someone passes it via wiring, it will still work; if not, it stays None)

        if "start_image" in locals() and locals()["start_image"] is not None:
            start_image = locals()["start_image"]
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            concat_latent_image = vae.encode_tiled(
                start_image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=temporal_size, overlap_t=temporal_overlap
            )
            concat_latent[:, latent_channels:, :concat_latent_image.shape[2]] = concat_latent_image[:, :, :concat_latent.shape[2]]
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        ref_latent = None
        if ref_image is not None:
            ref_image = comfy.utils.common_upscale(
                ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            ref_latent = vae.encode_tiled(
                ref_image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=1, overlap_t=0
            )

        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            concat_latent_image = vae.encode_tiled(
                control_video[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=temporal_size, overlap_t=temporal_overlap
            )
            concat_latent[:, :latent_channels, :concat_latent_image.shape[2]] = concat_latent_image[:, :, :concat_latent.shape[2]]

        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels}
        )

        if ref_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

        return (positive, negative, {"samples": latent})
