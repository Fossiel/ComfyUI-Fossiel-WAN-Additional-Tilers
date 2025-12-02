import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import comfy.latent_formats


class FossielWanAnimateToVideoTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),

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
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
                "continue_motion": ("IMAGE",),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent", "trim_image", "video_frame_offset")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self,
                positive, negative, vae,
                width, height, length, batch_size,
                continue_motion_max_frames,
                tile_size=512, overlap=64,
                temporal_size=64, temporal_overlap=8,
                video_frame_offset=0,
                clip_vision_output=None,
                reference_image=None, face_video=None, pose_video=None,
                background_video=None, character_mask=None, continue_motion=None):

        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0
        ref_motion_latent_length = 0

        # ------------------------------------------------------------------
        # 1. Reference image (always exists, even if black)
        # ------------------------------------------------------------------
        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3), device="cpu")

        image = comfy.utils.common_upscale(
            reference_image[:length].movedim(-1, 1), width, height, "area", "center"
        ).movedim(1, -1)

        # TILED ENCODE (exactly like reference node)
        concat_latent_image = vae.encode_tiled(
            image[:, :, :, :3],
            tile_x=tile_size, tile_y=tile_size, overlap=overlap,
            tile_t=temporal_size, overlap_t=temporal_overlap
        )

        mask = torch.zeros((1, 4, concat_latent_image.shape[2], concat_latent_image.shape[3], concat_latent_image.shape[4]),
                           device=concat_latent_image.device, dtype=concat_latent_image.dtype)
        trim_latent += concat_latent_image.shape[2]

        # ------------------------------------------------------------------
        # 2. Continue motion handling
        # ------------------------------------------------------------------
        if continue_motion is None:
            image = torch.ones((length, height, width, 3), device="cpu") * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)

            continue_motion = comfy.utils.common_upscale(
                continue_motion[-length:].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)

            image = torch.ones((length, height, width, 3), device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        # ------------------------------------------------------------------
        # 3. CLIP Vision
        # ------------------------------------------------------------------
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # ------------------------------------------------------------------
        # 4. Pose video
        # ------------------------------------------------------------------
        if pose_video is not None:
            if pose_video.shape[0] > video_frame_offset:
                pose_video = pose_video[video_frame_offset:]
            else:
                pose_video = None

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(
                pose_video[:length].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)

            if pose_video.shape[0] < length:
                last_frame = pose_video[-1:].repeat(length - pose_video.shape[0], 1, 1, 1)
                pose_video = torch.cat((pose_video, last_frame), dim=0)

            # TILED ENCODE for pose video
            pose_video_latent = vae.encode_tiled(
                pose_video[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=temporal_size, overlap_t=temporal_overlap
            )

            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        # ------------------------------------------------------------------
        # 5. Face video (no tiling needed — it's 512×512 single frames)
        # ------------------------------------------------------------------
        if face_video is not None:
            if face_video.shape[0] > video_frame_offset:
                face_video = face_video[video_frame_offset:]
            else:
                face_video = None

        if face_video is not None:
            face_video = comfy.utils.common_upscale(
                face_video[:length].movedim(-1, 1), 512, 512, "area", "center"
            ) * 2.0 - 1.0
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        # ------------------------------------------------------------------
        # 6. Background video & continue motion overlay
        # ------------------------------------------------------------------
        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(
                    background_video[:length].movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]

        # ------------------------------------------------------------------
        # 7. Character mask handling
        # ------------------------------------------------------------------
        mask_refmotion = torch.ones((1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                                    device=concat_latent_image.device, dtype=concat_latent_image.dtype)

        if continue_motion is not None:
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0

        if character_mask is not None:
            if character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat(length, 1, 1)
                else:
                    character_mask = character_mask[video_frame_offset:]

                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(0)
                character_mask = comfy.utils.common_upscale(
                    character_mask.unsqueeze(1)[:, :, :length], width//8, height//8, "nearest-exact", "center"
                )
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]

        # ------------------------------------------------------------------
        # 8. Final concatenation of the two encoded parts (reference + motion)
        # ------------------------------------------------------------------
        # TILED ENCODE the final image tensor (which now contains continue_motion / background)
        motion_latent = vae.encode_tiled(
            image[:, :, :, :3],
            tile_x=tile_size, tile_y=tile_size, overlap=overlap,
            tile_t=temporal_size, overlap_t=temporal_overlap
        )

        concat_latent_image = torch.cat((concat_latent_image, motion_latent), dim=2)

        # ------------------------------------------------------------------
        # 9. Final mask preparation (exactly like original + reference node)
        # ------------------------------------------------------------------
        mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4,
                                             mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # ------------------------------------------------------------------
        # 10. Empty latent output
        # ------------------------------------------------------------------
        latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
                             device=comfy.model_management.intermediate_device())

        return (positive, negative,
                {"samples": latent},
                trim_latent,
                max(0, ref_motion_latent_length * 4 - 3),
                video_frame_offset + length)
