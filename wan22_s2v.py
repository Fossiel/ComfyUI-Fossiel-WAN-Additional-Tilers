import math
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import numpy as np
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")
    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps
    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)
    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def get_audio_embed_bucket_fps(audio_embed, fps=16, batch_frames=81, m=0, video_rate=30):
    num_layers, audio_frame_num, audio_dim = audio_embed.shape
    return_all_layers = num_layers > 1
    scale = video_rate / fps
    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    batch_idx = get_sample_indices(
        original_fps=video_rate,
        total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0)
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride))
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]
            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            frame_audio_embed = torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
        batch_audio_eb.append(frame_audio_embed)
    batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)
    return batch_audio_eb, min_batch_num


def wan_sound_to_video(positive, negative, vae, width, height, length, batch_size, frame_offset=0,
                       ref_image=None, audio_encoder_output=None, control_video=None,
                       ref_motion=None, ref_motion_latent=None,
                       tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
    latent_t = ((length - 1) // 4) + 1

    if audio_encoder_output is not None:
        feat = torch.cat(audio_encoder_output["encoded_audio_all_layers"])
        video_rate = 30
        fps = 16
        feat = linear_interpolation(feat, input_fps=50, output_fps=video_rate)
        batch_frames = latent_t * 4
        audio_embed_bucket, num_repeat = get_audio_embed_bucket_fps(feat, fps=fps, batch_frames=batch_frames, m=0, video_rate=video_rate)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        audio_embed_bucket = audio_embed_bucket[:, :, :, frame_offset:frame_offset + batch_frames]
        if audio_embed_bucket.shape[3] > 0:
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_embed_bucket})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_embed_bucket * 0.0})

    if ref_image is not None:
        ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        ref_latent = vae.encode_tiled(ref_image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap)
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    if ref_motion is not None:
        if ref_motion.shape[0] > 73:
            ref_motion = ref_motion[-73:]
        ref_motion = comfy.utils.common_upscale(ref_motion.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if ref_motion.shape[0] < 73:
            r = torch.ones([73, height, width, 3], device=ref_motion.device) * 0.5
            r[-ref_motion.shape[0]:] = ref_motion
            ref_motion = r
        ref_motion_latent = vae.encode_tiled(ref_motion[:, :, :, :3],
                                             tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                                             tile_t=temporal_size, overlap_t=temporal_overlap)

    if ref_motion_latent is not None:
        ref_motion_latent = ref_motion_latent[:, :, -19:]
        positive = node_helpers.conditioning_set_values(positive, {"reference_motion": ref_motion_latent})
        negative = node_helpers.conditioning_set_values(negative, {"reference_motion": ref_motion_latent})

    latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=comfy.model_management.intermediate_device())

    control_video_out = comfy.latent_formats.Wan21().process_out(torch.zeros_like(latent))
    if control_video is not None:
        control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        control_video_encoded = vae.encode_tiled(control_video[:, :, :, :3],
                                                 tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                                                 tile_t=temporal_size, overlap_t=temporal_overlap)
        control_video_out[:, :, :control_video_encoded.shape[2]] = control_video_encoded

    positive = node_helpers.conditioning_set_values(positive, {"control_video": control_video_out})
    negative = node_helpers.conditioning_set_values(negative, {"control_video": control_video_out})

    out_latent = {"samples": latent}
    return positive, negative, out_latent, frame_offset


class FossielWanSoundImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", ),
                "ref_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
                "ref_motion": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                audio_encoder_output=None, ref_image=None, control_video=None, ref_motion=None):
        positive, negative, out_latent, _ = wan_sound_to_video(
            positive, negative, vae, width, height, length, batch_size,
            ref_image=ref_image, audio_encoder_output=audio_encoder_output,
            control_video=control_video, ref_motion=ref_motion,
            tile_size=tile_size, overlap=overlap,
            temporal_size=temporal_size, temporal_overlap=temporal_overlap
        )
        return (positive, negative, out_latent)


class FossielWanSoundImageToVideoExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "length": ("INT", {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "video_latent": ("LATENT", ),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."}),
            },
            "optional": {
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", ),
                "ref_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, length, video_latent,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                audio_encoder_output=None, ref_image=None, control_video=None):
        video_latent = video_latent["samples"]
        width = video_latent.shape[-1] * 8
        height = video_latent.shape[-2] * 8
        batch_size = video_latent.shape[0]
        frame_offset = video_latent.shape[2] * 4

        positive, negative, out_latent, _ = wan_sound_to_video(
            positive, negative, vae, width, height, length, batch_size,
            frame_offset=frame_offset,
            ref_image=ref_image,
            audio_encoder_output=audio_encoder_output,
            control_video=control_video,
            ref_motion=None,
            ref_motion_latent=video_latent,
            tile_size=tile_size, overlap=overlap,
            temporal_size=temporal_size, temporal_overlap=temporal_overlap
        )
        return (positive, negative, out_latent)


# === Extension system (kept for compatibility) ===
class FossielWanS2VExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [FossielWanSoundImageToVideo, FossielWanSoundImageToVideoExtend]

async def comfy_entrypoint():
    return FossielWanS2VExtension()
