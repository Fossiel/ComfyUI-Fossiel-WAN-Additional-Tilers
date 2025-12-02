from .wan22_animate import FossielWanAnimateToVideoTiled
from .wan22_fmlf2v_std import FossielWanFirstMiddleLastFrameToVideoTiled
from .wan22_fun_ctrl import FossielWan22FunControlToVideoTiled
from .wan22_painteri2v import FossielPainterI2VTiled
from .wan22_painterflf2v import FossielPainterFLF2VTiled
from .wan22_painterlongvid import FossielPainterLongVideoTiled
from .wan22_s2v import FossielWanSoundImageToVideo, FossielWanSoundImageToVideoExtend

NODE_CLASS_MAPPINGS = {
    "Wan22AnimateToVideoTiled": FossielWanAnimateToVideoTiled,
    "Wan22FirstMiddleLastFrameToVideoTiled": FossielWanFirstMiddleLastFrameToVideoTiled,
    "Wan22FunControlToVideoTiled": FossielWan22FunControlToVideoTiled,
    "Wan22PainterI2VTiled": FossielPainterI2VTiled,
    "Wan22PainterFLF2VTiled": FossielPainterFLF2VTiled,
    "Wan22PainterLongVideoTiled": FossielPainterLongVideoTiled,
    "Wan22SoundImageToVideoTiled": FossielWanSoundImageToVideo,
    "Wan22SoundImageToVideoExtendTiled": FossielWanSoundImageToVideoExtend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22AnimateToVideoTiled": "Wan22 Animate To Video (Tiled VAE Encode)",
    "Wan22FirstMiddleLastFrameToVideoTiled": "Wan22 First Middle Last Frame (Tiled VAE Encode)",
    "Wan22FunControlToVideoTiled": "Wan22 Fun Control To Video (Tiled VAE Encode)",
    "Wan22PainterI2VTiled": "Wan22 Painter I2V (Tiled VAE Encode)",
    "Wan22PainterFLF2VTiled": "Wan22 Painter FLF2V (Tiled VAE Encode)",
    "Wan22PainterLongVideoTiled": "Wan22 Painter Long Video (Tiled VAE Encode)",
    "Wan22SoundImageToVideoTiled": "Wan22 Sound Image To Video (Tiled VAE Encode)",
    "Wan22SoundImageToVideoExtendTiled": "Wan22 Sound Image To Video Extend (Tiled VAE Encode)"
}
