import time
import datetime
import random

MARKER = "anotherjesse-sd-timings-%s" % random.randint(0, 1000000)

last = start = time.time()


def crappy_log(*args):
    global last
    t = time.time()
    print(MARKER, *args, "%0.2f" % (t - start), "%0.2f" % (t - last))
    last = t


crappy_log("start")

from typing import Iterator

from cog import BasePredictor, Input, Path
import os
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)

crappy_log("finished imports")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        crappy_log("starting setup")

        if not os.path.exists("./weights"):
            self.real = False
            return

        if True:
            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                safety_checker=None,
            )
            crappy_log("finished importing pipeline")
        else:
            from diffusers.models import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPFeatureExtractor

            vae = AutoencoderKL.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="vae",
            )
            crappy_log("loaded vae")

            unet = UNet2DConditionModel.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="unet",
            )
            crappy_log("loaded unet")

            text_encoder = CLIPTextModel.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="text_encoder",
            )
            crappy_log("loaded text_encoder")

            tokenizer = CLIPTokenizer.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="tokenizer",
            )
            crappy_log("loaded tokenizer")

            scheduler = DDIMScheduler.from_pretrained(
                "./weights/scheduler/scheduler_config.json"
            )
            crappy_log("loaded scheduler")

            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="safety_checker",
            )
            crappy_log("loaded safety_checker")

            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "./weights",
                torch_dtype=torch.float16,
                local_files_only=True,
                subfolder="feature_extractor",
            )
            crappy_log("loaded feature_extractor")

            self.txt2img_pipe = StableDiffusionPipeline(
                vae=vae,
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            crappy_log("manually created pipeline")

        self.txt2img_pipe.to("cuda")
        crappy_log("moved to cuda")

        self.safety_checker = self.txt2img_pipe.safety_checker
        self.real = True

        crappy_log("finished setup")

    def make_scheduler(self, name, config):
        return {
            "DDIM": DDIMScheduler.from_config(config),
            "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
            "HeunDiscrete": HeunDiscreteScheduler.from_config(config),
            "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
            "K_EULER": EulerDiscreteScheduler.from_config(config),
            "KLMS": LMSDiscreteScheduler.from_config(config),
            "PNDM": PNDMScheduler.from_config(config),
            "UniPCMultistep": UniPCMultistepScheduler.from_config(config),
        }[name]

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=10,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
            ],
            description="Choose a scheduler.",
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""

        if not self.real:
            raise RuntimeError("This is a template, not a real model - add weights")

        crappy_log("Using txt2img pipeline")
        pipe = self.txt2img_pipe
        extra_kwargs = {
            "width": width,
            "height": height,
        }

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        crappy_log("Using seed: ", seed)

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = self.make_scheduler(scheduler, pipe.scheduler.config)

        if disable_safety_check:
            pipe.safety_checker = None
        else:
            pipe.safety_checker = self.safety_checker

        result_count = 0
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue

            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            yield Path(output_path)
            result_count += 1

        if result_count == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
