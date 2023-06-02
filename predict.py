import datetime
import random

MARKER = "anotherjesse-sd-timings-%s" % random.randint(0, 1000000)

print(MARKER, "start:", datetime.datetime.now())

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

MODEL_CACHE = "diffusers-cache"
BASE_MODEL_PATH = "./weights"

print(MARKER, "finished imports:", datetime.datetime.now())


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print(MARKER, "Loading pipeline...", datetime.datetime.now())

        if not os.path.exists(BASE_MODEL_PATH):
            self.real = False
            return

        print(MARKER, "Loading txt2img...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        self.safety_checker = self.txt2img_pipe.safety_checker
        self.real = True

        print(MARKER, "loaded  pipeline...", datetime.datetime.now())

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

        print(MARKER, "Using txt2img pipeline")
        pipe = self.txt2img_pipe
        extra_kwargs = {
            "width": width,
            "height": height,
        }

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(MARKER, f"Using seed: {seed}")

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
