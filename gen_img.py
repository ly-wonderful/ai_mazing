# 1. Create Config

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size : int = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 25
    save_model_epochs = 30
    
    input_channels = 3  # the number of input channels, 3 for RGB images
    output_channels = 3  # the number of output channels, 3 for RGB images
    
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = 'ddpm-butterflies-128'  # the model namy locally and on the HF Hub
    output_dir = './output'
    seed = 0

config = TrainingConfig()


# 2. Load Data
from datasets import load_dataset

dataset = load_dataset("imagefolder", 
                       data_dir="C:/Users/lywon/projects/AI_Mazing_img/input/nn", 
                       split="train")


# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")

# dataset[0]['image'].show()

# show a sample of the dataset
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]['image']):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# fig.show()


# 3. Preprocess Data
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images} # return a dictionary with a key "images"


dataset.set_transform(transform) # dataset is now a tensor on GPU now, instead of a PIL image

# show pre-processed data
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["images"]):
#     axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
#     axs[i].set_axis_off()
# fig.show()

# create a dataloader
import torch
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# 4. Create the Diffusion Model using UNet2DModel from diffusers
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=config.input_channels,  # the number of input channels, 3 for RGB images
    out_channels=config.output_channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
      ),
).to("cuda")

# try the model using one image
# sample_image = dataset[0]['images'].unsqueeze(0)
# # pass the image through the model and see if the output has correct shape
# assert sample_image.shape == model(sample_image, timestep=0).sample.shape, "The model output shape is incorrect"

# 5. Define a noise scheduler
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

from PIL import Image

# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# model(sample).sample.shape needs to be (batch, channels, height, width)
# need to change the sequence of the dimensions to (batch, height, width, channels) for PIL to display the image
# Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

# 6. Define the loss function
import torch.nn.functional as F

# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)

# 7. Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# a cosine learning rate scheduler with warmup
from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
# 8. Define functions to train the model
from diffusers import DDPMPipeline

def make_grid(images, rows, cols):
    """_summary_

    Args:
        images (List[PIL.Image]): a list of images to make a grid from
        rows (int): number of rows
        cols (in): number of columns

    Returns:
        _type_: _description_
    """
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
                    batch_size = config.eval_batch_size,
                    generator=torch.manual_seed(config.seed),
                    num_inference_steps=500
                        ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
from tqdm.auto import tqdm
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].cuda()
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler).to("cuda")

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                    

train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
                    

model.config.sample_size


import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
