"""VAE based generative object models."""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from common_fate_object_learning import datasets


class PartialTanh(nn.Module):
    def forward(self, inputs):
        inputs = inputs.clone()
        inputs[:, :3] = torch.tanh(inputs[:, :3])
        return inputs


class BetaVAE(torch.nn.Module):
    """Object model based on a beta-VAE."""

    _in_channels: int = 3
    _out_channels: int = 4

    def __init__(self,
                 sample_size: Tuple[int],
                 latent_dims: int,
                 hidden_dims: Iterable[int] = (32, 64, 128, 256, 512),
                 hidden_strides: Optional[Iterable[int]] = None,
                 normalization: str = "groupnorm",
                 beta: float = 0.0001,
                 image_loss_type: str = "l1",
                 mask_loss_weight: float = 0.01,
                 consistency_dims: int = 0,
                 consistency_loss_weight: float = 0.01) -> None:
        """Initializes the beta-VAE model.

        Args:
            sample_size: Spatial sample size as tuple `(height, width)`.
            latent_dims: Number of latent dimensions.
            hidden_dims: Number of channels for the hidden layers used by the
                encoder. The decoder is constructed using the same channel
                counts but in reverse order (with one additional conv layer).
            hidden_strides: Strides used by the hidden layers in the encoder and
                the decoder. If set to `None`, a stride of 2 is used for every
                layer.
            normalization: Normalization method used by hidden layers in the
                encoder and decoder: `"batchnorm" | "groupnorm"`.
            beta: Loss weight used for the prior loss (KLD).
            image_loss_type: Loss used to evaluate the reconstruction of the RGB
                image: `"l1" | "mse"`.
            mask_loss_weight: Loss weight for used for the mask reconstruction
                loss.
            consistency_dims: Number of latent dimensions that are penalized for
                not being constant over frames.
            consistency_loss_weight: Loss weight used for the temporal
                consistency regularization.
        """
        super(BetaVAE, self).__init__()

        if hidden_strides is None:
            hidden_strides = [2] * len(hidden_dims)

        if not len(hidden_dims) == len(hidden_strides):
            raise ValueError(
                f"The number of hidden strides ({len(hidden_strides)}) does not"
                f"match the number of hidden layers ({len(hidden_dims)})."
            )

        self._sample_size = sample_size
        self._latent_dim = latent_dims
        self._hidden_dims = hidden_dims
        self._hidden_strides = hidden_strides
        self._normalization = normalization.lower()
        self.beta = beta
        self.recons_loss_base_type = image_loss_type
        self.mask_loss_weight = mask_loss_weight
        self.consistent_hidden_dims = consistency_dims
        self.consistency_loss_weight = consistency_loss_weight

        # Calculate the size of the images after being passed through the
        # decoder. This is required to correctly map this representation to and
        # from the non-spatial latents.
        self._encoded_sample_size = sample_size
        for stride in self._hidden_strides:
            self._encoded_sample_size = (
                math.floor((self._encoded_sample_size[0] - 1) / stride + 1),
                math.floor((self._encoded_sample_size[1] - 1) / stride + 1)
            )

        self._build_encoder()
        self._build_decoder()


    def _get_norm_layer(self, hidden_dim: int) -> nn.Module:
        if self._normalization == 'batchnorm':
            return nn.BatchNorm2d(hidden_dim)

        if self._normalization == 'groupnorm':
            return nn.GroupNorm(1, hidden_dim)
            
        raise ValueError(f"Unknown normalization method: {self._normalization}")


    def _build_encoder(self) -> None:
        """Builds the encoder network.

        This method uses the attributes set in the constructor and makes the
        encoder network available as `self.encoder_base`, `self.encoder_mean`
        and `self.encoder_log_var`.
        """
        in_channels = self._in_channels
        layers = []

        for hidden_dim, hidden_stride in zip(self._hidden_dims, self._hidden_strides):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=hidden_stride,
                              padding=1
                    ),
                    self._get_norm_layer(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        self.encoder_base = nn.Sequential(*layers)

        num_encoded_locations = self._encoded_sample_size[0] * self._encoded_sample_size[1]
        flattened_dims = self._hidden_dims[-1] * num_encoded_locations
        self.encoder_mean = nn.Linear(flattened_dims, self._latent_dim)
        self.encoder_log_var = nn.Linear(flattened_dims, self._latent_dim)


    def _build_decoder(self) -> None:
        """Builds the decoder network.

        This method uses the attributes set in the constructor and makes the
        decoder network available as `self.decoder_input` and `self.decoder`.
        """
        num_encoded_locations = self._encoded_sample_size[0] * self._encoded_sample_size[1]
        flattened_dims = self._hidden_dims[-1] * num_encoded_locations
        self.decoder_input = nn.Linear(self._latent_dim, flattened_dims)

        layers = []

        hidden_dims = list(self._hidden_dims)
        hidden_dims.reverse()

        hidden_strides = list(self._hidden_strides)
        hidden_strides.reverse()

        for in_channels, out_channels, stride in zip(hidden_dims[:-1], hidden_dims[1:], hidden_strides):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1
                    ),
                    self._get_norm_layer(out_channels),
                    nn.LeakyReLU())
            )

        layers.append(
            nn.Sequential(
                # REVIEW Why is this additional layer used?
                nn.ConvTranspose2d(self._hidden_dims[0],
                                self._hidden_dims[0],
                                kernel_size=3,
                                stride=self._hidden_strides[0],
                                padding=1,
                                output_padding=1
                ),
                self._get_norm_layer(self._hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Conv2d(self._hidden_dims[0], 
                        self._out_channels,
                        kernel_size=3,
                        padding=1
                ),
                PartialTanh()
            )
        )

        self.decoder = nn.Sequential(*layers)


    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Returns the parameters of the latent distributions for the given samples.

        Args:
            input: Input images as `torch.Tensor`of shape `(N, 3, H, W)`

        Returns:
            Tuple `(mean, log_var)` of `torch.Tensor`s with shapes
            `(N, latent_dim)`. The two tensors represent means and log variances
            of factorial gaussian distributions over the latents for each
            sample.
        """
        result = self.encoder_base(input)
        result = torch.flatten(result, start_dim=1)

        mean = self.encoder_mean(result)
        log_var = self.encoder_log_var(result)

        return mean, log_var


    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns the samples decoded from the given latent vectors.

        Args:
            latent: Latent vectors as `torch.Tensor` of shape `(N, latent_dim)`.
        
        Returns:
            Decoded samples as `torch.Tensor` with shape `(N, C, H, W)`.
        """
        result = self.decoder_input(latent)
        result = result.view(-1, self._hidden_dims[-1], *self._encoded_sample_size)

        return self.decoder(result)


    def sample_latent(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Samples latents using the reparameterization trick.

        Args:
            mean: Mean of the latent distribution as `torch.Tensor` with shape
                `(N, latent_dim)`.
            log_var: Log variance of the latent distribution as `torch.Tensor`
                with shape `(N, latent_dim)`.

        Returns:
            Latent code as `torch.Tensor` with shape `(N, latent_dim)`.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction and latent distribution for the given images.
        
        Args:
            images: input images as `torch.Tensor` with shape `(*, 3, H, W)`.
        
        Returns:
            Tuple `(reconstruction, latent_mean, latent_log_var)`. The
            reconstruction is a `torch.Tensor` of shape `(*, 4, H, W)`. The
            latent mean and log variance are `torch.Tensor`s with shape
            `(*, latent_dims)`.
        """ 
        batch_shape = images.shape[:-3]
        images = images.view(-1, *images.shape[-3:])

        mean, log_var = self.encode(images)
        latent = self.sample_latent(mean, log_var)
        reconstruction = self.decode(latent)

        reconstruction = reconstruction.view(*batch_shape, *reconstruction.shape[-3:])

        return reconstruction, mean, log_var

    def _image_loss(self,
                    reconstructed_image: torch.Tensor,
                    target_image: torch.Tensor,
                    target_mask: torch.Tensor):        
        mask = (target_mask == datasets.FOREGROUND)
        
        batch_dims = mask.ndim - 2
        mask = mask.unsqueeze(-3).repeat(*([1] * batch_dims), 3, 1, 1)

        if self.recons_loss_base_type == "l1":
            return F.l1_loss(reconstructed_image[mask], target_image[mask], reduction="mean")
        elif self.recons_loss_base_type == "mse":
            return F.mse_loss(reconstructed_image[mask], target_image[mask], reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {self.recons_loss_base_type}")

    def _mask_loss(self,
                   reconstructed_mask: torch.Tensor,
                   target_mask: torch.Tensor):
        mask = (target_mask != datasets.OCCLUSION)
        return F.binary_cross_entropy_with_logits(
            reconstructed_mask[mask],
            target_mask[mask].float() / 2.0,
            reduction="mean"
        )
    
    def _prior_loss(self, latent_mean: torch.Tensor, latent_log_var: torch.Tensor):
        return torch.mean(
            -0.5 * torch.sum(1 + latent_log_var - latent_mean ** 2 - latent_log_var.exp(), dim=-1)
        )

    def _consistency_loss(self,
                          latent_mean: torch.Tensor,
                          latent_log_var: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.consistent_hidden_dims <= 0:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # REVIEW Use KLD between the two latent distributions instead?
        consistency_loss_mean = F.mse_loss(
            latent_mean[0, :self.consistent_hidden_dims],
            latent_mean[1, :self.consistent_hidden_dims]
        )
        consistency_loss_log_var = F.mse_loss(
            latent_log_var[0, :self.consistent_hidden_dims],
            latent_log_var[1, :self.consistent_hidden_dims]
        )
        consistency_loss = consistency_loss_mean + consistency_loss_log_var

        return consistency_loss, consistency_loss_mean, consistency_loss_log_var

    def loss(self,
             reconstruction: torch.Tensor,
             latent_mean: torch.Tensor,
             latent_log_var: torch.Tensor,
             target_image: torch.Tensor,
             target_mask: torch.Tensor) -> Dict:
        """Returns the overall and partial losses for the given samples."""
        reconstructed_image = reconstruction[..., :3, :, :]
        reconstructed_mask = reconstruction[..., 3, :, :]

        image_loss = self._image_loss(reconstructed_image, target_image, target_mask)

        mask_loss = self._mask_loss(reconstructed_mask, target_mask)

        prior_loss = self._prior_loss(latent_mean, latent_log_var)

        consistency_loss, consistency_loss_mean, consistency_loss_log_var = \
            self._consistency_loss(latent_mean, latent_log_var)

        overall_loss = image_loss + \
                       self.mask_loss_weight * mask_loss + \
                       self.beta * prior_loss + \
                       self.consistency_loss_weight * consistency_loss

        partial_losses = {
            "image_loss": image_loss,
            "mask_loss": mask_loss,
            "prior_loss": prior_loss,
            "consistency_loss": consistency_loss,
            "consistency_loss_mean": consistency_loss_mean,
            "consistency_loss_log_var": consistency_loss_log_var
        }

        return overall_loss, partial_losses


    def sample(self, num_samples: int, device: Any = "cuda") -> torch.Tensor:
        """Returns samples from the model.

        Args:
            num_samples: Number of samples to be generated.
            device: Device to run the model.

        Returns:
            Samples as `torch.Tensor` with shape `(N, 4, H, W)`.
        """
        latent = torch.randn(num_samples, self._latent_dim)
        latent = latent.to(device)
        return self.decode(latent)


    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction of the given input images.

        Args:
            input: Input images as `torch.Tensor` of shape `(*, 3, H, W)`.
        
        Returns:
            Reconstructed images as `torch.Tensor` of shape `(*, 4, H, W)`.
        """
        return self.forward(images)[0]


class DoubleDecoderVAE(torch.nn.Module):
    """Object model based on a beta-VAE."""

    _in_channels: int = 3
    _image_out_channels: int = 3
    _mask_out_channels: int = 1

    def __init__(self,
                 sample_size: Tuple[int],
                 latent_dims: int,
                 image_hidden_dims: Iterable[int] = (32, 64, 128, 256, 512),
                 image_hidden_strides: Optional[Iterable[int]] = None,
                 mask_hidden_dims: Iterable[int] = (32, 64, 128, 256, 512),
                 mask_hidden_strides: Optional[Iterable[int]] = None,
                 normalization: str = "groupnorm",
                 beta: float = 0.0001,
                 image_loss_type: str = "l1",
                 mask_loss_weight: float = 0.01,
                 consistency_dims: int = 0,
                 consistency_loss_weight: float = 0.01,
                 normalize_samples_in_loss: bool = False) -> None:
        """Initializes the beta-VAE model.

        Args:
            sample_size: Spatial sample size as tuple `(height, width)`.
            latent_dims: Number of latent dimensions.
            hidden_dims: Number of channels for the hidden layers used by the
                encoder. The decoder is constructed using the same channel
                counts but in reverse order (with one additional conv layer).
            hidden_strides: Strides used by the hidden layers in the encoder and
                the decoder. If set to `None`, a stride of 2 is used for every
                layer.
            normalization: Normalization method used by hidden layers in the
                encoder and decoder: `"batchnorm" | "groupnorm"`.
            beta: Loss weight used for the prior loss (KLD).
            image_loss_type: Loss used to evaluate the reconstruction of the RGB
                image: `"l1" | "mse"`.
            mask_loss_weight: Loss weight for used for the mask reconstruction
                loss.
            consistency_dims: Number of latent dimensions that are penalized for
                not being constant over frames.
            consistency_loss_weight: Loss weight used for the temporal
                consistency regularization.
        """
        super().__init__()

        if image_hidden_strides is None:
            image_hidden_strides = [2] * len(image_hidden_dims)

        if mask_hidden_strides is None:
            mask_hidden_strides = [2] * len(mask_hidden_dims)

        if not len(image_hidden_dims) == len(image_hidden_strides):
            raise ValueError(
                f"The number of hidden strides ({len(image_hidden_strides)}) does not"
                f"match the number of hidden layers ({len(image_hidden_dims)})."
            )

        if not len(mask_hidden_dims) == len(mask_hidden_strides):
            raise ValueError(
                f"The number of hidden strides ({len(mask_hidden_strides)}) does not"
                f"match the number of hidden layers ({len(mask_hidden_dims)})."
            )

        self._sample_size = sample_size
        self._latent_dim = latent_dims
        self._image_hidden_dims = image_hidden_dims
        self._image_hidden_strides = image_hidden_strides
        self._mask_hidden_dims = mask_hidden_dims
        self._mask_hidden_strides = mask_hidden_strides
        self._normalization = normalization.lower()
        self.beta = beta
        self.recons_loss_base_type = image_loss_type
        self.mask_loss_weight = mask_loss_weight
        self.consistent_hidden_dims = consistency_dims
        self.consistency_loss_weight = consistency_loss_weight
        self.normalize_samples_in_loss = normalize_samples_in_loss

        # Calculate the size of the images after being passed through the
        # decoder. This is required to correctly map this representation to and
        # from the non-spatial latents.
        self._encoded_sample_size = sample_size
        for stride in self._image_hidden_strides:
            self._encoded_sample_size = (
                math.floor((self._encoded_sample_size[0] - 1) / stride + 1),
                math.floor((self._encoded_sample_size[1] - 1) / stride + 1)
            )

        self._build_encoder()
        self._build_image_decoder()
        self._build_mask_decoder()


    def _get_norm_layer(self, hidden_dim: int) -> nn.Module:
        if self._normalization == 'batchnorm':
            return nn.BatchNorm2d(hidden_dim)

        if self._normalization == 'groupnorm':
            return nn.GroupNorm(1, hidden_dim)
            
        raise ValueError(f"Unknown normalization method: {self._normalization}")


    def _build_encoder(self) -> None:
        """Builds the encoder network.

        This method uses the attributes set in the constructor and makes the
        encoder network available as `self.encoder_base`, `self.encoder_mean`
        and `self.encoder_log_var`.
        """
        in_channels = self._in_channels
        layers = []

        for hidden_dim, hidden_stride in zip(self._image_hidden_dims, self._image_hidden_strides):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=hidden_stride,
                              padding=1
                    ),
                    self._get_norm_layer(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        self.encoder_base = nn.Sequential(*layers)

        num_encoded_locations = self._encoded_sample_size[0] * self._encoded_sample_size[1]
        flattened_dims = self._image_hidden_dims[-1] * num_encoded_locations
        self.encoder_mean = nn.Linear(flattened_dims, self._latent_dim)
        self.encoder_log_var = nn.Linear(flattened_dims, self._latent_dim)


    def _build_image_decoder(self) -> None:
        """Builds the decoder network.

        This method uses the attributes set in the constructor and makes the
        decoder network available as `self.decoder_input` and `self.decoder`.
        """
        num_encoded_locations = self._encoded_sample_size[0] * self._encoded_sample_size[1]
        flattened_dims = self._image_hidden_dims[-1] * num_encoded_locations
        self.image_decoder_input = nn.Linear(self._latent_dim, flattened_dims)

        layers = []

        hidden_dims = list(self._image_hidden_dims)
        hidden_dims.reverse()

        hidden_strides = list(self._image_hidden_strides)
        hidden_strides.reverse()

        for in_channels, out_channels, stride in zip(hidden_dims[:-1], hidden_dims[1:], hidden_strides):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1
                    ),
                    self._get_norm_layer(out_channels),
                    nn.LeakyReLU())
            )

        layers.append(
            nn.Sequential(
                # REVIEW Why is this additional layer used?
                nn.ConvTranspose2d(self._image_hidden_dims[0],
                                self._image_hidden_dims[0],
                                kernel_size=3,
                                stride=self._image_hidden_strides[0],
                                padding=1,
                                output_padding=1
                ),
                self._get_norm_layer(self._image_hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Conv2d(self._image_hidden_dims[0], 
                        self._image_out_channels,
                        kernel_size=3,
                        padding=1
                ),
                PartialTanh()
            )
        )

        self.image_decoder = nn.Sequential(*layers)


    def _build_mask_decoder(self) -> None:
        """Builds the decoder network.

        This method uses the attributes set in the constructor and makes the
        decoder network available as `self.decoder_input` and `self.decoder`.
        """
        num_encoded_locations = self._encoded_sample_size[0] * self._encoded_sample_size[1]
        flattened_dims = self._mask_hidden_dims[-1] * num_encoded_locations
        self.mask_decoder_input = nn.Linear(self._latent_dim, flattened_dims)

        layers = []

        hidden_dims = list(self._mask_hidden_dims)
        hidden_dims.reverse()

        hidden_strides = list(self._mask_hidden_strides)
        hidden_strides.reverse()

        for in_channels, out_channels, stride in zip(hidden_dims[:-1], hidden_dims[1:], hidden_strides):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1
                    ),
                    self._get_norm_layer(out_channels),
                    nn.LeakyReLU())
            )

        layers.append(
            nn.Sequential(
                # REVIEW Why is this additional layer used?
                nn.ConvTranspose2d(self._mask_hidden_dims[0],
                                self._mask_hidden_dims[0],
                                kernel_size=3,
                                stride=self._mask_hidden_strides[0],
                                padding=1,
                                output_padding=1
                ),
                self._get_norm_layer(self._mask_hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Conv2d(self._mask_hidden_dims[0], 
                        self._mask_out_channels,
                        kernel_size=3,
                        padding=1
                )
            )
        )

        self.mask_decoder = nn.Sequential(*layers)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Returns the parameters of the latent distributions for the given samples.

        Args:
            input: Input images as `torch.Tensor`of shape `(N, 3, H, W)`

        Returns:
            Tuple `(mean, log_var)` of `torch.Tensor`s with shapes
            `(N, latent_dim)`. The two tensors represent means and log variances
            of factorial gaussian distributions over the latents for each
            sample.
        """
        result = self.encoder_base(input)
        result = torch.flatten(result, start_dim=1)

        mean = self.encoder_mean(result)
        log_var = self.encoder_log_var(result)

        return mean, log_var


    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns the samples decoded from the given latent vectors.

        Args:
            latent: Latent vectors as `torch.Tensor` of shape `(N, latent_dim)`.
        
        Returns:
            Decoded samples as `torch.Tensor` with shape `(N, C, H, W)`.
        """
        image = self.image_decoder_input(latent)
        image = image.view(-1, self._image_hidden_dims[-1], *self._encoded_sample_size)
        image = self.image_decoder(image)

        mask = self.mask_decoder_input(latent)
        mask = mask.view(-1, self._mask_hidden_dims[-1], *self._encoded_sample_size)
        mask = self.mask_decoder(mask)

        return torch.cat([image, mask], dim=-3)


    def sample_latent(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Samples latents using the reparameterization trick.

        Args:
            mean: Mean of the latent distribution as `torch.Tensor` with shape
                `(N, latent_dim)`.
            log_var: Log variance of the latent distribution as `torch.Tensor`
                with shape `(N, latent_dim)`.

        Returns:
            Latent code as `torch.Tensor` with shape `(N, latent_dim)`.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction and latent distribution for the given images.
        
        Args:
            images: input images as `torch.Tensor` with shape `(*, 3, H, W)`.
        
        Returns:
            Tuple `(reconstruction, latent_mean, latent_log_var)`. The
            reconstruction is a `torch.Tensor` of shape `(*, 4, H, W)`. The
            latent mean and log variance are `torch.Tensor`s with shape
            `(*, latent_dims)`.
        """ 
        batch_shape = images.shape[:-3]
        images = images.view(-1, *images.shape[-3:])

        mean, log_var = self.encode(images)
        latent = self.sample_latent(mean, log_var)
        reconstruction = self.decode(latent)

        reconstruction = reconstruction.view(*batch_shape, *reconstruction.shape[-3:])

        return reconstruction, mean, log_var

    def _image_loss(self,
                    reconstructed_image: torch.Tensor,
                    target_image: torch.Tensor,
                    target_mask: torch.Tensor):        
        mask = (target_mask == datasets.FOREGROUND)
        
        batch_dims = mask.ndim - 2
        mask = mask.unsqueeze(-3).repeat(*([1] * batch_dims), 3, 1, 1)

        if self.recons_loss_base_type == "l1":
            if not self.normalize_samples_in_loss:
                return F.l1_loss(reconstructed_image[mask], target_image[mask], reduction="mean")
            
            num_samples = target_image.shape[0] * target_image.shape[1]
            weight = mask.type(torch.float32) / torch.sum(mask, dim=(-3, -2, -1), keepdim=True)
            loss = weight[mask] * F.l1_loss(reconstructed_image[mask], target_image[mask], reduction="none")
            return loss.sum() / num_samples

        elif self.recons_loss_base_type == "mse":
            if not self.normalize_samples_in_loss:
                return F.mse_loss(reconstructed_image[mask], target_image[mask], reduction="mean")

            num_samples = target_image.shape[0] * target_image.shape[1]
            weight = mask.type(torch.float32) / torch.sum(mask, dim=(-3, -2, -1), keepdim=True)
            loss = weight[mask] * F.mse_loss(reconstructed_image[mask], target_image[mask], reduction="none")
            return loss.sum() / num_samples

        else:
            raise ValueError(f"Unknown loss type: {self.recons_loss_base_type}")

    def _mask_loss(self,
                   reconstructed_mask: torch.Tensor,
                   target_mask: torch.Tensor):
        mask = (target_mask != datasets.OCCLUSION)

        if not self.normalize_samples_in_loss:
            return F.binary_cross_entropy_with_logits(
                reconstructed_mask[mask],
                target_mask[mask].float() / 2.0,
                reduction="mean"
            )

        weight = mask.type(torch.float32) / torch.sum(mask, dim=(-3, -2, -1), keepdim=True)
        
        loss = weight[mask] * F.binary_cross_entropy_with_logits(
            reconstructed_mask[mask],
            target_mask[mask].float() / 2.0,
            reduction="none"
        )

        num_samples = target_mask.shape[0] * target_mask.shape[1]
        return loss.sum() / num_samples
    
    def _prior_loss(self, latent_mean: torch.Tensor, latent_log_var: torch.Tensor):
        return torch.mean(
            -0.5 * torch.sum(1 + latent_log_var - latent_mean ** 2 - latent_log_var.exp(), dim=-1)
        )

    def _consistency_loss(self,
                          latent_mean: torch.Tensor,
                          latent_log_var: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.consistent_hidden_dims <= 0:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # REVIEW Use KLD between the two latent distributions instead?
        consistency_loss_mean = F.mse_loss(
            latent_mean[0, :self.consistent_hidden_dims],
            latent_mean[1, :self.consistent_hidden_dims]
        )
        consistency_loss_log_var = F.mse_loss(
            latent_log_var[0, :self.consistent_hidden_dims],
            latent_log_var[1, :self.consistent_hidden_dims]
        )
        consistency_loss = consistency_loss_mean + consistency_loss_log_var

        return consistency_loss, consistency_loss_mean, consistency_loss_log_var

    def loss(self,
             reconstruction: torch.Tensor,
             latent_mean: torch.Tensor,
             latent_log_var: torch.Tensor,
             target_image: torch.Tensor,
             target_mask: torch.Tensor) -> Dict:
        """Returns the overall and partial losses for the given samples."""
        reconstructed_image = reconstruction[..., :3, :, :]
        reconstructed_mask = reconstruction[..., 3, :, :]

        image_loss = self._image_loss(reconstructed_image, target_image, target_mask)

        mask_loss = self._mask_loss(reconstructed_mask, target_mask)

        prior_loss = self._prior_loss(latent_mean, latent_log_var)

        consistency_loss, consistency_loss_mean, consistency_loss_log_var = \
            self._consistency_loss(latent_mean, latent_log_var)

        overall_loss = image_loss + \
                       self.mask_loss_weight * mask_loss + \
                       self.beta * prior_loss + \
                       self.consistency_loss_weight * consistency_loss

        partial_losses = {
            "image_loss": image_loss,
            "mask_loss": mask_loss,
            "prior_loss": prior_loss,
            "consistency_loss": consistency_loss,
            "consistency_loss_mean": consistency_loss_mean,
            "consistency_loss_log_var": consistency_loss_log_var
        }

        return overall_loss, partial_losses


    def sample(self, num_samples: int, device: Any = "cuda") -> torch.Tensor:
        """Returns samples from the model.

        Args:
            num_samples: Number of samples to be generated.
            device: Device to run the model.

        Returns:
            Samples as `torch.Tensor` with shape `(N, 4, H, W)`.
        """
        latent = torch.randn(num_samples, self._latent_dim)
        latent = latent.to(device)
        return self.decode(latent)


    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction of the given input images.

        Args:
            input: Input images as `torch.Tensor` of shape `(*, 3, H, W)`.
        
        Returns:
            Reconstructed images as `torch.Tensor` of shape `(*, 4, H, W)`.
        """
        return self.forward(images)[0]
