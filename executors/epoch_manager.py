import os.path
from typing import Iterator, Union, List

import Levenshtein as Levenshtein
import torch
import torchvision.utils
from torch import tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader


# from datasets import CocoLocalizationDataset


class EpochManagerGAN:
    def __init__(self,
                 generator,
                 generator_optimizer,
                 generator_criterion,
                 discriminator,
                 discriminator_optimizer,
                 discriminator_criterion,
                 cfg,
                 dataloaders_dict=None,
                 device=None,
                 fixed_noise=None):
        self.cfg = cfg

        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_criterion = generator_criterion
        self.generator_losses = dict()
        self.fake_accuracy = dict()

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_criterion = discriminator_criterion
        self.discriminator_losses = dict()
        self.real_accuracy = dict()

        self.fixed_noise = fixed_noise is not None
        self.device = self.cfg.device if device is None else device
        self.noise = fixed_noise if self.fixed_noise \
            else torch.FloatTensor(cfg.batch_size, cfg.z_depth, 1, 1).to(self.device)
        self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()
        self.valid_noise = None

        self._global_step = dict()

    def train(self, stage_key):
        self.discriminator.train()
        self.generator.train()
        for i, batch_info in enumerate(self._epoch_generator(stage_key), start=1):
            for param in self.discriminator.parameters():
                param.grad = None
            batch_info['d_loss'].backward()
            self.discriminator_optimizer.step()

            if i % self.cfg.n_critic == 0:
                if not self.fixed_noise:
                    self.noise.normal_()
                fake = self.generator(self.noise)
                fake_output, fake_logits = self.discriminator(fake)
                g_loss = self.generator_criterion(fake_logits)
                for param in self.generator.parameters():
                    param.grad = None
                g_loss.backward()
                self.generator_optimizer.step()

                self.generator_losses[stage_key].append(g_loss.item())
                batch_info['g_loss'] = g_loss.item()

    @torch.no_grad()
    def validation(self, path, epoch, num_images=4, transforms=None, nrow=8):
        self.discriminator.eval()
        self.generator.eval()

        if not self.fixed_noise:
            if self.valid_noise is None:
                self.valid_noise = torch.normal(0, 1, (num_images, self.cfg.z_depth, 1, 1))
            noise = self.valid_noise.to(self.device)
        else:
            noise = self.noise[:num_images]

        generated = self.generator(noise).cpu()
        if transforms is not None:
            generated = transforms(generated)

        if not os.path.exists(path):
            os.makedirs(path)
        torchvision.utils.save_image(generated, os.path.join(path, f'{str(epoch)}.jpg'), nrow=nrow)

    @torch.no_grad()
    def test(self, stage_key):
        self.discriminator.eval()
        self.generator.eval()
        for batch_info in self._epoch_generator(stage=stage_key):
            ...

    def save_model(self, epoch, path=None):
        path = self.cfg.SAVE_PATH if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{epoch}.pth')

        checkpoint = dict(epoch=self._global_step,
                          discriminator=self.discriminator.state_dict(),
                          discriminator_optimizer=self.discriminator_optimizer.state_dict(),
                          generator=self.generator.state_dict(),
                          generator_optimizer=self.generator_optimizer.state_dict(),
                          generator_losses=self.generator_losses,
                          discriminator_losses=self.discriminator_losses,
                          fake_accuracy=self.fake_accuracy,
                          real_accuracy=self.real_accuracy
        )

        torch.save(checkpoint, path)
        print('model saved, epoch:', epoch)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self._global_step = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        self.generator_losses = checkpoint['generator_losses']
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        self.discriminator_losses = checkpoint['discriminator_losses']
        self.fake_accuracy = checkpoint['fake_accuracy']
        self.real_accuracy = checkpoint['real_accuracy']
        print('model loaded')

    def _epoch_generator(self, stage) -> Iterator[tensor]:

        if stage not in self._global_step:
            self._get_global_step(stage)
        batch_info = dict(d_loss=0., g_loss=0.)
        for i, (real, targets) in enumerate(self.dataloaders[stage], start=1):

            self._global_step[stage] += 1
            if not self.fixed_noise:
                self.noise.normal_()
            real = real.to(self.device)
            fake = self.generator(self.noise)
            real_output, real_logits = self.discriminator(real)
            fake_output, fake_logits = self.discriminator(fake)

            d_loss = self.discriminator_criterion(real, fake, real_logits, fake_logits)
            real_acc, fake_acc = ((o.detach().cpu().round() == i).sum().item() / self.cfg.batch_size
                                  for i, o in enumerate((fake_output, real_output)))
            batch_info['d_loss'] = d_loss
            self.discriminator_losses[stage].append(d_loss.item())
            self.real_accuracy[stage].append(real_acc)
            self.fake_accuracy[stage].append(fake_acc)

            yield batch_info

            if self.cfg.debug and i % self.cfg.show_each == 0 or i == len(self.dataloaders[stage]):
                print(f'step : {i}/{len(self.dataloaders[stage])}', f'd_loss: {d_loss.item():.4}',
                      f'g_loss: {batch_info["g_loss"]:.4}', f'real_acc:{real_acc:.4}', f'fake_acc:{fake_acc:.4}',
                      sep='\t\t')

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1
        self.discriminator_losses[data_type] = []
        self.generator_losses[data_type] = []
        self.fake_accuracy[data_type] = []
        self.real_accuracy[data_type] = []
