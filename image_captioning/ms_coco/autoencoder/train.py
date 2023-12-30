import sys
from typing import Dict, Tuple

import numpy as np
import torch

from image_captioning.ms_coco.autoencoder.model import EncoderCNN, DecoderRNN

TRAIN_STATS = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}, Perplexity: {:.4f}'


def train(imgs: torch.Tensor,
          captions: torch.Tensor,
          encoder: EncoderCNN,
          decoder: DecoderRNN,
          params: Dict,
          metrics: Dict[str, int]) -> Tuple[Dict, Dict]:
    imgs, captions = imgs.to(params['device']), captions.to(params['device'])

    encoder.zero_grad()
    decoder.zero_grad()

    context = encoder(imgs)
    outputs = decoder(context, captions)

    loss = params['criterion'](outputs.view(-1, params['vocab_size']), captions.view(-1))
    loss.backward()
    params['optimizer'].step()
    metrics['loss'] = loss.item()

    # Print training statistics (on same line).
    print('\r' + TRAIN_STATS.format(metrics['epoch'], metrics['epochs'],
                                    metrics['batch_idx'], metrics['batches'],
                                    metrics['loss'], np.exp(metrics['loss'])), end="")
    sys.stdout.flush()
    return params, metrics
