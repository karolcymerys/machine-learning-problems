import os.path

import nltk
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import transforms

from dataset import COCOImageCaptioningTrainDataset
from model import EncoderCNN, DecoderRNN
from train import train
from vocabulary import MSCOCOVocabularyFactory, Vocabulary

BATCH_SIZE = 256 + 128
EMBED_SIZE = 300
HIDDEN_SIZE = 512
EPOCHS = 4
MS_COCO_ROOT = '/home/kcymerys/Datasets/COCO'

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_STATS = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}, Perplexity: {:.4f}'
LOG_EPOCH_STEP = 50

if __name__ == '__main__':
    nltk.download('punkt')
    vocabulary = MSCOCOVocabularyFactory.build_vocabulary_from_file('ms_coco_vocabulary.pkl')

    vocab_size = len(vocabulary)

    train_dataset = COCOImageCaptioningTrainDataset(
        vocabulary=vocabulary,
        annotations_filepath=os.path.join(MS_COCO_ROOT, 'annotations/train/captions_train2014.json'),
        images_path=os.path.join(MS_COCO_ROOT, 'train2014'),
        img_transform=TRAIN_TRANSFORMS
    )
    train_dataset_loader = data.DataLoader(dataset=train_dataset, num_workers=8, shuffle=True, batch_size=BATCH_SIZE)

    encoder = EncoderCNN().to(DEVICE).eval()
    decoder = DecoderRNN(2048, EMBED_SIZE, HIDDEN_SIZE, vocab_size, lstm_layers=3).to(DEVICE)

    params = {
        'optimizer': torch.optim.Adam(lr=0.0003, betas=(0.9, 0.999), params=decoder.parameters()),
        'criterion': torch.nn.CrossEntropyLoss(),
        'device': DEVICE,
        'vocab_size': vocab_size
    }

    batches = len(train_dataset_loader)
    metrics = {
     'epochs': EPOCHS,
     'batches': batches
     }
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for batch_idx, (imgs, captions) in enumerate(train_dataset_loader, 1):
            params, metrics = train(imgs, captions, encoder, decoder, params, {**metrics,
                                                                               'epoch': epoch,
                                                                               'batch_idx': batch_idx
                                                                               })

            epoch_loss += metrics['loss']
            if batch_idx % LOG_EPOCH_STEP == 0 or batch_idx == batches:
                print('\r' + BATCH_STATS.format(epoch, EPOCHS,
                                                batch_idx, batches,
                                                epoch_loss / batch_idx,
                                                np.exp(epoch_loss / batch_idx)))

        print('\r' + BATCH_STATS.format(epoch, EPOCHS,
                                        batches, batches,
                                        epoch_loss / batches,
                                        np.exp(epoch_loss / batches)))
        torch.save(decoder.state_dict(), os.path.join('models', 'decoder-resnt152-%d.pkl' % epoch))
