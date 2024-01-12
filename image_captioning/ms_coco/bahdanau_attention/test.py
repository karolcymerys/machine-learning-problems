import os
from typing import Dict

import nltk
import torch
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision.transforms import transforms

from dataset import COCOImageCaptioningTestDataset
from model import EncoderCNN, DecoderRNN
from vocabulary import MSCOCOVocabularyFactory, Vocabulary

EMBED_SIZE = 300
HIDDEN_SIZE = 512
DEVICE = 'cuda'

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


def test(imgs: torch.Tensor,
         encoder: EncoderCNN,
         decoder: DecoderRNN,
         params: Dict,
         vocabulary: Vocabulary) -> str:
    imgs = imgs.to(params['device'])

    features = encoder(imgs)
    outputs = decoder.greedy_search(features)
    print('Greedy search: {}'.format(' '.join(vocabulary.token_to_caption(outputs))))
    for i in range(2, 11):
        outputs = decoder.beam_search(features, b=i)
        print('Beam search {}: {}'.format(i, ' '.join(vocabulary.token_to_caption(outputs))))

    return ' '.join(vocabulary.token_to_caption(outputs))


if __name__ == '__main__':
    nltk.download('punkt')
    vocabulary = MSCOCOVocabularyFactory.build_vocabulary_from_file('ms_coco_vocabulary.pkl')
    vocab_size = len(vocabulary)

    test_dataset = COCOImageCaptioningTestDataset(
        annotations_filepath='/home/kcymerys/Datasets/COCO/annotations/test/image_info_test2014.json',
        images_path='/home/kcymerys/Datasets/COCO/test2014',
        img_transform=TEST_TRANSFORMS
    )
    test_data_loader = data.DataLoader(dataset=test_dataset, shuffle=True)


    encoder = EncoderCNN().to(DEVICE).eval()
    decoder = DecoderRNN(2048, EMBED_SIZE, HIDDEN_SIZE, vocab_size, lstm_layers=3).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join('models', 'decoder-resnt152-4.pkl')))
    decoder = decoder.eval()

    for org_img, transformed_img in test_data_loader:
        org_img, transformed_img = org_img, transformed_img.cuda()
        response = test(transformed_img, encoder, decoder, {'device': DEVICE}, vocabulary)

        plt.imshow(transforms.ToPILImage()(org_img.squeeze()))
        plt.show()
