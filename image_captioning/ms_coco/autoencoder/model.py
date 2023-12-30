from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet152_Weights


class EncoderCNN(nn.Module):
    def __init__(self, output_size: int) -> None:
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.batch_norm = nn.BatchNorm2d(num_features=2048)
        self.fc = nn.Linear(resnet.fc.in_features, output_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.init_h = nn.Linear(1, hidden_size*num_layers)
        self.init_c = nn.Linear(1, hidden_size*num_layers)

    def forward(self, features, captions):
        batch_size = features.shape[0]

        # Remove end token and transform caption into embedding form
        captions = captions[:, :-1]
        captions_embeded = self.embedding(captions)

        # Prepare input for LSTM. We are passing whole sequence at once.
        features_reshaped = features.view(batch_size, 1, -1)
        lstm_input = torch.cat((features_reshaped, captions_embeded), dim=1)

        # Pass through LSTM and Linear layer
        lstm_output, _ = self.lstm(lstm_input, self.__init_hidden(features))
        fc_output = self.fc(lstm_output)
        return fc_output

    def __init_hidden(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_context = torch.mean(context, dim=1).view(context.size(0), -1)
        h0 = self.init_h(mean_context).view(self.num_layers, mean_context.size(0), self.hidden_size)
        c0 = self.init_c(mean_context).view(self.num_layers, mean_context.size(0), self.hidden_size)
        return h0, c0

    def beam_search(self, context: torch.Tensor,
                    b: int = 3,
                    start_token: int = 0,
                    stop_token: int = 1,
                    max_len: int = 20) -> list[int]:
        # TODO: Handle end token
        step_predictions, step_hidden = self.__model_prediction(context.unsqueeze(1), self.__init_hidden(context))
        start_token = torch.tensor(start_token, device=context.device)

        # Selection phase
        p = torch.tensor([[1.0 for _ in range(b)]], device=context.device).T
        tokens = torch.stack([start_token.view(1) for _ in range(b)])
        hidden = step_hidden

        # Step 1
        step_lstm_input = self.embedding(start_token.view(1, 1))
        step_predictions, step_hidden = self.__model_prediction(step_lstm_input, hidden)
        step_p, step_tokens = step_predictions.topk(b, dim=1)

        # Selection phase -> Expand to b
        tokens = torch.cat([tokens, step_tokens.T], dim=1)
        p = p * step_p
        hidden = (torch.stack([step_hidden[0] for _ in range(b)], dim=1).squeeze(2),
                  torch.stack([step_hidden[1] for _ in range(b)], dim=1).squeeze(2))

        i = 1
        while True:
            step_lstm_input = self.embedding(tokens[:, -1].view(b, 1))
            step_predictions, step_hidden = self.__model_prediction(step_lstm_input, hidden)
            step_p, step_tokens = step_predictions.topk(b, dim=1)

            # Selection phase
            step_global_p = p * step_p
            step_top_global_p, step_top_global_p_indices = step_global_p.view(1, -1).topk(b)

            x_id = torch.div(step_top_global_p_indices, b, rounding_mode='trunc')
            y_id = step_top_global_p_indices - x_id * b

            p = step_top_global_p.T
            tokens = torch.cat([tokens[x_id.T, :].squeeze(1), step_tokens[x_id, y_id].T], dim=1)
            hidden = (step_hidden[0][:, x_id, :].squeeze(1), step_hidden[1][:, x_id, :].squeeze(1))

            i = i + 1
            if i >= max_len:
                break

        _, best = p.topk(1, dim=0)
        return tokens[best.item(), :].tolist()

    def sample(self, inputs, max_len=20):
        output_sentence = []
        inputs = inputs.unsqueeze(1)

        hidden = self.__init_hidden(inputs)
        lstm_input = inputs
        while True:
            softmax_output, hidden = self.__model_prediction(lstm_input, hidden)
            _, token = torch.max(softmax_output, dim=1)
            output_sentence.append(token.item())

            if len(output_sentence) == max_len or output_sentence[-1] == 1:
                break

            lstm_input = self.embedding(token).unsqueeze(1)

        return output_sentence

    def __model_prediction(self, lstm_input, hidden):
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        fc_output = self.fc(lstm_output.view(lstm_output.size(0), self.hidden_size))
        softmax_output = F.softmax(fc_output, dim=1)
        return softmax_output, hidden
