from typing import Tuple, List

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import weight_norm
from torchvision.models import ResNet152_Weights


class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        return features \
            .permute(0, 2, 3, 1) \
            .view(features.size(0), -1, features.size(1))


class DecoderRNN(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 embeding_size: int,
                 hidden_size: int,
                 vocabulary_size: int,
                 attention_dim: int = 256,
                 lstm_layers: int = 2) -> None:
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embeding_size)

        self.attention = BahdanauAttention(encoder_dim, hidden_size, attention_dim)

        self.init_h = weight_norm(nn.Linear(encoder_dim, lstm_layers*hidden_size))
        self.init_c = weight_norm(nn.Linear(encoder_dim, lstm_layers*hidden_size))
        self.dropout = nn.Dropout(p=0.5)
        self.lstm_input_layer = nn.LSTMCell(encoder_dim+embeding_size, hidden_size)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for _ in range(lstm_layers-1)])

        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        hidden = self.init_hidden(features)
        embeddings = self.embedding(captions)

        outputs = []
        for token_id in range(embeddings.size(1)):
            context = self.attention(features, hidden[0][0])

            lstm_input = torch.cat((embeddings[:, token_id], context), dim=1)
            h, c = self.lstm_input_layer(lstm_input, hidden[0])
            hidden[0] = (h, c)

            for idx, lstm_layer in enumerate(self.lstm_layers, 1):
                lstm_input = self.dropout(h)
                h, c = lstm_layer(lstm_input, hidden[idx])
                hidden[idx] = (h, c)

            outputs.append(h)

        return self.fc(torch.stack(outputs, dim=1))

    def init_hidden(self, features: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        mean_encoder_out = features.mean(dim=1)
        hidden = (self.init_h(mean_encoder_out).view(len(self.lstm_layers)+1, features.size(0), -1),
                  self.init_c(mean_encoder_out).view(len(self.lstm_layers)+1, features.size(0), -1))

        return [
            (hidden[0][i, :, :], hidden[1][i, :, :]) for i in range(len(self.lstm_layers)+1)
        ]

    def greedy_search(self, features: torch.Tensor, max_sequence_length=25,
                      start_token: int = 0, stop_token: int = 1) -> List[int]:
        previous_token = torch.tensor(start_token).unsqueeze(0).cuda()
        hidden = self.init_hidden(features)

        sentence = [previous_token.item()]
        while len(sentence) < max_sequence_length and sentence[-1] != stop_token:
            embedding = self.embedding(previous_token)
            context = self.attention(features, hidden[0][0])
            lstm_input = torch.cat((embedding, context), dim=1)
            h, c = self.lstm_input_layer(lstm_input, hidden[0])
            hidden[0] = (h, c)

            for idx, lstm_layer in enumerate(self.lstm_layers, 1):
                h, c = lstm_layer(h, hidden[idx])
                hidden[idx] = (h, c)

            output = self.fc(h)
            scoring = torch.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]

            sentence.append(top_idx.item())
            previous_token = top_idx

        return sentence

    def prediction(self,
                   features: torch.Tensor,
                   embeddings: torch.Tensor,
                   hidden: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        context = self.attention(features, hidden[0][0])
        lstm_input = torch.cat((embeddings, context), dim=1)
        h, c = self.lstm_input_layer(lstm_input, hidden[0])
        hidden[0] = (h, c)

        for idx, lstm_layer in enumerate(self.lstm_layers, 1):
            h, c = lstm_layer(h, hidden[idx])
            hidden[idx] = (h, c)

        output = torch.softmax(self.fc(h), dim=1)
        return output, hidden

    def beam_search(self,
                    features: torch.Tensor,
                    b: int = 3,
                    start_token: int = 0,
                    stop_token: int = 1,
                    max_len: int = 20) -> list[int]:
        # TODO: Handle end token
        start_token = torch.tensor(start_token, device=features.device)
        step_hidden = self.init_hidden(features)

        # Selection phase
        p = torch.tensor([[1.0 for _ in range(b)]], device=features.device).T
        tokens = torch.stack([start_token.view(1) for _ in range(b)])
        hidden = step_hidden

        # Step 1
        step_embeddings = self.embedding(start_token).view(1, -1)
        step_predictions, step_hidden = self.prediction(features, step_embeddings, hidden)
        step_p, step_tokens = step_predictions.topk(b, dim=1)

        # Selection phase -> Expand to b
        tokens = torch.cat([tokens, step_tokens.T], dim=1)
        p = p * step_p
        hidden = [
            (torch.stack([step_hidden[0][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[0][1] for _ in range(b)], dim=1).squeeze(0)),
            (torch.stack([step_hidden[1][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[1][1] for _ in range(b)], dim=1).squeeze(0)),
            (torch.stack([step_hidden[2][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[2][1] for _ in range(b)], dim=1).squeeze(0))
        ]

        i = 1
        while True:
            step_embeddings = self.embedding(tokens[:, -1].view(b, 1)).view(b, -1)
            step_predictions, step_hidden = self.prediction(features, step_embeddings, hidden)
            step_p, step_tokens = step_predictions.topk(b, dim=1)

            # Selection phase
            step_global_p = p * step_p
            step_top_global_p, step_top_global_p_indices = step_global_p.view(1, -1).topk(b)

            x_id = torch.div(step_top_global_p_indices, b, rounding_mode='trunc')
            y_id = step_top_global_p_indices - x_id * b

            p = step_top_global_p.T
            tokens = torch.cat([tokens[x_id.T, :].squeeze(1), step_tokens[x_id, y_id].T], dim=1)
            hidden = [
                (step_hidden[0][0][x_id, :].squeeze(0), step_hidden[0][1][x_id, :].squeeze(0)),
                (step_hidden[1][0][x_id, :].squeeze(0), step_hidden[1][1][x_id, :].squeeze(0)),
                (step_hidden[2][0][x_id, :].squeeze(0), step_hidden[2][1][x_id, :].squeeze(0))
            ]

            i = i + 1
            if i >= max_len:
                break

        _, best = p.topk(1, dim=0)
        return tokens[best.item(), :].tolist()

    def beam_search_2(self,
                    features: torch.Tensor,
                    b: int = 3,
                    start_token: int = 0,
                    stop_token: int = 1,
                    max_len: int = 20) -> list[int]:
        # TODO: With defect
        start_token = torch.tensor(start_token, device=features.device).unsqueeze(0)
        step_hidden = self.init_hidden(features)

        # Selection phase
        p = torch.tensor([[1.0 for _ in range(b)]], device=features.device).T
        tokens = torch.stack([start_token.view(1) for _ in range(b)])
        hidden = step_hidden

        # Step 1
        step_embeddings = self.embedding(start_token)
        step_predictions, step_hidden = self.prediction(features, step_embeddings, hidden)
        step_p, step_tokens = step_predictions.topk(b, dim=1)

        # Selection phase -> Expand to b
        tokens = torch.cat([tokens, step_tokens.T], dim=1)
        p = p * step_p.T
        hidden = [
            (torch.stack([step_hidden[0][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[0][1] for _ in range(b)], dim=1).squeeze(0)),
            (torch.stack([step_hidden[1][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[1][1] for _ in range(b)], dim=1).squeeze(0)),
            (torch.stack([step_hidden[2][0] for _ in range(b)], dim=1).squeeze(0),
             torch.stack([step_hidden[2][1] for _ in range(b)], dim=1).squeeze(0))
        ]

        features = torch.stack([features for _ in range(b)], dim=0).squeeze(1)

        i = 1
        while True:
            non_completed_sentences_indices = (tokens[:, -1] != stop_token).nonzero().squeeze().view(-1)
            completed_sentences_indices = (tokens[:, -1] == stop_token).nonzero().squeeze().view(-1)

            step_embeddings = self.embedding(tokens[non_completed_sentences_indices, -1])
            step_predictions, step_hidden = self.prediction(
                features[non_completed_sentences_indices, :, :],
                step_embeddings,
                [(layer[0][non_completed_sentences_indices, :], layer[1][non_completed_sentences_indices, :]) for layer in hidden])
            step_p, step_tokens = step_predictions.topk(b, dim=1)

            # Selection phase
            step_global_p = torch.zeros((b, b), device=features.device)
            step_global_p[non_completed_sentences_indices, :] = p[non_completed_sentences_indices, :] * step_p
            step_global_p[completed_sentences_indices, :] = p[completed_sentences_indices, :] * torch.tensor([[1] +[0 for _ in range(b-1)]], device=features.device)

            step_global_tokens = torch.zeros((b, b), device=features.device, dtype=torch.long)
            step_global_tokens[non_completed_sentences_indices, :] = step_tokens
            if completed_sentences_indices.shape[0] > 0:
                step_global_tokens[completed_sentences_indices, :] = torch.tensor([[stop_token for _ in range(b)] for _ in range(completed_sentences_indices.shape[0])], device=features.device)

            step_top_global_p, step_top_global_p_indices = step_global_p.view(-1).topk(b)
            x_id = torch.div(step_top_global_p_indices, b, rounding_mode='trunc')
            y_id = step_top_global_p_indices - x_id * b

            p = p[x_id, :] * step_top_global_p.view(p[x_id, :].shape)
            tokens = torch.cat([tokens[x_id.T, :].squeeze(1), step_global_tokens[x_id, y_id].T.unsqueeze(dim=1)], dim=1)
            hidden = [
                (step_hidden[0][0][x_id, :].squeeze(0), step_hidden[0][1][x_id, :].squeeze(0)),
                (step_hidden[1][0][x_id, :].squeeze(0), step_hidden[1][1][x_id, :].squeeze(0)),
                (step_hidden[2][0][x_id, :].squeeze(0), step_hidden[2][1][x_id, :].squeeze(0))
            ]
            i = i + 1
            if i >= max_len or tokens[:, -1].sum() == b:
                break

        _, best = p.topk(1, dim=0)
        return tokens[best.item(), :].tolist()


class BahdanauAttention(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 decoder_dim: int,
                 attention_dim: int,
                 output_dim: int = 1) -> None:
        super(BahdanauAttention, self).__init__()
        self.W_a = weight_norm(nn.Linear(encoder_dim, attention_dim))
        self.U_a = weight_norm(nn.Linear(decoder_dim, attention_dim))
        self.v_a = weight_norm(nn.Linear(attention_dim, output_dim))
        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor, decoder_hidden: torch.Tensor) -> torch.Tensor:
        attention_1 = self.W_a(features)
        attention_2 = self.U_a(decoder_hidden)

        attention_tan = self.tanh(attention_1 + attention_2.unsqueeze(dim=1))
        attention_tan = self.dp(attention_tan)
        attention_scores = self.v_a(attention_tan).squeeze(dim=2)

        attention_weights = self.softmax(attention_scores)

        context = features * attention_weights.unsqueeze(dim=2)
        return context.sum(dim=1)
