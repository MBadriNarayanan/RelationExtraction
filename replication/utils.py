import math
import torch

from torch import nn, Tensor
from torch.nn.parameter import Parameter


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    target_dims = target.dim()
    if target_dims != lprobs.dim():
        target = target.unsqueeze(-1)

    ce_loss = torch.gather(-lprobs, dim=-1, index=target)
    smoothing_component = torch.sum(-lprobs, dim=-1, keepdim=True)

    if ignore_index is not None:
        mask = (target != ignore_index).float()
        ce_loss *= mask
        smoothing_component *= mask
    else:
        ce_loss = ce_loss.squeeze(-1)
        smoothing_component = smoothing_component.squeeze(-1)

    total_ce_loss = torch.sum(ce_loss)
    total_smooth_loss = torch.sum(smoothing_component)

    smoothing_term = epsilon / lprobs.size(-1)
    final_loss = (1.0 - epsilon) * total_ce_loss + smoothing_term * total_smooth_loss

    return final_loss, total_ce_loss


class Trilinear(nn.Module):
    __constants__ = ["in1_features", "in2_features", "in3_features", "out_features"]

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        in3_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.in3_features = in3_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.Tensor(out_features, in1_features, in2_features, in3_features)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
        result = torch.einsum("bn,bm,bo,anmo->ba", input1, input2, input3, self.weight)
        if self.bias is not None:
            result = result + self.bias

        return result

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, in3_features={}, out_features={}, bias={}".format(
            self.in1_features,
            self.in2_features,
            self.in3_features,
            self.out_features,
            self.bias is not None,
        )


class BartTripletHead(nn.Module):
    def __init__(
        self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float
    ) -> None:
        super().__init__()
        self.feature_combiner = nn.Sequential(
            nn.Linear(input_dim * 3, inner_dim), nn.Tanh()
        )
        self.dropout = nn.Dropout(pooler_dropout)
        self.classifier = nn.Linear(inner_dim, num_classes)

    def forward(
        self, head_states: Tensor, tail_states: Tensor, context_states: Tensor
    ) -> Tensor:
        features = torch.cat(
            [
                self.dropout(tensor)
                for tensor in [head_states, tail_states, context_states]
            ],
            dim=1,
        )
        hidden = self.feature_combiner(features)
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


def shift_tokens_left(input_ids: Tensor, pad_token_id: int) -> Tensor:
    if pad_token_id is None:
        raise ValueError("pad_token_id must be provided")
    shifted = torch.zeros_like(input_ids)
    shifted[:, :-1] = input_ids[:, 1:]
    shifted[:, -1] = pad_token_id
    return shifted


def extract_triplets(text: str):
    def clean_text(t: str) -> str:
        return t.strip().replace("<s>", "").replace("<pad>", "").replace("</s>", "")

    def create_triplet(s: str, r: str, o: str):
        return {"head": s.strip(), "type": r.strip(), "tail": o.strip()}

    triplets = []
    current_subject = current_relation = current_object = ""
    state = "initial"
    text = clean_text(text).split()

    for token in text:
        if token == "<triplet>":
            if all(
                x.strip() for x in [current_subject, current_relation, current_object]
            ):
                triplets.append(
                    create_triplet(current_subject, current_relation, current_object)
                )
            state = "subject"
            current_subject = ""

        elif token == "<subj>":
            if all(
                x.strip() for x in [current_subject, current_relation, current_object]
            ):
                triplets.append(
                    create_triplet(current_subject, current_relation, current_object)
                )
            state = "object"
            current_object = ""

        elif token == "<obj>":
            state = "relation"
            current_relation = ""

        else:
            if state == "subject":
                current_subject += f" {token}"
            elif state == "object":
                current_object += f" {token}"
            elif state == "relation":
                current_relation += f" {token}"

    if all(x.strip() for x in [current_subject, current_relation, current_object]):
        triplets.append(
            create_triplet(current_subject, current_relation, current_object)
        )

    return triplets


def extract_triplets_typed(text, mapping_types= {'<peop>': 'Peop', '<org>': 'Org', '<other>': 'Other', '<loc>': 'Loc'}):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = mapping_types[token]
            else:
                current = 'o'
                object_type = mapping_types[token]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets
