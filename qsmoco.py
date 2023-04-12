from torch import nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from avgpooling import AvgPooling
import torch


class QsMoco(nn.Module):
    def __init__(self, args):
        super(QsMoco, self).__init__()
        self.K = args.contrast_K  # the queue size K is 4096
        self.M = args.contrast_M  # the momentum coefficient m is 0.999
        self.T = args.contrast_T  # temperature τ is 0.07
        if 'bart' in args.model:
            config = BartConfig.from_pretrained(args.model)
            dim = config.d_model
            self.model = BartForConditionalGeneration.from_pretrained(args.model)

            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            self.encoder_k = BartEncoder(config, nn.Embedding(vocab_size, config.d_model, padding_idx))
        else:
            raise Exception('only support bart')

        for param_base, param_k in zip(self.model.get_encoder().parameters(), self.encoder_k.parameters()):
            param_k.data = param_base.data
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, args.contrast_K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.avg_pool = AvgPooling()
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # Momentum update of the key encoder
        for param_q, param_k in zip(self.model.get_encoder().parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.M + param_q.data * (1. - self.M)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  #
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys[:batch_size, :].T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def freeze_encoder(self):
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask, device=None, labels=None, train_contrast=False,
                hard_sample_ids=None, hard_sample_mask=None, alpha=1, beta=1, only_hard=False):
        out = self.model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_ids, labels=labels,
                         output_hidden_states=True, return_dict=True)
        loss = out.loss
        MLE_loss = loss.item()
        sims = []
        if train_contrast:
            q = self.avg_pool(out.encoder_last_hidden_state, src_mask)
            q = nn.functional.normalize(q, dim=1)
            with torch.no_grad():
                encoder_outputs = self.encoder_k(input_ids=tgt_ids, attention_mask=tgt_mask,
                                                 output_hidden_states=True,
                                                 return_dict=True)
                k = self.avg_pool(encoder_outputs.last_hidden_state, tgt_mask)
                k = nn.functional.normalize(k, dim=1)

            logits_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            logits_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            sim_pos = torch.mean(logits_pos)
            sim_neg = torch.mean(logits_neg)

            logits_simple = torch.cat([logits_pos, logits_neg], dim=1) / self.T
            contrast_label = torch.zeros(logits_simple.shape[0], dtype=torch.long).to(device)
            loss_simple = self.criterion(logits_simple, contrast_label)

            loss = loss + alpha * loss_simple

            if hard_sample_ids is not None:
                logits_hard = None
                cycle = hard_sample_ids.shape[0] // tgt_ids.shape[0]
                for i in range(cycle):
                    f = i * tgt_ids.shape[0]
                    t = (i + 1) * tgt_ids.shape[0]
                    with torch.no_grad():
                        hids = hard_sample_ids[f:t, :]
                        hmsk = hard_sample_mask[f:t, :]
                        hard_outputs = self.encoder_k(input_ids=hids, attention_mask=hmsk, output_hidden_states=True,
                                                      return_dict=True)
                        hard_k = self.avg_pool(hard_outputs.last_hidden_state, hmsk)
                        hard_k = nn.functional.normalize(hard_k, dim=1)
                    hardl = torch.einsum('nc,nc->n', [q, hard_k]).unsqueeze(-1)
                    if logits_hard is None:
                        logits_hard = hardl
                    else:
                        logits_hard = torch.cat([logits_hard, hardl], dim=1)

                sim_hard = torch.mean(logits_hard)
                logits_hard = torch.cat([logits_pos, logits_hard], dim=1) / self.T
                hard_label = torch.zeros(logits_hard.shape[0], dtype=torch.long).to(device)
                loss_hard = self.criterion(logits_hard, hard_label)

                loss = loss + beta * loss_hard
                sims = [sim_pos, sim_neg, sim_hard]

            self._dequeue_and_enqueue(k)
            self._momentum_update_key_encoder()
        elif only_hard:
            q = self.avg_pool(out.encoder_last_hidden_state, src_mask)
            q = nn.functional.normalize(q, dim=1)
            with torch.no_grad():
                encoder_outputs = self.encoder_k(input_ids=tgt_ids, attention_mask=tgt_mask,
                                                 output_hidden_states=True,
                                                 return_dict=True)
                k = self.avg_pool(encoder_outputs.last_hidden_state, tgt_mask)
                k = nn.functional.normalize(k, dim=1)

            logits_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            sim_pos = torch.mean(logits_pos)

            if hard_sample_ids is not None:
                logits_hard = None
                cycle = hard_sample_ids.shape[0] // tgt_ids.shape[0]
                for i in range(cycle):
                    f = i * tgt_ids.shape[0]
                    t = (i + 1) * tgt_ids.shape[0]
                    with torch.no_grad():
                        hids = hard_sample_ids[f:t, :]
                        hmsk = hard_sample_mask[f:t, :]
                        hard_outputs = self.encoder_k(input_ids=hids, attention_mask=hmsk, output_hidden_states=True,
                                                      return_dict=True)
                        hard_k = self.avg_pool(hard_outputs.last_hidden_state, hmsk)
                        hard_k = nn.functional.normalize(hard_k, dim=1)
                    hardl = torch.einsum('nc,nc->n', [q, hard_k]).unsqueeze(-1)
                    if logits_hard is None:
                        logits_hard = hardl
                    else:
                        logits_hard = torch.cat([logits_hard, hardl], dim=1)

                sim_hard = torch.mean(logits_hard)
                logits_hard = torch.cat([logits_pos, logits_hard], dim=1) / self.T
                hard_label = torch.zeros(logits_hard.shape[0], dtype=torch.long).to(device)
                loss_hard = self.criterion(logits_hard, hard_label)

                loss = loss + beta * loss_hard
                sims = [sim_pos, sim_hard]

            self._dequeue_and_enqueue(k)
            self._momentum_update_key_encoder()

        if hard_sample_ids is None:
            loss_hard = 0
        if not train_contrast:
            loss_simple = 0
        return loss, MLE_loss, alpha * loss_simple, beta * loss_hard, sims
