import os

# os.environ['HF_HOME'] = '/home/vuhl/cache'
# os.environ['HF_DATASETS_CACHE'] = '/home/vuhl/cache'
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import config
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
)
from transformers import Wav2Vec2ForCTC
from ezspeech.models.conformer_asr import Conformer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(config.checkpoint_dir+"/detail")


class Wav2Vec2ForCTCSelfCondBaseline(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang=None):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.adapter = nn.Linear(768, 512)
        self.conformer = Conformer(
            d_input=512,
            d_hidden=512,
            num_heads=8,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            vocab_size=config.vocab_size,
        )
        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_layers(self):
        self.wav2vec2.encoder.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        input_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        logits, last_hidden, lengths = self.conformer(hidden_states, input_lengths)
        # print(f'final output: {tokenizer.batch_decode(torch.argmax(logits, dim=-1))}')

        loss = None

        intermediate_loss = []
        avg_intermediate_loss = 0

        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # print('main logits: ', logits.shape)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            # print('ctc softmax: ', log_probs.shape)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # return SelfConditionModelOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, avg_intermediate_loss=avg_intermediate_loss
        # )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2ForCTCSelfCond(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang=None):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.adapter = nn.Linear(768, 512)
        self.conformer = Conformer_self_condition(
            d_hidden=512,
            num_heads=8,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            vocab_size=config.vocab_size,
            inter_layer=[3, 6, 9],
        )
        self.post_init()

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_layers(self):
        self.wav2vec2.encoder.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        input_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        logits, lengths, inter_layer_lst = self.conformer(hidden_states, input_lengths)

        loss = None

        intermediate_loss = []
        avg_intermediate_loss = 0

        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            inter_loss_lst = []

            with torch.backends.cudnn.flags(enabled=False):
                final_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                for inter_softmax in inter_layer_lst:
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                    inter_loss_lst.append(temp_loss)
                loss = 0.8 * final_loss + 0.2 * sum(inter_loss_lst) / len(
                    inter_loss_lst
                )
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # return SelfConditionModelOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, avg_intermediate_loss=avg_intermediate_loss
        # )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Wav2Vec2ForCTCSelfCondPhoneme(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang=None):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.adapter = nn.Linear(768, 512)
        self.conformer = Conformer_self_condition_phoneme(
            d_hidden=512,
            num_heads=8,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            vocab_size=config.vocab_size,
            phoneme_vocab_size=config.phoneme_vocab_size,
            inter_layer=config.inter_layer,
            phoneme_inter_layer=config.phoneme_inter_layer,
        )
        self.step=0
        self.post_init()

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_layers(self):
        self.wav2vec2.encoder.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        phoneme_labels: Optional[torch.Tensor]=None
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        input_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        logits, lengths, inter_layer_lst,phoneme_inter_layer_lst = self.conformer(hidden_states, input_lengths)

        loss = None

        intermediate_loss = []
        avg_intermediate_loss = 0

        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)


            phoneme_labels_mask = phoneme_labels >= 0
            phoneme_target_lengths = phoneme_labels_mask.sum(-1)
            phoneme_flattened_targets = phoneme_labels.masked_select(phoneme_labels_mask)

            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            inter_loss_lst = []

            with torch.backends.cudnn.flags(enabled=False):
                final_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                for idx,inter_softmax in zip(self.config.inter_layer,inter_layer_lst):
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                    writer.add_scalar(f'loss char at layer {str(idx)}',temp_loss,
                                self.step)
                    inter_loss_lst.append(temp_loss)
                for idx,inter_softmax in zip(self.config.phoneme_inter_layer,phoneme_inter_layer_lst):
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        phoneme_flattened_targets,
                        input_lengths,
                        phoneme_target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                    inter_loss_lst.append(temp_loss)
                    writer.add_scalar(f'loss phoneme at layer {str(idx)}',temp_loss,
                                self.step)

                loss = 0.8 * final_loss + 0.2 * sum(inter_loss_lst) / len(
                    inter_loss_lst
                )
                writer.add_scalar(f'loss',final_loss,
                                self.step)

                self.step+=1
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # return SelfConditionModelOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, avg_intermediate_loss=avg_intermediate_loss
        # )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
