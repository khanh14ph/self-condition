import os

# os.environ['HF_HOME'] = '/home/vuhl/cache'
# os.environ['HF_DATASETS_CACHE'] = '/home/vuhl/cache'
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import config
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
from transformers import HubertForCTC
from ezspeech.models.conformer_asr import     Conformer_self_condition_phoneme_share
from torch.utils.tensorboard import SummaryWriter
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel,HubertConfig,HubertFeatureEncoder,HubertFeatureProjection,_compute_mask_indices


writer = SummaryWriter(config.checkpoint_dir + "/detail")


class HubertModel1(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())


        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # encoder_outputs = self.encoder(
        #     hidden_states,
        #     attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,




        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
            attentions=None,
        )


class HubertForCTCSelfCondPhoneme(HubertForCTC):
    def __init__(self, config, target_lang=None):
        super().__init__(config)
        self.hubert = HubertModel1(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.adapter = nn.Linear(768, 512)
        self.conformer = Conformer_self_condition_phoneme_share(
            d_hidden=512,
            num_heads=4,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            vocab_size=config.vocab_size,
            phoneme_vocab_size=config.phoneme_vocab_size,
            inter_layer=config.inter_layer,
            phoneme_inter_layer=config.phoneme_inter_layer,
        )
        self.step = 0
        self.post_init()

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.hubert.feature_extractor._freeze_parameters()
    def freeze_base_layers(self):
        for param in self.hubert.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        phoneme_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )
        
        outputs = self.hubert(
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
        logits, lengths, inter_layer_lst, phoneme_inter_layer_lst = self.conformer(
            hidden_states, input_lengths
        )

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
            phoneme_flattened_targets = phoneme_labels.masked_select(
                phoneme_labels_mask
            )
            # print(phoneme_labels)
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            inter_loss_lst = []
            phoneme_inter_loss_lst=[]
            is_inter = False
            with torch.backends.cudnn.flags(enabled=False):
                final_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=False,
                )
                for idx, inter_softmax in zip(self.config.inter_layer, inter_layer_lst):
                    is_inter = True
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=False,
                    )
                    # writer.add_scalar(
                    #     f"loss char at layer {str(idx)}", temp_loss, self.step
                    # )
                    inter_loss_lst.append(temp_loss)

                for idx, inter_softmax in zip(
                    self.config.phoneme_inter_layer, phoneme_inter_layer_lst
                ):
                    is_inter = True
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        phoneme_flattened_targets,
                        input_lengths,
                        phoneme_target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=False,
                    )
                    inter_loss_lst.append(temp_loss)
                    # writer.add_scalar(
                    #     f"loss phoneme at layer {str(idx)}", temp_loss, self.step
                    # )
                
                if is_inter == True:
                    intermediate_loss=sum(inter_loss_lst) / len(
                            inter_loss_lst)
                    # if self.step < 40000:
                    #     alpha = self.step /80000
                    #     loss = (1-alpha) * final_loss + alpha * intermediate_loss
                    # else:
                    loss = 0.5 * final_loss + 0.5 * intermediate_loss
                else:
                    loss = final_loss
                writer.add_scalar(f"last loss", final_loss, self.step)

    

                self.step += 1
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # return SelfConditionModelOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, avg_intermediate_loss=avg_intermediate_loss
        # )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            
            hidden_states=hidden_states,
        )


class HubertForCTCSelfCondPhonemeInference(HubertForCTC):
    def __init__(self, config, target_lang=None):
        super().__init__(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.adapter = nn.Linear(768, 512)
        self.conformer = Conformer_self_condition_phoneme_share(
            d_hidden=512,
            num_heads=4,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            vocab_size=config.vocab_size,
            phoneme_vocab_size=config.phoneme_vocab_size,
            inter_layer=config.inter_layer,
            phoneme_inter_layer=config.phoneme_inter_layer,
        )
        self.step = 0
        self.post_init()
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        phoneme_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(
                f"Label values must be <= vocab_size: {self.config.vocab_size}"
            )
        
        outputs = self.hubert(
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
        logits, lengths, inter_layer_lst, phoneme_inter_layer_lst = self.conformer(
            hidden_states, input_lengths
        )

        loss = None

        intermediate_loss = []
        final_phoneme_lst=[]
        final_grapheme_lst=[]

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
            phoneme_flattened_targets = phoneme_labels.masked_select(
                phoneme_labels_mask
            )
            # print(phoneme_labels)
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            inter_loss_lst = []
            phoneme_inter_loss_lst=[]
            is_inter = False
            with torch.backends.cudnn.flags(enabled=False):
                final_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=False,
                )
                for idx, inter_softmax in zip(self.config.inter_layer, inter_layer_lst):
                    is_inter = True
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=False,
                    )
                    inter_loss_lst.append(temp_loss)

                for idx, inter_softmax in zip(
                    self.config.phoneme_inter_layer, phoneme_inter_layer_lst
                ):
                    is_inter = True
                    temp_loss = nn.functional.ctc_loss(
                        inter_softmax.transpose(0, 1),
                        phoneme_flattened_targets,
                        input_lengths,
                        phoneme_target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=False,
                    )
                    inter_loss_lst.append(temp_loss)
                    final_phoneme_lst
                
                if is_inter == True:
                    intermediate_loss=sum(inter_loss_lst) / len(
                            inter_loss_lst)
                    # if self.step < 40000:
                    #     alpha = self.step /80000
                    #     loss = (1-alpha) * final_loss + alpha * intermediate_loss
                    # else:
                    loss = 0.5 * final_loss + 0.5 * intermediate_loss
                else:
                    loss = final_loss
                writer.add_scalar(f"last loss", final_loss, self.step)

    

                self.step += 1
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # return SelfConditionModelOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, avg_intermediate_loss=avg_intermediate_loss
        # )
        return CausalLMOutput(
            logits=(logits,inter_layer_lst,phoneme_inter_layer_lst),
            hidden_states=hidden_states,
        )