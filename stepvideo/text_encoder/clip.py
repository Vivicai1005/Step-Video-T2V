import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os


class HunyuanClip(nn.Module):
    """
        Hunyuan clip code copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
        hunyuan's clip used BertModel and BertTokenizer, so we copy it.
    """
    def __init__(self, model_dir, max_length=77):
        super(HunyuanClip, self).__init__()
        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        self.text_encoder = BertModel.from_pretrained(os.path.join(model_dir, 'clip_text_encoder'))
        
    @torch.no_grad
    def forward(self, prompts, with_mask=True):
        self.device = next(self.text_encoder.parameters()).device
        all_last_hidden_states = []
        all_pooler_outputs = []

        for prompt in prompts:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            outputs = self.text_encoder(input_ids=text_inputs.input_ids.to(self.device),
                                        attention_mask=text_inputs.attention_mask.to(
                                            self.device) if with_mask else None, )
            all_last_hidden_states.append(outputs.last_hidden_state)
            all_pooler_outputs.append(outputs.pooler_output)

        last_hidden_states = torch.cat(all_last_hidden_states, dim=0)
        pooler_output = torch.cat(all_pooler_outputs, dim=0)

        return last_hidden_states, pooler_output
        