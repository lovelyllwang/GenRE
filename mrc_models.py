# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 10:22
# @Author  : luluwang
# @Email   : llwangxju@163.com
# @File    : models.py
# @Software: PyCharm
import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from transformers import get_linear_schedule_with_warmup,AdamW
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn as nn
import re
from modeling_glm import VocabEmbedding
transformers.logging.set_verbosity_error()
pl.seed_everything(42)

class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(self, tokenizer, model,use_cl=False,learning_rate=1e-4):
        """
        initiates a PyTorch Lightning Model

        Args:
            tokenizer : mbart tokenizer
            model :  mbart model
            add_loss:whether add contrastive learning
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.use_cl=use_cl
        self.learning_rate = learning_rate
        self.tau=1.0
        self.projection = nn.Sequential(nn.Linear(model.config.hidden_size, model.config.hidden_size),
                                        nn.ReLU())


    def forward(self,input_ids=None,position_ids=None,attention_mask=None,labels=None,
                source_attention_mask=None,target_attention_mask=None,
                input_source_attention_mask=None,istrain=False):
        output = self.model(input_ids,position_ids,attention_mask,labels)
        hidden_state=output.hidden_state
        ce_loss=output.loss
        cl_loss=None
        if istrain:
            if self.use_cl:
                cl_loss=self.compute_cl_loss(input_ids,hidden_state,source_attention_mask,
                                             target_attention_mask)
        return ce_loss,cl_loss

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    #计算对比损失
    def compute_cl_loss(self,input_ids,hidden_sate,source_attention_mask,target_attention_mask):
        batch_size = input_ids.size(0)
        proj_h = self.projection(hidden_sate)
        avg_source = self.avg_pool(proj_h, source_attention_mask)
        avg_target = self.avg_pool(proj_h, target_attention_mask)
        cos = nn.CosineSimilarity(dim=-1)
        sim_matrix = cos(avg_source.unsqueeze(1),
                         avg_target.unsqueeze(0))
        logits = sim_matrix / self.tau
        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(batch_size,
                             device=input_ids.device)
        cl_loss = cont_crit(logits, labels)

        return cl_loss

    def _step(self,batch,istrain=False):
        input_ids = batch["input_ids"]
        input_position_ids = batch["input_position_ids"]
        input_attention_mask = batch["input_attention_mask"]

        labels = batch["labels"]
        source_attention_mask=batch["source_attention_mask"]
        target_attention_mask = batch["target_attention_mask"]
        input_source_attention_mask=batch["input_source_attention_mask"]
        ce_loss,cl_loss= self.forward(
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
            input_ids=input_ids,
            position_ids=input_position_ids,
            attention_mask=input_attention_mask,
            input_source_attention_mask=input_source_attention_mask,
            labels=labels,
            istrain=istrain
        )
        return ce_loss,cl_loss

    def postprocess(self,text):

        answer = re.findall(r"\<\|startofpiece\|\> (.*) \<\|endofpiece\|\>", text)
        if len(answer)>0:
            return answer[0].strip()
        else:
            return text.split("<|startofpiece|>")[-1].strip()

    def pred_ids_to_clean_text(self, generated_ids):
        gen_text = [self.postprocess(self.tokenizer.decode(
            generated_id.tolist())
        ) for generated_id in generated_ids]
        return gen_text

    def refer_ids_to_clean_text(self, generated_ids):
        generated_ids[generated_ids[:, :] == -100] = 0
        gen_text = [self.tokenizer.decode(
            generated_id.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
        ) for generated_id in generated_ids]
        return gen_text

    def _generative_step(self,batch,data="val"):
        val_loss,_ = self._step(batch,istrain=False)
        # print (batch["generation_attention_mask"])
        outputs = self.model.generate(
            input_ids=batch["infer_input_ids"],
            position_ids=batch["infer_position_ids"],
            generation_attention_mask=batch["generation_attention_mask"],
            max_length=512,
            num_beams=5,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            top_p=0.95,
            top_k=50,
            eos_token_id=self.tokenizer.eop_token_id
        )
        preds = self.pred_ids_to_clean_text(outputs)
        # print (batch["labels"])
        targets = self.refer_ids_to_clean_text(batch["labels"])
        return {data+'_loss': val_loss, "preds": preds, "labels": targets}

    def training_step(self, batch, batch_size):
        """ training step """
        ce_loss,cl_loss= self._step(batch,istrain=True)

        if cl_loss is not None:
            total_loss = ce_loss +cl_loss
            loss_dict = dict(train_loss=total_loss, ce_loss=ce_loss, cl_closs=cl_loss)
            self.log_dict(loss_dict, prog_bar=True, logger=True)
            return total_loss
        else:
            total_loss = ce_loss
            loss_dict = dict(train_loss=total_loss, ce_loss=ce_loss)
            self.log_dict(loss_dict, prog_bar=True, logger=True)
            return total_loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        val_base_metrics=self._generative_step(batch,data="val")
        return val_base_metrics

    def test_step(self, batch, batch_size):
        """ test step """
        test_base_metrics=self._generative_step(batch,data="test")
        return test_base_metrics


    def configure_optimizers(self):
        """ configure optimizers """
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.06*self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def validation_epoch_end(self, validation_step_outputs):
        val_loss = np.round(
                    torch.mean(torch.stack([x["val_loss"] for x in validation_step_outputs])).item(),
                    4,
                )
        preds=[]
        targets=[]

        for x in validation_step_outputs:
            preds.extend(x["preds"])
            targets.extend(x["labels"])

        refers= [[target] for target in targets]
        preds = [pred for pred in preds]

        bleu_score=corpus_bleu(refers, preds,smoothing_function=SmoothingFunction().method3)
        loss_dict = dict(val_loss=val_loss, bleu_score=bleu_score)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return {'val_loss': val_loss,'bleu_score':bleu_score}


    def test_epoch_end(self, validation_step_outputs):
        test_loss = np.round(
                    torch.mean(torch.stack([x["test_loss"] for x in validation_step_outputs])).item(),
                    4,
                )
        preds=[]
        targets=[]

        for x in validation_step_outputs:
            preds.extend(x["preds"])
            targets.extend(x["labels"])

        refers= [[target] for target in targets]
        preds = [pred for pred in preds]

        bleu_score=corpus_bleu(refers, preds,smoothing_function=SmoothingFunction().method3)
        loss_dict = dict(test_loss=test_loss, bleu_score=bleu_score)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return {'test_loss': test_loss,'bleu_score':bleu_score}