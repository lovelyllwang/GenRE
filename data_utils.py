# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 10:04
# @Author  : luluwang
# @Email   : llwangxju@163.com
# @File    : data_utils.py
# @Software: PyCharm

import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import  PreTrainedTokenizer
import pytorch_lightning as pl
import transformers
import torch
transformers.logging.set_verbosity_error()


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data

        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.max_token_len=self.source_max_token_len + self.target_max_token_len
        self.all_data=[]
        for subdata in self.data:
            context=subdata["context"]
            for qa in subdata["qas"]:
                question=qa["question"]
                answer=qa["answer"]
                source="<Qusetion>"+question+"[SEP]"+"<Context>"+context+"<Answer> [MASK]"
                target="|".join(answer)
                self.all_data.append({"source":source,"target":target})

    def __len__(self):
        """ returns length of data """
        return len(self.all_data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.all_data[index]
        source_text=data_row["source"]
        target_text=data_row["target"]

        encoding = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.source_max_token_len,
            return_tensors="pt",
            truncation=True,
        )

        inputs= self.tokenizer.build_inputs_for_generation(encoding, targets=target_text, max_gen_length=self.target_max_token_len,
                                                       padding=True)
        inputs_nolabel=self.tokenizer.build_inputs_for_generation(encoding, max_gen_length=self.target_max_token_len,
                                                       padding=True)

        return dict(
            input_source_attention_mask=encoding["attention_mask"].flatten(),
            source_attention_mask=inputs["source_attention_mask"].flatten(),
            target_attention_mask=inputs["target_attention_mask"].flatten(),
            input_ids=inputs["input_ids"].flatten(),
            input_position_ids=inputs["position_ids"].view(-1,self.max_token_len),
            input_attention_mask=inputs["attention_mask"].view(-1,self.max_token_len,self.max_token_len),
            infer_input_ids=inputs_nolabel["input_ids"].flatten(),
            infer_position_ids=inputs_nolabel["position_ids"].view(-1,self.max_token_len),
            generation_attention_mask=inputs_nolabel["generation_attention_mask"].view(-1,self.max_token_len,self.max_token_len),
            labels=inputs["labels"].flatten()
        )



class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        super().__init__()

        self.train_df = train_df
        self.valid_df=valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
        self.valid_dataset = PyTorchDataModule(
            self.valid_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
