# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/26 14:54
@Author ： Wanglulu
@File ：trainer.py
@IDE ：PyCharm

"""
import argparse
import json
import pytorch_lightning as pl
from data_utils import LightningDataModule
from mrc_models import LightningModel
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tokenization_glm import  GLMTokenizer
from modeling_glm import GLMForConditionalGeneration


def get_args():
    parser = argparse.ArgumentParser(description="Generative Multi-turn Question Answering with Contrastive Learning for Entity-Relation Extraction.")
    parser.add_argument("--model_dir", type=str, default="THUDM/glm-large/", help="model dir")
    parser.add_argument("--train_data_dir", type=str, default="./data/train.json", help="train data dir")
    parser.add_argument("--valid_data_dir", type=str, default="./data/dev.json", help="dev data dir")
    parser.add_argument("--test_data_dir", type=str, default="./data/test.json", help="test data dir")
    parser.add_argument("--out_dir", type=str, default="./checkpoint/", help="checkpoint save dir")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--source_max_token_len", type=int, default=464, help="source max token length")
    parser.add_argument("--target_max_token_len", type=int, default=48, help="target max token length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="accumulate grad batches")
    parser.add_argument("--max_epochs", type=int, default=2, help="max epoch")
    parser.add_argument("--gpus", type=str, default='6,7', help="gpu")
    parser.add_argument("--resume_checkpoint_path", type=str, default="", help="resume checkpoint path")
    parser.add_argument("--save_top_k", type=int, default=1, help="save the top-k checkpoint")
    parser.add_argument("--random_seed", type=int, default=42, help="max epoch")
    parser.add_argument("--save_model_path", type=str, default="./model/", help="save model path")
    return parser.parse_args()

#定义参数解析器
args = get_args()
#设置随机种子
pl.seed_everything(args.random_seed)

#加载GLM模型和分词器
model = GLMForConditionalGeneration.from_pretrained(args.model_dir)
tokenizer = GLMTokenizer.from_pretrained(args.model_dir)

#读取文件train、dev、test
with open(args.train_data_dir, 'r') as fr:
    train_data= json.loads(fr.read())[:10]
with open(args.valid_data_dir, 'r') as fr:
    valid_data= json.loads(fr.read())[:2]

with open(args.test_data_dir, 'r') as fr:
    test_data = json.loads(fr.read())[:2]



PLM_Model = LightningModel(model=model, tokenizer=tokenizer, learning_rate=args.learning_rate)
data_module = LightningDataModule(
    train_data,
    valid_data,
    test_data,
    tokenizer=tokenizer,
    batch_size=args.batch_size,
    source_max_token_len=args.source_max_token_len,
    target_max_token_len=args.target_max_token_len,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.out_dir,
    filename="best-checkpoint-{epoch}-{val_loss:.4f}-{bleu_score:.4f}",
    save_top_k=args.save_top_k,
    verbose=True,
    monitor="bleu_score",
    mode="max",
    save_last=True,
)
early_stop_callback = EarlyStopping(
    monitor="bleu_score",
    patience=2,
    verbose=True,
     mode="max")

# pruning_callback = PyTorchLightningPruningCallback(trial, monitor="sari_score")

logger = TensorBoardLogger("PLMFineTuner_log", name="PLMFineTuner-Logger")
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback,early_stop_callback],
    max_epochs=args.max_epochs,
    devices=args.gpus,
    accumulate_grad_batches=args.accumulate_grad_batches,
    precision=16,
    accelerator="gpu",
    strategy="ddp",
    num_sanity_val_steps=2)

trainer.fit(PLM_Model, data_module)
trainer.test(PLM_Model, data_module, ckpt_path="best")
# trainer.test(PLM_Model, data_module, ckpt_path="last")

##保存模型
model_save=LightningModel.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,
                                                tokenizer=tokenizer, model=model,strict=True)
model_save.model.save_pretrained(args.save_model_path)
tokenizer.save_pretrained(args.save_model_path)

