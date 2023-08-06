# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 14:38
# @Author  : luluwang
# @Email   : llwangxju@163.com
# @File    : evaluate.py
# @Software: PyCharm

class Evaluate:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    def tokenization(self,old_dict):
        new_dict={}
        for key in old_dict:
            inputs = self.tokenizer([old_dict[key]],
                return_tensors="pt", padding="max_length", max_length=16)
            decode_input = [
                self.tokenizer.decode(g,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False,
                                 )
                for g in inputs["input_ids"]
            ][0]
            new_dict[key]=decode_input.strip()
        return new_dict

    def remove_space(self,old_dict):
        new_dict={}
        for key in old_dict:
            new_dict[key]=old_dict[key].strip()
        return new_dict

    def RecallScore(self,true_edu_records, pred_edu_records):
        tp = 0
        for true_edu_record in true_edu_records:
            k = len(true_edu_record)
            true_edu_record = self.tokenization(true_edu_record)

            shared_items_len = []
            for pred_edu_record in pred_edu_records:
                pred_edu_record = self.remove_space(pred_edu_record)
                pred_edu_record = self.tokenization(pred_edu_record)
                shared_items = set(pred_edu_record.items()) & set(true_edu_record.items())
                if shared_items_len != [] and len(shared_items) > shared_items_len[-1]:
                    shared_items_len.append(len(shared_items))
                elif shared_items_len == []:
                    shared_items_len.append(len(shared_items))
                else:
                    pass
            if len(shared_items_len)==0:
                score=0
            else:
                score = shared_items_len[-1] / k
            tp += score
        return tp

    def PrecisionScore(self,true_edu_records, pred_edu_records):
        tp = 0
        for pred_edu_record in pred_edu_records:
            k = len(pred_edu_record)
            pred_edu_record = self.tokenization(pred_edu_record)
            pred_edu_record = self.remove_space(pred_edu_record)
            shared_items_len = []
            for true_edu_record in true_edu_records:
                true_edu_record = self.tokenization(true_edu_record)
                shared_items = set(pred_edu_record.items()) & set(true_edu_record.items())
                if shared_items_len != [] and len(shared_items) > shared_items_len[-1]:
                    shared_items_len.append(len(shared_items))
                elif shared_items_len == []:
                    shared_items_len.append(len(shared_items))
                else:
                    pass
            score = shared_items_len[-1] / k
            tp += score
        return tp

    def F1Score(self,truemaps, predmaps):
        '''
        计算F1 得分,若ID比对不上，抛出IdError异常
        :param truemap:
        :param predmap:
        :return: f1score
        '''
        true_edu_num = 0
        pred_edu_num = 0
        tp_recall = 0.0
        tp_precision = 0.0
        num=0
        for truemap in truemaps:
            id=truemap["id"]
            true_edu_records = truemap["records"]
            true_edu_num += len(true_edu_records)
            pred_record = list(filter(lambda x: x["id"] == id, predmaps))
            # pred_record=[predmaps[num]]
            # print (pred_record)
            if len(pred_record)==0:
                pass
                #raise ValueError("id:{id} is not exist in your set".format(id=id))
            else:
                predmap = pred_record[0]
                pred_edu_records=[i for i in predmap.get("records") if i!={}]
                pred_edu_num += len(pred_edu_records)
                tp_recall += self.RecallScore(true_edu_records, pred_edu_records)
                tp_precision += self.PrecisionScore(true_edu_records, pred_edu_records)
            num+=1
        precision = tp_precision / pred_edu_num if pred_edu_num != 0 else 0
        recall = tp_recall / true_edu_num
        f1score = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0
        return precision, recall, f1score

