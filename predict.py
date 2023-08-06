# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 11:28
# @Author  : luluwang
# @Email   : llwangxju@163.com
# @File    : predict.py
# @Software: PyCharm


from modeling_glm import GLMForConditionalGeneration
from tqdm import tqdm
import torch
from evaluate_utils import Evaluate
from tokenization_glm import  GLMTokenizer
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import re
import json


gpus = "0"
model_dir = "/data/wanglulu/GenRE/glm_init/models/robert/"
model = GLMForConditionalGeneration.from_pretrained(f"{model_dir}")
tokenizer = GLMTokenizer.from_pretrained(f"{model_dir}")

device = torch.device("cuda:{}".format(gpus))
simplify_model = model.to(device)
print("load success")

eval=Evaluate(tokenizer)

#生成问题模板
def question_generation(tag, school='', degree='', major='', name=''):
    if tag == 'UNV':
        return "Where was" + name.strip() + " educated?"
    elif tag == 'DEG':
        return "What degrees did " + name.strip() + " receive from " + school.strip() + "?"
    elif tag == 'START':
        return "When did " + name.strip() + " start " + degree.strip() + " in " + major.strip() + " at " + school.strip() + "?"
    elif tag == 'END':
        return "When did " + name.strip() + " receive " + degree.strip() + " in " + major.strip() + " from " + school.strip() + "?"
    elif tag == 'MAJ':
        return "What major did " + name.strip() + " take for the " + degree.strip() + " degree from " + school.strip() + "?"
    else:
        return None

#截断目标结果
def postprocess(text):
    answer = re.findall(r"\<\|startofpiece\|\> (.*) \<\|endofpiece\|\>", text)
    if len(answer) > 0:
        return answer[0].strip()
    else:
        return text.split("<|startofpiece|>")[-1].strip()


#根据token id 解码预测结果
def pred_ids_to_clean_text(generated_ids):
    gen_text = [postprocess(tokenizer.decode(
        generated_id.tolist())) for generated_id in generated_ids]
    return gen_text

#答案生成
def predict(context):
    encoding =tokenizer(
        context,
        padding="max_length",
        max_length=464,
        return_tensors="pt",
        truncation=True,
    )
    inputs = tokenizer.build_inputs_for_generation(encoding, max_gen_length=48,
                                                                padding=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        position_ids=inputs["position_ids"].to(device),
        generation_attention_mask=inputs["generation_attention_mask"].to(device),
        max_length=512,
        eos_token_id=tokenizer.eop_token_id,
        num_beams=8,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        top_p=0.95,
        top_k=50,
    )
    outputs=pred_ids_to_clean_text(outputs)[0]
    return outputs

#根据问题和上下文生成答案
def get_answer(question,context):
    raw_text = "<Qusetion>" + question + "[SEP]" + "<Context>" + context + "<Answer> [MASK]"
    sentence = predict(raw_text)
    answer_items = set([item.strip() for item in sentence.split('|')])
    return answer_items

#多字段生成
def multi_answer(id,name, context):
    question=question_generation(tag='UNV', name=name)
    schools = get_answer(question,context)
    result = []
    for sc in schools:
        question = question_generation(tag='DEG', name=name,school=sc)
        degrees = get_answer(question,context)
        for deg in degrees:
            question =question_generation(tag='MAJ', name=name, school=sc,degree=deg)
            majors = get_answer(question,context)
            for major in majors:
                res = [id, name, sc, deg, major]
                result.append(res)
                for tag in ['START', 'END']:
                    question =question_generation(tag=tag, name=name, school=sc, degree=deg,major=major)
                    answers = get_answer(question,context)
                    for answer in answers:
                        result[-1].append(str(answer))
    return result



if __name__ == '__main__':
    with open('/data/structure_test.json','r',encoding='utf-8') as f:
        test_data = json.load(f)
    title = ['id', 'name', 'school', 'degree', 'major', 'start', 'end']
    result = []
    for data in tqdm(test_data):
        id=data["id"]
        context = data['context']
        name = data["name"]
        context=name+"##"+context
        try:
            result.extend(multi_answer(id,name, context))
        except Exception as e:
            print(e)
            result.extend(multi_answer(id,name, context))
    ids = set()
    edu_all = []
    edu_record = {}
    for items in result:
        id, name, univ, degree, major, start_time, end_time = items
        if id not in ids:
            ids.add(id)
            edu_record = {}
            edu_record["id"] = str(id)
            edu_record["name"] = name
            edu_record["records"] = []
            record = {}
            record["univ"] = univ
            record["degree"] = degree
            record["major"] = major
            record["start_time"] = start_time
            record["end_time"] = end_time
            edu_record["records"].append(record)
            edu_all.append(edu_record)
        else:
            record = {}
            record["univ"] = univ
            record["degree"] = degree
            record["major"] = major
            record["start_time"] = start_time
            record["end_time"] = end_time
            edu_record["records"].append(record)
    pred_json= json.dumps(edu_all, ensure_ascii=False, indent=4)
    f1score = eval.F1Score(test_data, edu_all)
    print (f1score)
    with open('./'+'pred.json', 'w',encoding="utf-8") as json_file:
        json_file.write(pred_json)
