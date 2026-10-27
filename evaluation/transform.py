import argparse
import json
import re 
import regex


def clean_repeated_substrings(text):
    """Clean repeated substrings  and \n in text"""
    text = text.replace(" ",'').replace("\n",'').replace("\u3000",'')
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # n = len(text)
    # if n<8000:
    #     return text
    # import pdb 
    # pdb.set_trace()
    # for length in range(2, n // 10 + 1):
    #     candidate = text[-length:] 
    #     count = 0
    #     i = n - length
        
    #     while i >= 0 and text[i:i + length] == candidate:
    #         count += 1
    #         i -= length

    #     if count >= 10:
    #         return text[:n - length * (count - 1)]  


    return text


def transform_data(input_path, output_path):
    '''
    transform_data 的 Docstring
    vllm 推理结果转换为评测格式，并且清理重复文本和所有的\n 
    :param input_path: 说明
    :param output_path: 说明
    '''
    transformed_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            transformed_item = {
                'predict': item['predict'],
                'label': item['label']
            }
            transformed_data.append(transformed_item)
    
    metric_data = []
    for i, item in enumerate(transformed_data):
        label = item.get('label', '')
        predict = item.get('predict', '')
        metric_item = {
            'ID':i+1,
            "Task_ID":1,
            "Response":clean_repeated_substrings(predict),
            "Answer_Simplified":clean_repeated_substrings(label)
        }
        metric_data.append(metric_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metric_data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    transform_data(args.input_path, args.output_path)
