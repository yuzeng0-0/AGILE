from openai import OpenAI
import requests
import random
import re
import os

openai_api_key = "EMPTY"
openai_api_base_list = [
    # "http://172.30.52.123:8000/v1",
    # "http://10.39.3.123:18901/v1",
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://173.3.32.75:18901/v1"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models['data'][0]['id'])



def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1==0:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1==0:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    question_text = extra_info['question']
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()

    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    format_reward = 0.0 if is_format_error else 1.0
    score = 0.8 * acc_reward + 0.2 * format_reward

    return {'score': score,
            'acc': acc_reward,
            'acc_reward': acc_reward,
            'format_reward': format_reward}


if __name__ == '__main__':
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")