import json


def get_score(pred_ans, ground_truth) -> float:
    if len(pred_ans) != len(ground_truth):
        return 0.0
    score = 0.0
    for block1, (block2, angle2) in zip(pred_ans, ground_truth):
        if block1 == block2:
            score += 1

    # return float(score/4.0)
    return 1.0 if score > 3.5 else 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    try:
        pred_ans = json.loads(answer_text)
        ground_truth = json.loads(ground_truth)
        acc_reward = get_score(pred_ans, ground_truth)
    except:
        is_format_error = True
        acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    acc = 1.0 if acc_reward>0.99 else 0.0
    tool_reward_base = 1.0 if count_vision_1 else 0.0
    # tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = 1.0 if not is_format_error else 0.0

    turn_reward = -0.05 * count_vision_1 if acc_reward>0.99 else -0.25
    score = 0.8 * acc_reward + 0.2 * format_reward + turn_reward

    return {'score': score,
            'acc': acc,
            'acc_reward': acc_reward,
            'format_reward': format_reward}
