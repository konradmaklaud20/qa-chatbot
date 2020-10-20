import torch
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
)
import torch.nn.functional as F


dir_name = 'runs/'
model = BertLMHeadModel.from_pretrained(dir_name)
tokenizer = BertTokenizer.from_pretrained(dir_name)
device = 'cpu'


max_history = 2
no_sample = False
max_length = 80
temperature = 1.0
top_k = 0
top_p = 0.8
no_info = False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:

        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(conversation, model, num_samples=1):
    context = torch.tensor(conversation, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(max_length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)

            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


history = []
speaker1_tag = '<speaker1>'
speaker2_tag = '<speaker2>'
speaker1_tag_id = tokenizer.convert_tokens_to_ids(speaker1_tag)
speaker2_tag_id = tokenizer.convert_tokens_to_ids(speaker2_tag)
history = f"""{speaker2_tag} Привет!{speaker1_tag} Чем могу помочт?"""
print(history)
history = history.split('\n')
while True:
    message = None
    while not message:
        message = input(f'{speaker2_tag} ')
        if message == 'h':
            print('\n'.join(history))
            message = None

    history.append(f'{speaker2_tag} {message}')

    recent_history = history[-(2*max_history):]

    history_str = '{}\n{}'.format('\n'.join(recent_history), speaker1_tag)

    history_enc = tokenizer.encode(history_str, add_special_tokens=True)
    with torch.no_grad():
        out_ids = sample_sequence(history_enc, model)
    out_ids = out_ids[:, len(history_enc):].tolist()[0]
    if not no_info:
        print(20*'-')
        print('Output of model:')
        full_output = tokenizer.decode(out_ids, clean_up_tokenization_spaces=True)
        print(full_output)
        print('\nInput to the model:')
        print(history_str)
        print(20*'-' + '\n')

    for i, out_id in enumerate(out_ids):
        if out_id in [speaker1_tag_id, speaker2_tag_id]:
            break
    answer = '{} {}'.format(speaker1_tag, tokenizer.decode(out_ids[:i]))
    print(answer)

    history.append(answer)
