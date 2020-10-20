import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import datetime
import random
from tqdm import tqdm, trange
import re
import nltk
import os
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertLMHeadModel,
)

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('dataset.csv')


max_l = 20  # Максимальное количество слов в реплике


def clean_text(text):
    """
    Функция для предобработки текста диалогов
    """
    text = str(text)
    text = text.lower()
    if text.islower():
        text = nltk.sent_tokenize(text)[0]  # Берём только предложение из реплики
        text = re.sub('[^а-яА-яё|0-9]', ' ', text)
        text = re.findall(r'\w+', text)
        if len(text) <= max_l:
            text = ' '.join(text)
            return text


df = df.sample(frac=1, random_state=1).reset_index(drop=True)
# df = df[: int(len(df)//2)] при необходимости сократить размер датасета
df = df.dropna()
df2 = df['answer'].apply(clean_text)
df1 = df['question'].apply(clean_text)

df = pd.DataFrame({'question': df1.values, 'answer': df2.values})
df = df.dropna().reset_index(drop=True)

print(df.shape)

train_ans = df['answer']
train_text = df['question']

assert len(train_ans) == len(train_text)

pretrained_model = 'DeepPavlov/rubert-base-cased'
batch_size = 32
lr = 5e-5
eps = 1e-8
epochs = 3
max_norm = 1
gradient_accumulation_steps = 8
global_step = 0
epochs_trained = 0
steps_trained_in_current_epoch = 0
save_every = 100
run_name = 'run'
av_loss = 0

tokenizer = BertTokenizer.from_pretrained(pretrained_model)


def text_to_id(tokenizer_foo, text_list):
    """
    Функция для токенизации текста и
    добавления паддинга
    Приводит все предложения к одной длин
    """
    max_length = 50
    tokenized_text1 = []
    for item in text_list:
        print(item)
        tokenized_text1.append(tokenizer_foo.convert_tokens_to_ids(tokenizer_foo.tokenize(item)))

    processed_input_ids_list1 = []
    for item in tokenized_text1:
        if len(item) < max_length:
            seq_list = [0] * (max_length - len(item))
            item = item + seq_list
        elif len(item) >= max_length:
            item = item[:max_length]
        processed_input_ids_list1.append(item)

    examples1 = []

    for i in range(len(processed_input_ids_list1)):

        examples1.append(tokenizer_foo.build_inputs_with_special_tokens(processed_input_ids_list1[i]))

    return examples1


train_ans_ids = text_to_id(tokenizer, train_ans)
train_text_ids = text_to_id(tokenizer, train_text)


train_a = torch.tensor(train_ans_ids)
train_q = torch.tensor(train_text_ids)

train_data = TensorDataset(train_q, train_a)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


model = BertLMHeadModel.from_pretrained(pretrained_model, is_decoder=True)

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=eps)

total_steps = len(train_dataloader) * epochs
print("total_steps = {}".format(total_steps))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


def flat_accuracy(preds, labels_foo):
    """
    Функция для подсчёта accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels_foo.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(spent_time_function):
    """
    Функция для замера времени
    """
    elapsed_rounded = int(round(spent_time_function))
    return str(datetime.timedelta(seconds=elapsed_rounded))


epoch_pbar = trange(epochs_trained, int(epochs))
for current_epoch in epoch_pbar:
    epoch_pbar.set_description(f"Epoch [{current_epoch+1}/{epochs}]")
    pbar = tqdm(train_dataloader, position=0)
    for step, batch in enumerate(pbar):
        print(step)

        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue
        model.train()

        inputs, labels = (batch[0], batch[1])
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss, *_ = model(inputs, labels=labels)
        loss.backward()
        tr_loss = loss.item()

        av_loss = (step*av_loss + tr_loss)/(step + 1)
        pbar.set_description(f"Average loss: {av_loss:.4f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        print('step before')
        if (step + 1) % gradient_accumulation_steps == 0:
            print('step after')
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if global_step % save_every == 0 and global_step > 0:
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join('runs', run_name, "{}-{}".format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

output_dir = 'runs/'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("")
print("Training complete!")
