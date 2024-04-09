from utterance import UTTERANCES
import numpy as np
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_next_utterance(prev_utterances, num_sents=5):
  tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
  dlg_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
  context = f"{tokenizer.bos_token}"
  for dialog in prev_utterances:
    context = f"{context}{dialog}{tokenizer.eos_token}"

  gen_len = 20

  bot_input_ids = tokenizer.encode(context, return_tensors='pt')
  print(bot_input_ids.shape)
  out_ids = dlg_model.generate(
      bot_input_ids, 
      top_p=0.8,
      temperature=0.9,
      max_new_tokens = gen_len, 
      num_beams = 2*num_sents,
      num_return_sequences = num_sents,
      no_repeat_ngram_size = 2
  )
  gen_text = []
  for oid in out_ids:
    gtext = tokenizer.decode(oid[bot_input_ids.size()[1]:])
    gtext = gtext.split(tokenizer.eos_token)[0]
    gen_text.append(gtext)

  print("GENERATED OPTIONS:")
  print(gen_text)
  return gen_text

