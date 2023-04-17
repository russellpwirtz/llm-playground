import shutil
from transformers import PreTrainedTokenizerFast

original_model = ""
# destination = "results/checkpoint-6"
destination = "results/checkpoint-final"

tokenizer = PreTrainedTokenizerFast.from_pretrained(original_model)
tokenizer.save_pretrained(destination)
