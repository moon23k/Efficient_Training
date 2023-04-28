import torch, evaluate
import torch.nn as nn


class Tester:
	def __init__(self, config, model, tokenizer, test_dataloader):
		self.model = model
		self.tokenizer = tokenizer
		self.device = config.device
		self.dataloader = test_dataloader
		self.metric_module = evaluate.load('bleu')


	def test(self):
		self.mode.eval()

		greedy_score, beam_score = 0, 0

		with torch.no_grad():
			for batch in self.dataloader:
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)

				pred = self.model.generate(input_ids=input_ids,
										   attention_mask=attention_mask, 
										   use_cache=True)

				pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)

		print(greedy_score)
		print(beam_score)

		return greedy_score, beam_score


