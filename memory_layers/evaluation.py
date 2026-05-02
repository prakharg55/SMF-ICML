import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re
from torch.nn import functional as F

class ModelEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_response(self, prompt, max_new_tokens=200, stop_sequences=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Greedy for evaluation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def evaluate_triviaqa(self, num_samples=100, split="validation"):
        print(f"Evaluating on TriviaQA ({split}, {num_samples} samples)...")
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)
        if num_samples:
            dataset = dataset.select(range(num_samples))

        correct = 0
        total = 0

        for item in tqdm(dataset):
            question = item['question']
            aliases = item['answer']['aliases']
            
            prompt = f"Question: {question}\nAnswer:"
            response = self.generate_response(prompt, max_new_tokens=32)
            
            # Check if any alias is in the response (case insensitive)
            is_correct = any(alias.lower() in response.lower() for alias in aliases)
            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"TriviaQA Accuracy: {accuracy:.2%}")
        return accuracy

    def evaluate_gsm8k(self, num_samples=100, split="test"):
        print(f"Evaluating on GSM8K ({split}, {num_samples} samples)...")
        dataset = load_dataset("gsm8k", "main", split=split)
        if num_samples:
            dataset = dataset.select(range(num_samples))

        correct = 0
        total = 0

        for item in tqdm(dataset):
            question = item['question']
            answer = item['answer']
            # Extract the numerical answer from the ground truth (usually after ####)
            ground_truth = answer.split("####")[-1].strip()
            
            prompt = f"Question: {question}\nLet's think step by step.\nAnswer:"
            response = self.generate_response(prompt, max_new_tokens=256)
            
            # Extract number from response (simple heuristic: last number)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            if numbers:
                pred_ans = numbers[-1]
                if pred_ans == ground_truth:
                    correct += 1
            
            total += 1

        accuracy = correct / total
        print(f"GSM8K Accuracy: {accuracy:.2%}")
        return accuracy

    def evaluate_hellaswag(self, num_samples=100, split="validation"):
        """
        Evaluates HellaSwag using log-likelihood scoring for multiple choice.
        """
        print(f"Evaluating on HellaSwag ({split}, {num_samples} samples)...")
        dataset = load_dataset("hellaswag", split=split)
        if num_samples:
            dataset = dataset.select(range(num_samples))

        correct = 0
        total = 0

        for item in tqdm(dataset):
            ctx = item['ctx']
            endings = item['endings']
            label = int(item['label']) # 0-3

            scores = []
            for ending in endings:
                # Construct prompt
                text = f"{ctx} {ending}"
                
                # Tokenize
                encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = encodings.input_ids
                
                # Calculate loss/likelihood
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    # We want the loss on the 'ending' part only, but full sequence loss is a proxy
                    # For better accuracy, we should mask the context loss, but this is a quick check
                    loss = outputs.loss.item()
                    scores.append(-loss) # Higher score (lower loss) is better

            pred = np.argmax(scores)
            if pred == label:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"HellaSwag Accuracy: {accuracy:.2%}")
        return accuracy

    def evaluate_all(self, num_samples=50):
        results = {}
        results['trivia_qa'] = self.evaluate_triviaqa(num_samples=num_samples)
        results['gsm8k'] = self.evaluate_gsm8k(num_samples=num_samples)
        results['hellaswag'] = self.evaluate_hellaswag(num_samples=num_samples)
        # Add others as needed
        return results
