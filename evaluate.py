import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import math

def calculate_ppl(model, tokenizer, text_segment):
    inputs = tokenizer(text_segment, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = math.exp(loss.item())
    return ppl

def measure_speed(model, tokenizer, prompt="The history of artificial intelligence is"):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1)
    ttft = time.time() - start_time

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    total_time = time.time() - start_time
    tpot = total_time / 50
    
    return ttft, tpot

def main():
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    print("\n正在加载 Wikitext 数据集...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    sample_text = "".join([t for t in test_data["text"] if len(t) > 100][:3])
    
    print("正在计算 Wikitext Baseline PPL...")
    base_ppl = calculate_ppl(model, tokenizer, sample_text)
    
    print("正在测试生成速度...")
    ttft, tpot = measure_speed(model, tokenizer)

    # --- 3. 打印结果 (建议保存这个表格，以后写 README 用) ---
    print("\n" + "="*30)
    print(f"Baseline 测试结果 (Pythia-70M)")
    print("-" * 30)
    print(f"Wikitext PPL: {base_ppl:.2f}")
    print(f"首字延迟 (TTFT): {ttft:.4f} s")
    print(f"单 token 耗时 (TPOT): {tpot:.4f} s")
    print("="*30)

if __name__ == "__main__":
    main()