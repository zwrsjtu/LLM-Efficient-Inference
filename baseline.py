import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

# 建议设置：如果你的网络下载 HuggingFace 模型慢，取消下面这行的注释（删掉前面的 #）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    model_name = "EleutherAI/pythia-70m"
    print(f"正在加载模型: {model_name} ...")

    # 1. 加载 Tokenizer 和 模型
    # device_map="auto" 会自动根据你是否有显卡来决定用 GPU 还是 CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # 2. 准备测试文本 (Prompt)
    prompt = "Large language models are becoming more efficient because"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 推理测试
    print("模型加载成功！开始生成...")
    start_time = time.time()
    
    # generate 核心函数，max_new_tokens 指定生成长度
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()

    # 4. 解码并展示结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("模型输出内容:")
    print(generated_text)
    print("="*50)
    print(f"\n生成耗时: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()