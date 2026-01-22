# test_regeneration.py - Test GPT-2 text regeneration

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Quantize for CPU
print("Quantizing model...")
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("✅ Model loaded and quantized\n")

# Test cases - typical OCR errors
test_cases = [
    "S1LENCE PLE SE",  # Number instead of I, missing A
    "N0 SM0KING",      # Zeros instead of Os
    "WELC ME",         # Missing O
    "EX T",            # Missing I
    "PR VATE",         # Missing I
    "D0 N0T ENT R",    # Multiple errors
    "5TOP",            # 5 instead of S
    "CAUT 0N",         # Missing I, 0 instead of O
    "ENTRANCE",        # Perfect text (should not change)
]

print("Testing text regeneration:")
print("="*60)

import time

for corrupted in test_cases:
    print(f"\nInput:  '{corrupted}'")
    
    start = time.time()
    
    # Encode input
    inputs = tokenizer.encode(corrupted, return_tensors="pt")
    
    # Generate completion
    with torch.no_grad():
        outputs = model_quantized.generate(
            inputs,
            max_length=inputs.shape[1] + 5,
            num_return_sequences=3,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    elapsed = (time.time() - start) * 1000
    
    print(f"Time:   {elapsed:.1f}ms")
    print("Outputs:")
    
    for i, output in enumerate(outputs):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        print(f"  {i+1}. '{generated}'")

print("\n" + "="*60)
print("✅ Testing complete!")
print("\nNotes:")
print("- GPT-2 generates multiple candidates")
print("- Choose the most similar to original")
print("- Works best with context/partial words")
print("- May be too creative for simple OCR fixes")
print("\nFor simple typos, consider using spell checker instead:")
print("  pip install textblob")
print("  from textblob import TextBlob")
print("  corrected = str(TextBlob(text).correct())")