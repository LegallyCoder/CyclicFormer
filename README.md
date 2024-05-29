# CyclicFormer
<img src="https://github.com/LegallyCoder/CyclicFormer/assets/119312866/c8c4b941-471d-44b7-95d3-3da146652cdd" width="500" height="500" alt="cyclicformer">

Implementation of the CyclicFormer architecture with hf_integration. 

This architecture created by Talha Rüzgar Akkuş.

# History Of Development CyclicFormer:
One day, I found myself contemplating how to advance the transformer architecture. I pondered, "What if I introduce a cyclic loop into the RNN, enabling the first node to be aware of the last node?" Excited by the idea, I quickly implemented the code. To my amazement, CyclicRNN outperformed standard RNNs on several tasks. This led me to think, "Why not apply this concept to the transformer architecture?" By doing so, I could enhance inter-layer connectivity within the model, much like the human brain.

After examining existing transformer implementations, I intentionally avoided searching for similar architectures in the literature, aiming to create something unprecedented. I initially developed this from scratch using plain Torch, and later decided to integrate it with Hugging Face. And thus, I present to you: "CyclicFormer with hf_integration."

# Working Mechanism:
![](https://github.com/LegallyCoder/CyclicFormer/assets/119312866/5fe31a39-33f0-4ff1-b89c-de4b424f1373)

# Usage:
To use the **CyclicFormer**, follow these steps:

1. Clone the repository to your local machine.
   
```bash
git clone https://github.com/LegallyCoder/CyclicFormer
```
2. Open a terminal or command prompt and navigate to the script's directory.
```bash
cd src
```

3. Install the required packages using this command:

```bash
pip3 install -r requirements.txt
```

4. Open new python file at the script's directory.
```python
from modeling_cyclicformer import CyclicFormerForCausalLM
from transformers import AutoTokenizer

model = CyclicFormerForCausalLM.from_pretrained('Q-bert/CyclicFormer-10M')
tokenizer = AutoTokenizer.from_pretrained('Q-bert/CyclicFormer-10M')

text = "Hi"

input_ids = tokenizer.encode(text, return_tensors="pt")

output = model.generate(input_ids, max_length=20, num_beams=5, no_repeat_ngram_size=2)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

```

# For more:

You can reach me on,

[Linkedin](https://www.linkedin.com/in/talha-r%C3%BCzgar-akku%C5%9F-1b5457264/)

[Twitter](https://x.com/TalhaRuzga35606)

[Hugging Face](https://huggingface.co/Q-bert)
