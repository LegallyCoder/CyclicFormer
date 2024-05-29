# CyclicFormer
<img src="https://github.com/LegallyCoder/CyclicFormer/assets/119312866/c8c4b941-471d-44b7-95d3-3da146652cdd" width="500" height="500" alt="cyclicformer">

Implementation of the CyclicFormer architecture with hf_integration. 

This architecture created by Talha Rüzgar Akkuş.

# History Of Development CyclicFormer:
One day, I found myself contemplating how to advance the transformer architecture. I pondered, "What if I introduce a cyclic loop into the RNN, enabling the first node to be aware of the last node?" Excited by the idea, I quickly implemented the code. To my amazement, CyclicRNN outperformed standard RNNs on several tasks. This led me to think, "Why not apply this concept to the transformer architecture?" By doing so, I could enhance inter-layer connectivity within the model, much like the human brain.

After examining existing transformer implementations, I intentionally avoided searching for similar architectures in the literature, aiming to create something unprecedented. I initially developed this from scratch using plain Torch, and later decided to integrate it with Hugging Face. And thus, I present to you: "CyclicFormer with hf_integration."

# Working Mechanism:
![](https://github.com/LegallyCoder/CyclicFormer/assets/119312866/5fe31a39-33f0-4ff1-b89c-de4b424f1373)
