# Perplexity Explorer &bull; The Degree of Confusion in LLMs

An interactive, single-file laboratory for exploring and understanding **Perplexity**—the standard metric used to quantify how "confused" or "surprised" a Language Model (LLM) is when predicting a sequence of text.

Built strictly with vanilla **HTML, CSS, and plain JavaScript**, running a real neural language model (**DistilGPT2**, 8-bit quantized) entirely in your browser via **Transformers.js**. No backend server, no build systems, no frameworks, and no API keys required.

---

## 🌟 Overview & Intuition

When an autoregressive language model generates text, it assigns a probability distribution over its entire vocabulary for each next token given the previous context:

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

- **Probability ($P$):** How likely the model thought this exact token was.
- **Surprise / Branching Factor ($1/P$):** How many equally likely words the model felt it had to choose from. A token with $P = 0.5$ has a surprise of $2$; a rare or unexpected token with $P = 0.01$ has a surprise of $100$.
- **Perplexity ($\text{PPL}$):** The **geometric mean** of the branching factors across all tokens in the sequence:

$$\text{Perplexity} = \left( \prod_{i=1}^{N} \frac{1}{P_i} \right)^{\frac{1}{N}}$$

### The "Degree of Confusion" Analogy
If a model reads a sentence and achieves a perplexity of **$8.0$**, it was on average as uncertain as someone making a uniform random choice among **$8$ equally plausible words** at each step.
- **Lower perplexity** = The model found the text predictable, natural, and fluent.
- **Higher perplexity** = The model was surprised or confused by unnatural phrasing, syntax errors, or unexpected word choices.

---

## 🎨 Design & Aesthetic Highlights

- **Minimalist & Distraction-Free:** Soft canvas background (`#f8fafc`) paired with crisp white cards (`#ffffff`), subtle borders (`#e2e8f0`), and soft elevation shadows.
- **Purple & Indigo Accents:** Modern gradient buttons (`#4f46e5` to `#7c3aed`), rounded corners (`12px` to `16px`), and soft status pills.
- **Explicit Space Visualization:** In GPT-2 tokenization, leading spaces are distinct tokens. Spaces are formatted visibly as `␣` (e.g., `␣love`, `␣Python`) so students can clearly see token boundaries.
- **Responsive & Lightweight:** Fast loading, zero dependencies outside the on-demand Transformers.js CDN script.

---

## 🧭 Application Sections

### 1. Section 1 – Type a Sentence
- **Interactive Input:** Type any sentence or choose from one of the quick presets:
  - `"I love Python"` (fluent, natural)
  - `"Python love I"` (broken grammar / un-idiomatic order)
  - `"The capital of France is Paris"` (factual, high confidence)
- **On-Demand Model Download:** DistilGPT2 (`Xenova/distilgpt2`, `{ dtype: "q8" }`) is only downloaded when the button is first clicked.
- **Live Status & Progress:** Displays download percentages for ONNX model weights and tokenizer files.
- **Graceful Error Handling:** If network access fails or WebAssembly encounters an error, a red banner displays the exact error message.

### 2. Section 2 – Probability of Each Token &rarr; Perplexity
- **Token Breakdown Table:** Displays token index `#`, decoded token with spaces as `␣`, probability $P$ (with percentage), and surprise $1/P$.
- **Step 1:** Displays the multiplication of all individual surprise values $\prod \frac{1}{P_i}$.
- **Step 2:** Displays the sequence length $N$ (number of evaluated tokens).
- **Step 3:** Prominently presents the final Perplexity $(\text{Product})^{1/N}$.
- **Log-Space Computation:** To prevent numerical overflow on longer sentences, the calculation is evaluated using log-probabilities:

$$\text{Perplexity} = \exp\left( \frac{1}{N} \sum_{i=1}^{N} -\ln P_i \right)$$

### 3. Section 3 – What Does the Model Predict Next?
- Extracts the raw logits at the **very last position** of the prompt.
- Computes stable softmax over the 50,257 vocabulary words.
- Visualizes the **Top 8 most likely next tokens** as horizontal progress bars with percentages, highlighting the #1 top candidate.

### 4. Section 4 – Try It by Hand (`Simulated, not from the model`)
- A live mathematical sandbox allowing students to edit token names and probabilities directly.
- Add (`+ Add token`) and delete (`Remove`) rows dynamically.
- Live updating of the table, Step 1, Step 2, and Step 3 as you type.
- Built-in verification presets:
  - `"I love Python"`: $P = [0.5, 0.8, 0.2] \implies \mathbf{2.32}$
  - `"All 1/8"`: $8$ tokens at $0.125$ each $\implies \mathbf{8.00}$
  - `"One surprising word"`: $P = [0.9, 0.9, 0.9, 0.01] \implies \mathbf{3.42}$

---

## 🔬 Mathematical Verification & Proofs

### Preset 1: `"I love Python"`
$$P_1 = 0.5 \implies \frac{1}{P_1} = 2.0$$
$$P_2 = 0.8 \implies \frac{1}{P_2} = 1.25$$
$$P_3 = 0.2 \implies \frac{1}{P_3} = 5.0$$
$$\text{Product} = 2.0 \times 1.25 \times 5.0 = 12.5$$
$$N = 3$$
$$\text{Perplexity} = (12.5)^{1/3} \approx \mathbf{2.32}$$

$$\text{Log calculation: } \exp\left(\frac{-\ln(0.5) - \ln(0.8) - \ln(0.2)}{3}\right) = \exp\left(\frac{0.69315 + 0.22314 + 1.60944}{3}\right) = \exp(0.84191) \approx \mathbf{2.32}$$

### Preset 2: `"All 1/8"`
$$N = 8, \quad P_i = 0.125 = \frac{1}{8} \implies \frac{1}{P_i} = 8$$
$$\text{Product} = 8^8 = 16,777,216$$
$$\text{Perplexity} = (8^8)^{1/8} = \mathbf{8.00}$$

### Preset 3: `"One surprising word"`
Shows why perplexity is so sensitive to rare tokens:
$$P = [0.9, 0.9, 0.9, 0.01]$$
$$\text{Product} = \left(\frac{1}{0.9}\right)^3 \times 100 \approx 1.3717 \times 100 = 137.17$$
$$N = 4$$
$$\text{Perplexity} = (137.17)^{1/4} \approx \mathbf{3.42}$$
*Even though three words had 90% confidence, a single 1% surprise word pulls the entire perplexity from $\sim 1.11$ up to $3.42$.*

---

## 🛠️ Key Technical Implementations

### 1. Context for the First Word (`50256`)
In GPT-2, token ID `50256` represents `<|endoftext|>`. An autoregressive model cannot compute the probability of a token without a prior state. By prepending `[50256]`:
$$\text{Tokens} = [50256, \text{token}_1, \text{token}_2, \dots, \text{token}_N]$$
The model uses position $0$ (the start marker) to predict token $1$, ensuring that the first word of the sentence receives a valid probability.

### 2. Explicit ONNX Tensors (`attention_mask` & `position_ids`)
When performing manual forward passes with Transformers.js on `Xenova/distilgpt2`, the underlying ONNX Runtime session requires all three inputs:
- `input_ids`: Tensor of shape `[1, seqLen]` containing the token IDs.
- `attention_mask`: Tensor of shape `[1, seqLen]` filled with `1n` (indicating full attention to all prompt tokens).
- `position_ids`: Tensor of shape `[1, seqLen]` containing `[0n, 1n, ..., (seqLen - 1)n]`.

### 3. Numerically Stable Softmax
To avoid `NaN` or floating-point overflow when computing exponentials of raw logits:
$$z_{\text{max}} = \max_{v} z_v$$
$$\log P_T = (z_T - z_{\text{max}}) - \ln \left( \sum_{v} e^{z_v - z_{\text{max}}} \right)$$
$$P_T = e^{\log P_T}$$

---

## 🚀 How to Run

Because ES Modules and WebAssembly are loaded via CDN, the file should be served via a local web server (or opened in a browser that supports ES Modules):

### Option A: Python Built-in Server (Recommended)
```bash
# In the project directory:
python3 -m http.server 8000
```
Open your browser to:
[http://localhost:8000](http://localhost:8000)

### Option B: Node.js `npx serve`
```bash
npx serve .
```

### Option C: VS Code Live Server
Right-click `index.html` and click **"Open with Live Server"**.

---

## 📋 Verification Checklist

| Requirement | Specification | Status |
| :--- | :--- | :---: |
| **Single File** | All HTML, CSS, and JS inside `index.html` | ✅ Verified |
| **No Frameworks** | Pure vanilla JS, no React, Vue, Tailwind, or backend | ✅ Verified |
| **Model** | `Xenova/distilgpt2` with `{ dtype: "q8" }` | ✅ Verified |
| **Start Token** | Prepended `50256` for 1st word probability | ✅ Verified |
| **Stable Softmax** | Max-subtraction before exponentiation | ✅ Verified |
| **Token Spaces** | Spaces displayed as `␣` | ✅ Verified |
| **3 Steps** | Step 1 (Product), Step 2 ($N$), Step 3 ($\text{PPL}$) | ✅ Verified |
| **Next Predictions** | Top 8 horizontal bars with highlight on #1 | ✅ Verified |
| **Preset 1 Math** | `"I love Python"` = **2.32** | ✅ Verified |
| **Preset 2 Math** | `"All 1/8"` = **8.00** | ✅ Verified |
