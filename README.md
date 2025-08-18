# Generative-AI-with-Large-Language-Models


**LLM Use Cases and Tasks:**

Chat is just one use case: While LLMs are popularized by chatbots, their underlying next-word prediction ability powers many other applications.

**Text generation tasks:**

Essay writing from prompts

Summarization of dialogues or conversations

Translation (between human languages or natural language → machine code)

Code generation: Models can generate executable code, e.g., Python functions for data analysis.

Information retrieval: Tasks like named entity recognition (identifying people, places, etc. in text).

External augmentation: LLMs can connect to external data sources and APIs to access up-to-date or real-world information.

Scaling matters: Larger models (hundreds of billions of parameters) exhibit greater language understanding.

Smaller models still useful: They can be fine-tuned for specific, focused tasks.

Rapid advances: The growth in LLM capability is driven by architectural improvements in foundation models.



**Transformer Architecture**

**Transformer Structure**

**Two main components:**

1.Encoder (processes inputs).
2.Decoder (generates outputs).

Inputs = words → tokenization (convert to IDs/numbers).

**1. From RNNs → Transformers**

Old models (RNNs): Could only look at a few words at a time → bad at remembering long sentences.

Problem: Language needs context (example: “bank” could mean riverbank or money bank).

Solution (2017): Google’s paper “Attention is All You Need” → introduced the Transformer.

**2. Why Transformers Are Better**

They can look at all words in a sentence at once (not just one after another).

They figure out how much each word matters to every other word.

Example: In “The teacher taught the students with the book” → the model can work out who actually had the book.

This ability is called self-attention.

**3. How Transformers Process Text**

Tokenization – Break text into numbers (words or pieces of words).

Embeddings – Turn those numbers into vectors (so the computer understands meaning).

Similar words end up close together in this “map of meaning.”

Positional Encoding – Adds info about word order (so “dog bites man” ≠ “man bites dog”).

Self-Attention Layers – The model compares all words to all other words to find connections.

Done with multiple “attention heads”, each learning different things (e.g., grammar, names, rhymes).

Feed-Forward Network – Processes the attention results and prepares predictions.

Softmax Layer – Chooses the most likely next word out of all possibilities.

4. Why This Matters

RNNs: Struggled with long context → weak predictions.

Transformers: Can handle whole documents and learn deep relationships → big leap in performance.

This is the foundation of GPT, BERT, ChatGPT, and modern LLMs.

👉 In short:
Transformers changed the game because they understand context, see the whole picture, and scale efficiently with big data and GPUs.


🔑 How Transformers Make Predictions (Step by Step)
1. Translation Example (French → English)

- Input: French phrase → tokenized into numbers.

- Tokens go into the encoder:
  -Encoder creates a deep representation (understanding of meaning + structure).

- That representation goes into the decoder:
  -Decoder uses it to start predicting English tokens.

Decoder process:

- Begin with a start token.
- Predict next token (word).
- Feed that token back into the decoder.
- Repeat until it predicts an end token.

Output tokens → detokenized back into words → final translation.

2. Key Parts of the Transformer

Encoder

Reads the input sentence.

Produces a “context map” (deep understanding).

Decoder

Uses encoder’s context to generate output word by word.

Loops until it finishes the sentence.

3. Variations of Transformer Models

Encoder-only models → Good for understanding tasks (e.g., classification).

Example: BERT (sentiment analysis, question answering).

Encoder-decoder models → Good for sequence-to-sequence tasks (e.g., translation, summarization).

Examples: T5, BART.

Decoder-only models → Generate text word by word. Now the most common (scale well, generalize to many tasks).

Examples: GPT family, BLOOM, LLaMA, Jurassic.

4. Big Picture Takeaway

You don’t need to remember all technical details.

What matters is:

Transformers encode meaning (encoder).

Transformers generate outputs (decoder).

You’ll interact with them using prompts in natural language → this is called prompt engineering.

👉 In short:
Encoder understands → Decoder generates → Repeat until done.
Different models (BERT, T5, GPT) use these parts differently depending on the task.

📝 **Key Terminology**

**Prompt** → The text you give the model (the input).

**Completion** → The text the model generates (the output).

**Context window** → The maximum amount of text the model can “see” at once (input + output).

🎯 Prompt Engineering

The art of crafting your prompt to get better results.

Strategy: add examples inside the prompt → model performs better.

This technique is called in-context learning.

🤖 Inference Types (How the model learns from the prompt)

Zero-shot → No examples, just instructions.

e.g., “Translate this sentence into Spanish.”

One-shot → One example given.

e.g., Show 1 translation before asking for another.

Few-shot → Several examples given.

e.g., Provide multiple translations, then ask for a new one.

📊 **Model Performance**

Larger models → Better at zero-shot tasks.

Smaller models → Perform better with one-shot or few-shot prompts.

If performance is still poor → fine-tuning the model might be needed.

⏳ **Context Window Limitations**

The model can only handle a certain amount of text at once.

Too many examples = context overflow → in that case, fine-tuning is a better approach.


⚙️ **Inference Parameters (How to Control LLM Output)**
1. Max New Tokens

- Sets a limit on how many tokens the model can generate.
- Not exact — the model can still stop early (e.g., if it predicts an end-of-sequence token).

2. Decoding Strategies
a) Greedy Decoding

Always pick the word with the highest probability.

✅ Simple, good for short answers.

❌ Can get repetitive (loops).

b) Random Sampling

Picks words at random, weighted by their probability.

✅ More natural, varied output.

❌ Can be too “creative” → nonsense at times.

3. Sampling Controls

Top-k sampling

Model only chooses from the k most likely words.

Example: k=3 → choose randomly from top 3 words.

✅ Adds randomness but keeps output sensible.

Top-p sampling (nucleus sampling)

Model chooses from words that together add up to probability p.

Example: p=0.3 → only considers words whose combined likelihood = 30%.

✅ Dynamic and flexible — focuses on most reasonable words.

4. Temperature

Controls creativity vs. predictability:

Low (<1): Safer, more focused, predictable answers.

High (>1): More creative, varied, sometimes unpredictable.

=1: Default (no change to probabilities).


**Generative AI Project Life Cycle (Simplified)**

When you build an LLM-powered app, you can think of it like following a roadmap from idea → launch.

**1. Define the Scope**

Decide what exactly your model should do.

Ask: Do I need a model that can do many things (like writing long essays), or just one thing really well (like finding names in text)?

Being specific saves time and cost.

**2. Choose a Model**

Two options:

Use an existing model (most common and faster).

Train your own model from scratch (rare, expensive, only for special cases).

**3. Adapt and Improve the Model**

Test performance first.

Try prompt engineering → give good examples in your prompts to guide the model.

If that’s not enough:

Fine-tune (train the model with extra data for your task).

Later, you can use RLHF (Reinforcement Learning with Human Feedback) to make the model align better with human preferences (be safer, more helpful).

**4. Evaluate**

Always check: Is the model doing well?

Use metrics, benchmarks, and feedback.

This step is iterative:

Try → Evaluate → Improve → Repeat until results are good enough.

**5. Deploy**

Once the model is ready, integrate it into your app.

Optimize it so it runs fast and efficiently for users.

**6. Build Extra Infrastructure**

LLMs have limits:
- They sometimes make things up.
- They struggle with complex reasoning or math.
- To fix this, you may need extra tools like:
- Databases for facts.
- External reasoning systems (like calculators, search, or APIs).

