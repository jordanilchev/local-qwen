# DDTree Explained: A Beginner's Guide

*No coding or machine learning background required.*

## What Is This?

DDTree-MLX makes AI models run **faster on your Mac** — up to 2.6x faster — without losing any quality. It's completely free, completely private, and runs entirely on your machine.

## How AI Normally Generates Text

When you run an AI model locally on your Mac (instead of using ChatGPT's servers), the AI generates its response **one word at a time**. It thinks, picks a word, thinks again, picks the next word, and so on. This is like a chef who makes one dish at a time — perfectly fine, but slow.

**Autoregressive generation** is the fancy name for this. Your Mac does it at about **38 words per second** — fast enough, but we wanted more.

## The First Speedup: DFlash (The Rough Draft Trick)

Someone clever came up with this idea: what if we had a **small, fast, dumb AI** that guesses what the **big, smart, slow AI** is going to say next?

It's like having an eager assistant who scribbles a rough draft of the next 8 words really quickly. Then the big AI just has to **check the draft** instead of thinking from scratch. Checking is way faster than creating.

- The small AI guesses 8 words at once
- The big AI checks them all in one shot
- Usually 6-7 of the 8 guesses are correct
- The big AI fixes the wrong ones and moves on

This is called **DFlash** (speculative decoding), and it gets us to about **58-73 words per second** — roughly **1.5-2x faster** than the one-at-a-time approach.

## DDTree: The Choose-Your-Own-Adventure Trick

Here's where DDTree comes in. DFlash's problem is that it only guesses **one sequence** of 8 words. If it's wrong at word 3, words 4-8 are wasted.

**DDTree** (which stands for Diffusion Draft Tree) turns that single guess into a **choose-your-own-adventure book**:

```
         "The"
        /     \
     "cat"   "dog"
      /         \
   "sat"      "ran"
    /            \
  "on"         "fast"
```

Instead of guessing one path ("The cat sat on"), we guess **multiple paths** at once. If the big AI rejects "cat", no problem — we already have "dog" ready as a backup. We don't waste a whole round going back to guess again.

### An Analogy: Ordering Food at a Restaurant

- **Normal (autoregressive)**: You tell the waiter one item at a time. "I'll have the soup." Waiter walks to kitchen, comes back. "And the salad." Walks to kitchen, comes back. Slow.

- **DFlash**: Your friend who knows you well writes down what they think you'll order — all 8 items at once. The waiter checks with you: "Your friend guessed soup, salad, steak, fries, water, pie, coffee, and a cookie — first 6 are right?" You just nod and correct the last 2. Much faster.

- **DDTree**: Your friend writes down a **decision tree**: "They'll probably want soup, but if not, maybe the chowder. Then salad or Caesar. Then steak, but if they're feeling light, maybe the salmon..." More of those guesses land, so you order faster and the waiter makes fewer trips.

### Another Analogy: Guessing a Song

Imagine you're trying to guess what song someone is humming:

- **Normal**: You guess one note at a time. "Is it C? Yes! Is it D? Yes! Is it E? No..." Start over from E.
- **DFlash**: You guess 8 notes in a row. "Is it C-D-E-F-G-A-B-C?" They check and say "first 5 were right."
- **DDTree**: You guess a **tree of possibilities**. "It starts C-D, then probably E but maybe E-flat, then F or F-sharp..." More guesses get accepted per round because you covered the likely alternatives.

## The Results

| Method | Speed | How much faster |
|--------|------:|:----------------|
| Normal (one word at a time) | ~28 words/sec | baseline |
| DFlash (single rough draft) | ~39 words/sec | ~1.4x faster |
| **DDTree (tree of drafts)** | **~42 words/sec** | **~1.5x faster** |

These are real measured numbers on a code generation prompt. DDTree adds about **10-15% on top of DFlash**, for a total of about **1.5x over autoregressive**.

**Important caveat**: DDTree helps most with code and structured content. For creative writing and open-ended prose, the draft model's acceptance rate drops to 5-10%, and DDTree is roughly the same speed as autoregressive.

## Is There Any Loss of Quality?

**No. Zero. None.**

This is the best part. DDTree produces **exactly the same output** as the basic one-word-at-a-time method. It's not an approximation or a shortcut — it's a way of **verifying faster**, not thinking less.

Here's why: the big, smart AI still checks every single word. The tree just helps it check more candidates at once. If a guessed word is wrong, it gets rejected — just like in the slow method. The only difference is that DDTree has backup guesses ready, so it wastes less time when a guess is wrong.

Think of it like spell-check: whether you type one word at a time or paste a whole paragraph, the spell-checker catches the same errors. DDTree just lets the spell-checker look at more words per pass.

**The technical term is "lossless speculative decoding"** — the output is mathematically identical to what you'd get without any speculation.

## What's "Acceptance Rate"?

When the small AI guesses words and the big AI checks them, the **acceptance rate** is how often the guesses are right.

- **High acceptance (86%)**: The small AI predicted the big AI really well. DFlash is already fast here, and DDTree doesn't add much (because there's nothing to fix).
- **Moderate acceptance (68-70%)**: The small AI got some wrong. This is where DDTree shines — its backup branches catch the misses, recovering tokens that DFlash would have wasted.

DDTree helps most when the draft model struggles, which tends to happen on creative writing, complex reasoning, and open-ended explanations.

## Why This Matters

1. **Privacy**: Everything runs on YOUR Mac. Nothing leaves your machine. No cloud, no subscription, no one reading your prompts.

2. **Speed**: A 27-billion-parameter AI model — genuinely smart, can write code, analyze documents, explain complex topics — runs at **~42 words per second** on a Mac Studio with DDTree, up from ~28 without it. That's fast enough to feel instant.

3. **Free after hardware**: Once you have the Mac, there's no per-token cost. Generate millions of words for $0.

4. **No quality trade-off**: Unlike some speed tricks that make the AI dumber, DDTree keeps the exact same output quality.

5. **Works best for code and structured content**: The speed boost depends on how well the draft model predicts. For code generation (85% prediction accuracy), DDTree adds 10-15%. For creative prose (5-10% accuracy), the benefit disappears.

## What Made This Hard

The model we optimized (Qwen 3.5 27B) has a unusual brain architecture. 75% of its "thinking layers" are sequential — they can only process one thing at a time, like a calculator. The other 25% are parallel — they can check multiple things at once, like a librarian looking up several books.

DDTree's tree-based approach naturally benefits the parallel layers (they can check all tree branches simultaneously), but the sequential layers have to check each branch one by one. This is why we built a custom Metal shader — a special program that runs directly on the Mac's GPU chip — to make even the sequential layers handle tree branches more efficiently.

Despite this architectural challenge, DDTree still achieves a meaningful speedup because it concentrates its guesses on the most likely words, getting more right per verification round.

## Try It Yourself

If you have a Mac with Apple Silicon (M1 or newer) and at least 32GB of memory:

```bash
pip install dflash-mlx
git clone https://github.com/humanrouter/ddtree-mlx.git
cd ddtree-mlx
pip install -e .
python ddtree_server.py --port 8006
```

Then point any OpenAI-compatible app at `http://localhost:8006/v1` and enjoy faster local AI.
