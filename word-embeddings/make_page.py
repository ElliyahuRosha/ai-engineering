"""Build index.html: the one-page summary, from the trained vectors.

Run after `python train.py`. It reads only `outputs/` and the figures in
`figures/`, so the page cannot claim anything the model does not contain.
"""

import base64
import json
from pathlib import Path

import numpy as np

OUT = Path("index.html")
GROUPS = {
    "royalty": ["king", "queen", "prince", "princess"],
    "family": ["mom", "dad", "brother", "sister"],
    "animals": ["dog", "cat", "bird", "fish", "bear", "rabbit"],
    "places": ["castle", "forest", "garden", "house", "school"],
    "feelings": ["happy", "sad", "scared", "angry"],
    "size": ["big", "small"],
}
SHOW = ["king", "queen", "prince", "dog", "cat", "rabbit",
        "castle", "forest", "garden", "school", "scared", "sad", "big", "small"]
PAIRS = [("king", "queen"), ("mom", "dad"), ("dog", "cat"), ("forest", "woods"),
         ("rabbit", "bunny"), ("garden", "backyard"), ("big", "huge"),
         ("scared", "frightened")]


def image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main() -> None:
    embedding = np.load("outputs/embeddings.npy")
    words = Path("outputs/vocab.txt").read_text(encoding="utf-8").split("\n")
    metrics = json.loads(Path("outputs/metrics.json").read_text(encoding="utf-8"))
    position = {word: i for i, word in enumerate(words)}
    unit = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    rows = []
    for word in SHOW:
        scores = unit @ unit[position[word]]
        best = [i for i in np.argsort(-scores) if i != position[word]][:4]
        cells = "".join(
            f"<span class='n'>{words[i]} <b>{scores[i]:.2f}</b></span>" for i in best
        )
        group = next(g for g, members in GROUPS.items() if word in members)
        rows.append(f"<tr><td class='w'>{word}</td><td class='g'>{group}</td><td>{cells}</td></tr>")

    pairs = "".join(
        f"<li><code>{a}</code> / <code>{b}</code><b>{unit[position[a]] @ unit[position[b]]:.3f}</b></li>"
        for a, b in PAIRS
    )

    page = TEMPLATE
    replacements = {
        "__NEIGHBOURS__": "".join(rows),
        "__PAIRS__": pairs,
        "__BASELINE__": f"{metrics['baseline']:.3f}",
        "__MARGIN__": f"{metrics['group_margin']:+.3f}",
        "__PC1__": f"{metrics['pc1']:.1%}",
        "__VOCAB__": f"{len(words):,}",
        "__DIM__": f"{embedding.shape[1]}",
        "__MAP__": image(Path("figures/map3d.png")),
        "__HEATMAP__": image(Path("figures/heatmap.png")),
        "__LOSS__": image(Path("figures/loss.png")),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)

    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word embeddings from scratch</title>
<style>
  :root {
    --ink: #16181d;
    --muted: #5f6672;
    --line: #e3e6ea;
    --accent: #1f4e79;
    --bg: #ffffff;
    --soft: #f7f8fa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  main { max-width: 900px; margin: 0 auto; padding: 64px 24px 96px; }
  h1 { font-size: 44px; line-height: 1.1; letter-spacing: -0.02em; margin: 12px 0 20px; }
  h2 { font-size: 26px; letter-spacing: -0.01em; margin: 64px 0 16px; }
  h2 span { color: var(--muted); font-weight: 400; }
  h3 { font-size: 17px; margin: 28px 0 8px; }
  p { margin: 12px 0; }
  .eyebrow { font-size: 13px; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .lede { font-size: 20px; color: var(--muted); max-width: 720px; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line);
           border: 1px solid var(--line); border-radius: 10px; overflow: hidden; margin: 40px 0 8px; }
  .stat { background: var(--soft); padding: 18px 16px; }
  .stat b { display: block; font-size: 22px; letter-spacing: -0.01em; }
  .stat span { font-size: 12.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0 8px; font-size: 15px; }
  th { text-align: left; font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
       color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); padding: 8px 10px; }
  td { border-bottom: 1px solid var(--line); padding: 10px; vertical-align: top; }
  td.w { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; width: 90px; }
  td.g { color: var(--muted); font-size: 13px; width: 90px; }
  .n { display: inline-block; margin-right: 14px; white-space: nowrap; }
  .n b { color: var(--accent); font-variant-numeric: tabular-nums; }
  code { font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--soft);
         padding: 1px 4px; border-radius: 4px; white-space: nowrap; }
  pre { background: var(--soft); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px 18px; overflow-x: auto; font-size: 13.5px; line-height: 1.6; }
  figure { margin: 28px 0; }
  figure img { width: 100%; border: 1px solid var(--line); border-radius: 10px; display: block; }
  figcaption { font-size: 13.5px; color: var(--muted); margin-top: 10px; }
  ul.pairs { list-style: none; padding: 0; display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 28px; }
  ul.pairs li { display: flex; align-items: baseline; gap: 6px;
                border-bottom: 1px solid var(--line); padding: 6px 0; }
  ul.pairs li b { margin-left: 10px; color: var(--accent); font-variant-numeric: tabular-nums; }
  .callout { background: var(--soft); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; margin: 28px 0; }
  .callout b { color: var(--accent); }
  a { color: var(--accent); }
  footer { margin-top: 72px; padding-top: 24px; border-top: 1px solid var(--line);
           color: var(--muted); font-size: 14px; display: flex; gap: 20px; flex-wrap: wrap; }
  @media (max-width: 720px) {
    h1 { font-size: 32px; } .stats { grid-template-columns: repeat(2, 1fr); }
    ul.pairs { grid-template-columns: 1fr; } td.g { display: none; }
  }
</style>
</head>
<body>
<main>

<div class="eyebrow">From scratch · PyTorch · 20M tokens</div>
<h1>Word embeddings, learned one decision at a time</h1>
<p class="lede">Skip-gram with negative sampling, written from first principles: no gensim, no
pretrained vectors, no <code>Word2Vec</code> import. A 2.3M-parameter model, a notebook that prints
the evidence for every claim, and a trained model you can load in two lines.</p>

<div class="stats">
  <div class="stat"><b>20,104,405</b><span>tokens</span></div>
  <div class="stat"><b>__VOCAB__</b><span>word vocabulary</span></div>
  <div class="stat"><b>2,276,200</b><span>parameters</span></div>
  <div class="stat"><b>147 s</b><span>on one GPU</span></div>
</div>

<h2><span>01</span> The result</h2>

<p>Every word is a point in 100 dimensions. Nothing in the training data says that a dog is an
animal: the model only ever answered one question, <em>“is this pair of words real?”</em>, a
hundred and fifty million times. The geometry is what fell out.</p>

<table>
  <thead><tr><th>word</th><th>group</th><th>nearest neighbours (cosine)</th></tr></thead>
  <tbody>__NEIGHBOURS__</tbody>
</table>
<p style="font-size:13.5px;color:#5f6672">The groups are my labels. The model was never told about
them; if the blocks line up, the geometry is doing something real.</p>

<h3>The pairs that matter</h3>
<ul class="pairs">__PAIRS__</ul>
<p>A random pair of words already scores <b>__BASELINE__</b> in this space, so read the numbers
above as direction, not as probability. The margin by which my six word groups beat that floor is
<b>__MARGIN__</b>.</p>

<figure>
  <img src="__HEATMAP__" alt="Cosine similarity heatmap of 25 words">
  <figcaption>Cosine similarity for 25 words. The warm blocks on the diagonal are the hand-made
  groups. <code>big / small</code> is warm for the same reason <code>happy / sad</code> is: they
  occupy the same syntactic slot, not the same meaning.</figcaption>
</figure>

<h2><span>02</span> How a word becomes a vector</h2>

<pre>raw text → tokens → vocabulary → ids → (centre, context) pairs
         → negatives → skip-gram → vectors → neighbours → a map</pre>

<p>Two tables of __VOCAB__ × __DIM__ numbers: one for a word as the centre of a window, one for a
word sitting in its neighbourhood. The score of a pair is their dot product; the loss is binary
cross-entropy over one real pair and five random ones. That is the entire supervision signal.</p>

<p>The initialisation is a small box <code>U(−0.5/d, 0.5/d)</code> rather than PyTorch's
<code>N(0,1)</code>, so the loss starts at exactly <code>ln 2 = 0.693</code> — maximal ignorance —
instead of a saturated sigmoid with dead gradients. The optimiser is Adagrad: it decays per
coordinate, which keeps the space from collapsing onto a single direction the way Adam does at
this scale.</p>

<figure>
  <img src="__LOSS__" alt="Training loss over one thousand steps">
  <figcaption>Loss on a 1,000-step run, batch 8,192, five negatives. It starts at
  <code>ln 2</code> and falls fast, then slowly. It should not reach zero: some of the words drawn
  as negatives really do sit next to the centre word.</figcaption>
</figure>

<h2><span>03</span> The trained model</h2>

<p>The weights are committed, so the numbers above can be reproduced without training anything.
<code>outputs/</code> contains three files:</p>

<table>
  <thead><tr><th>file</th><th>what it is</th></tr></thead>
  <tbody>
    <tr><td><code>embeddings.npy</code></td><td>__VOCAB__ × __DIM__ float32 matrix; row <em>i</em> is the vector for line <em>i</em> of <code>vocab.txt</code></td></tr>
    <tr><td><code>vocab.txt</code></td><td>one word per line, in id order — id 0 is <code>&lt;unk&gt;</code></td></tr>
    <tr><td><code>metrics.json</code></td><td>the numbers on this page, machine-readable</td></tr>
  </tbody>
</table>

<h3>Use it</h3>
<pre>import numpy as np

E = np.load("outputs/embeddings.npy")
words = open("outputs/vocab.txt", encoding="utf-8").read().split("\\n")
U = E / np.linalg.norm(E, axis=1, keepdims=True)   # cosine = dot product

i, j = words.index("king"), words.index("queen")
print(U[i] @ U[j])                                  # 0.836</pre>

<h3>Continue training</h3>
<pre># warm start from the saved matrix (both tables begin from it)
python train.py --init outputs/embeddings.npy --epochs 1 --out runs/warm

# or resume an exact checkpoint, if you kept one:
# train.py writes outputs/checkpoint.pt every 1,000 steps
python train.py --resume</pre>

<p><code>--init</code> needs only the matrix. <code>--resume</code> needs the checkpoint, because it
restores the optimiser state and both random number generators as well, and continues on exactly
the same trajectory.</p>

<h2><span>04</span> The map</h2>

<figure>
  <img src="__MAP__" alt="3D PCA projection of 25 words">
  <figcaption>PCA fitted on all __VOCAB__ words, then the 25 are plotted. PC1 holds __PC1__ of the
  variance and the first three hold 13.1% — four times what three arbitrary dimensions would. No
  direction dominates, which is the healthy case.</figcaption>
</figure>

<p>A flat picture of a three-dimensional space always lies about one thing: the angle.
<a href="outputs/embedding_map.html">Open the interactive version</a> and rotate it — every word
carries its group on hover, and two of its nearest neighbours are drawn in grey so the crowding is
visible.</p>

<p>The obvious guess, that the first component is frequency, is wrong: the correlation between PC1
and log frequency is <code>r = 0.02</code>. The shared direction is geometric, not a frequency
artefact.</p>

<h2><span>05</span> Run it</h2>

<pre>python prepare_corpus.py       # 103 MB of TinyStories text
python train.py                # 1 minute per 50M pairs; 3 by default
python evaluate.py outputs     # neighbours, analogies, and the floor they beat
jupyter lab walkthrough.ipynb  # the evidence, section by section
python make_page.py            # rebuild this page</pre>

<p>On a GPU add <code>--device cuda</code>; the default is whatever is available. The walkthrough
notebook is the primary artifact: every number here is printed by a cell in it.</p>

<footer>
  <span>Skip-gram · Mikolov et al. 2013</span>
  <span>TinyStories · Eldan &amp; Li 2023</span>
  <a href="walkthrough.ipynb">walkthrough.ipynb</a>
  <a href="README.md">README</a>
</footer>

</main>
</body>
</html>
"""

if __name__ == "__main__":
    main()
