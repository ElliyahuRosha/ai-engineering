# Learning Progress: Word Embeddings POC

## מה הפרויקט

skip-gram עם Negative Sampling ב-PyTorch, מאפס: בלי gensim, בלי וקטורים מאומנים מראש.
קורפוס TinyStories (20,104,405 טוקנים), אוצר מילים 11,381 מילים (min_count=5).

התוצר: מחברת שמדגימה כל שלב עם הפלט שלו, מודל מאומן ששמור בריפו, ועמוד HTML אחד שמסכם.

## מה עובד ונבדק (הוכח בהרצה)

- `prepare_corpus.py` — 115,537 סיפורים → `data/tinystories_20m.txt` (103MB), deterministic.
- `word_embeddings.py` — tokenize, build_vocab, encode (np.fromiter), sample_pairs (מדלג על UNK),
  negative_distribution (UNK במשקל 0), sample_negatives.
- `skipgram.py` — שתי טבלאות nn.Embedding, אתחול U(±0.5/d), דוט-פרודכט. 2,276,200 פרמטרים.
- `train.py` — Adagrad lr=0.1, batch 8192, window 2, K=5, 3 מעברים = 150M זוגות.
  checkpoint מלא (model+optimizer+שני ה-RNG) כל 1,000 צעדים; `--resume` ממשיך בדיוק מוחלט
  (אומת: 200 צעדים רצוף מול 100+resume+100 → max diff 0.0). `--init` מתחיל מ-embeddings.npy.
- תוצאות המודל ב-`outputs/`: baseline 0.232, group margin +0.295, 1 PC1=5.2%,
  king/queen 0.836, dog/cat 0.793, rabbit/bunny 0.895, garden/backyard 0.905.
- `walkthrough.ipynb` — 59 תאים, 10 פרקים, רץ נקי ב-~85s על CPU, אפס שגיאות.
- `index.html` — עמוד סיכום (341KB), נבנה ע"י `make_page.py` מתוך `outputs/` בלבד.
- `outputs/embedding_map.html` — מפה תלת-ממדית: 25 מילים + 50 שכנים, hover עם קוסינוס.

## החלטות מהותיות

- **אתחול** U(-0.5/d, 0.5/d) ולא N(0,1): ב-100 ממדים הדוט-פרודכט של N(0,1) סוטה ~10,
  ה-sigmoid רווי והגרדיאנט מת. עם התיבה הקטנה loss מתחיל ב-ln2=0.693 בדיוק.
- **Adagrad ולא Adam**: Adam מנרמל כל קואורדינטה לצעד ≈lr ולכן המרחב קורס לכיוון אחד.
- **חלון 2**: הקשרים הצמודים לוכדים תחביר ומורפולוגיה; חלון 5 נותן נושא כללי.
- **sparse=True על CPU** (gradient רק לשורות בשימוש); ב-CUDA טבלאות dense.

## מפת קבצים

- `AGENTS.md` — כללי העבודה. `LEARNING_PROGRESS.md` — קובץ זה.
- `word_embeddings.py` — קורפוס, טוקניזציה, אוצר מילים, דגימת זוגות ושליליים.
- `skipgram.py` — המודל.
- `train.py` — לופ האימון, checkpoint, כל הדגלים.
- `evaluate.py` — baseline, PC1, group margin, אנלוגיות, שכנים → `metrics.json`.
- `make_page.py` — בונה את `index.html`.
- `walkthrough.ipynb` — המחברת. `index.html` — עמוד הסיכום.
- `figures/` — loss, heatmap, map3d.
- `outputs/` — `embeddings.npy`, `vocab.txt`, `metrics.json`, `embedding_map.html`.
- `data/` — הקורפוס (git-ignored). `scratch/`, `runs/` — עבודה, git-ignored.

## הרצה

```
python prepare_corpus.py
python train.py                # ~6 דקות על 4 ליבות CPU
python evaluate.py outputs
jupyter lab walkthrough.ipynb
python make_page.py
```

## היכן עצרנו

המודל המאומן נשמר ב-`outputs/` ועולה ל-git יחד עם הקוד, המחברת והעמוד.
המחברת והעמוד נבנו מחדש מול המודל הזה. נותר: אישור המשתמש לפני commit.
