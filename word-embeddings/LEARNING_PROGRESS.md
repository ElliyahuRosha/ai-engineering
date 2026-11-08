# Learning Progress: Word Embeddings POC

## כיוון הפרויקט

מסלול ראשי: **skip-gram עם Negative Sampling ב-PyTorch** (אתחול אקראי של טבלת embeddings).
הקורפוס: **TinyStories** (רדוד, סיפורי). PPMI+SVD נשארו בקוד כמסלול השוואה בלבד.

## מה עובד ונבדק (הוכח בהרצה)

- שלב 1–4 על WikiText-2: tokenize, vocab(19,964), cooccurrence, ppmi — רצו, אך PPMI לבדו
  נתן שכנים גרועים (rare-pair bias). תוצאה שלילית מתועדת.
- `prepare_corpus.py`: הוריד 115,537 סיפורים → `data/tinystories_20m.txt` (99MB),
  20,000,186 טוקנים, 30 שניות.
- סטטיסטיקות הקורפוס החדש: 20,104,405 טוקנים, 19,354 צורות, vocab(>=5)=11,381,
  UNK 0.07%.
- אומת שכל 25 מילות התצוגה המוצעות קיימות, counts 1,308–120,495.

## מה נכתב אך טרם נבדק

- כלום.

## סביבה

- kernel `python3` → `/root/.venvs/jupyter/bin/python`, Python 3.14.8.
- numpy 2.5.3, scipy 1.18.1, sklearn 1.9.1, pandas 3.0.5, matplotlib 3.11.2, torch 2.14.1+cpu,
  tensorflow 2.22.0-rc0, datasets 5.1.0, plotly 7.1.0.

## מפת קבצים

- `AGENTS.md` — כללי העבודה.
- `LEARNING_PROGRESS.md` — קובץ זה.
- `word_embeddings.py` — load_corpus, tokenize, build_vocab, encode, cooccurrence, ppmi,
  window_pairs, sample_pairs (מדלג על UNK), subsample_ids, negative_distribution, sample_negatives.
- `skipgram.py` — `class SkipGram(nn.Module)`: שתי טבלות embedding, אתחול U(±0.5/d), דוט-פרודכט.
  `sparse=True` כברירת מחדל (CPU); GPU משתמש בטבלאות dense כי אין sparse Adagrad ב-CUDA.
- `train.py` — argparse מלא (dim/window/negatives/lr/batch/epochs/steps/subsample/min-count/
  seed/device/out/save-every), ברירת מחדל = ההתנהגות הקודמת, batch 8192, Adagrad lr=0.1.
  שומר `checkpoint.pt` + `run.json` + embeddings בכל תיקיית `--out`.
- `evaluate.py` — baseline, PC1, group margin, 16 אנלוגיות (accuracy + median rank), שכנים
  ל-25 מילים; כותב `metrics.json`; `--compare` מדפיס טבלה אחת לכל הריצות.
- `grid.py` — 8 קונפיגים על ה-CLI, מדלג על ריצה שכבר יש לה `metrics.json`.
- `kaggle/` — `skipgram_kaggle.ipynb` (עצמאי לחלוטין: כל הקוד בתאים) + `README.md` (העלאה והרצה).
- `outputs/embeddings.npy`, `outputs/vocab.txt` — מודל epoch מלא (48,828 צעדים), בתוספת
  `embedding_map.html` (Plotly תלת-ממד) ו-`metrics.json`.
- `prepare_corpus.py` — הורדת דגימת TinyStories ושמירתה כטקסט.
- `walkthrough.ipynb` — המחברת המוצגת: 58 תאים, 10 פרקים, רצה נקי ב-85s.
- `data/tinystories_20m.txt` — הקורפוס (103MB, מכוסה ב-`.gitignore`).
- `.gitignore` — מתעלם מ-`data/`.

## מושגים שהוסברו

- count-based מול prediction-based; SVD בשני תפקידים; כשל rare-pair של PMI.
- skip-gram: טבלת E אקראית, זוגות (center, context), negative sampling, עדכון גרדיאנט,
  ולמה אתחול אקראי הכרחי (symmetry breaking).
- ארכיטקטורה (אושרה ומומשה): קלט = ids; labels = 1/0 לכל זוג (BCE, לא softmax); שתי טבלות
  nn.Embedding(11,381, 100); אתחול U(-0.5/d, 0.5/d); דוט-פרודכט; 2,276,200 פרמטרים.
  `sample_pairs` מסנן `<unk>` משני צדי הזוג ואומת (0 בבת').
- שלב 7 הושלם: `skipgram.py` (SkipGram) + פרק 7 במחברת. אומת: logits std=8.5e-5,
  sigmoid=0.5000, BCE=0.6931=ln2, cos(king,queen)=-0.017 לפני אימון.
- **פרק 8 במחברת (לופ אימון אמיתי):** 44 תאים, רץ ב-85s, אפס שגיאות. 1,000 צעדים /
  8.2M זוגות / 51s / 160k זוגות/s, loss 0.6931→0.3977, גרף לוס מוחלק + קו ln2.
  לקח שנתפס: cosine מוחלט מטעה (אנזוטרופיה) — הטבלה מציגה baseline של זוג אקראי
  (0.385) ועמודת "vs random": dog/cat +0.306, king/queen +0.155, king/banana +0.020,
  dog/castle -0.125. נוסף `torch.manual_seed(0)` בפרק 7 — קודם כל הרצה נתנה מספרים אחרים.
- **באג שנתפס בניסוי האימון:** עם Adam (lr=3e-3 ואפילו 1e-4) כל הווקטורים קרסו לכיוון אחד —
  cosine≈1.00 לכל זוג, PC1=97.6% מהשונות, גם אחרי centering. הסיבה: Adam מנרמל כל קואורדינטה
  לצעד ≈lr, ולכן כל הטיה משותפת קטנה מצטברת ליניארית. מעבר ל-Adagrad פותר: PC1=0.04–0.12,
  rand_cos=0.11–0.32. גם dense Adam היה איטי (30 steps/s) לעומת sparse Adagrad (113 steps/s).
- `nn.Embedding(sparse=True)` + Adagrad: 113 steps/s על CPU, loss 0.693→0.42 ב-300 צעדים.
- **epoch מלא (48,828 צעדים, 50M זוגות) הושלם ב-462.7s (7:42):** loss 0.411→0.3925,
  plateau מ-~25k צעדים. שכנים: king→prince 0.77, queen 0.76, knight 0.73;
  dog→rex 0.76, cat 0.74, puppy 0.74. נשמר `outputs/embeddings.npy`.
  הריצה הזו השתמשה בגרסה הישנה של `train.py` ולכן **אין לה checkpoint**.
- **checkpoint אומת בהרצה:** `train.py 100` ואז `train.py 100 --resume` →
  `resumed from step 100`, לוג ממשיך 105..200, loss רציף (0.4321→0.4323).
  הקובץ כולל model, optimizer(Adagrad state), step, numpy rng, torch rng. 18MB.
  מודל ה-200 צעדים וקובץ ה-checkpoint הועברו ל-/tmp/opencode כדי לא לדרוס את מודל ה-epoch.
- **כיול batch (נמדד על 2.05M זוגות לכל הגדרה):** 1024/2thr=118k זוגות/s, 1024/4=129k,
  2048=153k, 4096=146k, 8192/4=186–194k. batch=8192 נבחר (1.6x מהר יותר, PC1 זהה 0.12,
  dog→cat=0.68). חוטים: 1=117k, 2=149k, 3=133k, 4=194k (רעש 10–15% בין ריצות).
- **באג שנתפס בלוג המהירות:** `steps / elapsed` במקום `done / elapsed` — הציג 92 steps/s
  בזמן שהקצב האמיתי היה 18.3. תוקן; הלוג מדפיס עכשיו גם k pairs/s.
- **הוכחה ש-batch לא משנה את הלמידה לזוג:** ריצה חדשה (batch 8192) בצעד 1,220 = 10.0M זוגות
  → loss 0.3972. ריצה ישנה (batch 1024) בצעד 9,764 = 10.0M זוגות → loss 0.3972. זהה.
- ריצת epoch עם batch 8192 נעצרה ביוזמת המשתמש בצעד 1,220; לא נשמר checkpoint ממנה
  (SAVE_EVERY היה 5,000). הופחת ל-1,000 (~55s).
- מודל ה-epoch המלא (48,828 צעדים, batch 1024) שוחזר ל-`outputs/embeddings.npy`.
- **אודיט לפני Kaggle (scratch/audit.py, resume_identity.py, mem_probe.py):**
  `<unk>` נדגם כ-0.08% מהשליליים (באג — תוקן: `weights[0]=0`, מיליון הגרלות → 0).
  שליליים שגויים (נג == הקשר אמיתי) 0.60% — מקובל, לא תוקן.
  כל 11,380 המילים אומנו (רק `<unk>` לא); אין hubs חולניים (max mean-cos 0.44).
  25 מילות התצוגה — שכנים סמנטיים טובים. אנלוגיות עובדות: king-boy+girl → queen.
- **checkpoint זהות ביט-לביט:** 200 צעדים רצוף מול 100+resume+100 → `torch.equal=True`,
  max abs diff 0.0, 4 טנסורי optimizer זהים. כלומר resume מדויק, לא מקורב.
- **זיכרון:** 1,303MB ל-20M טוקנים = 65B/token (רשימת המחרוזות של tokenize דומיננטית).
  `encode` עבר ל-`np.fromiter` (36MB פחות בפועל; 8B/token של מצביעים נחסכו).
  ל-470M טוקנים הקוד הישן היה דורש ~31GB → חייב זרימה בנתחים (שלב 4).
- `train.py`: קונפיג (vocab/dim/window/negatives) נשמר ונבדק בטעינה — dim=50 או window=7
  מפילים ValueError ברור. הגנת דריסה: ריצה שנייה בלי דגל נעצרת עם הודעת step; `--fresh` עוקף.
- מודל ה-epoch המלא שוחזר ל-`outputs/embeddings.npy` אחרי בדיקות ה-20 צעדים.
- **פרק 9 (25 מילים מהמודל המלא):** טבלת שכנים + heatmap 25×25 עם מפרידי קבוצות.
  mom/dad=0.8755 התא הבהיר במטריצה, brother/sister=0.8353 שני, dog/cat=0.741.
  baseline לזוג אקראי 0.284, חציון המטריצה 0.263 → כל התא ערכו בפער מעל הרצפה.
  **תיקון טענה:** happy/sad=0.511 בלבד (מקום 31/300) — הם לא קרובים; המודל למד קוטביות
  (sad→upset/unhappy/helpless, happy→relieved/glad/grateful). האנטונימיה שכן מופיעה:
  big/small=0.636 (מקום 14/300) — "אותה משבצת תחבירית", לא "אותה משמעות".
- **פרק 10 (מפה תלת-ממדית):** PCA (SVD של numpy) על הווקטורים המנורמלים, fitted על כל
  11,381 המילים. PC1=5.7%, PC2=5.0%, PC3=4.4%, שלושתם 15.1% (3 ממדים אקראיים = 3%).
  נוצר `outputs/embedding_map.html` (4.7MB, עצמאי, Plotly) + PNG סטטי עם שמות הקבוצות.
  המפה מציגה מגמות ולא איים: PC3 מסדר places למעלה → feelings למטה, PC1 מפריד size מ-family.
- **כלי Kaggle שנבנו ואומתו:** `evaluate.py` (outputs: 11/16 אנלוגיות, median rank 1.0;
  מודל 200 צעדים: 0/16, rank 835 — המטריקה מבדילה היטב), `grid.py` (רץ מקצה לקצה על שני
  קונפיגי בדיקה, מדלג על קיימים), ו-`kaggle/skipgram_kaggle.ipynb` העצמאי (24 תאים, רץ נקי
  ב-50s עם שני קונפיגי smoke, כולל dim=300).
- **subsampling נמדד:** `subsample=1e-4` משאיר 6,271,535 מ-20,104,405 טוקנים (3.2x פחות).
  לכן "epoch" = 50M זוגות ולא מעבר על הקורפוס — חייב להיאמר בכל השוואה.
- **פרופיל עלות צעד (batch 8192, dim 100):** דגימה 7.2ms, מודל+צעד 70.8ms → GPU אמור לתת
  פי 5–8 (הטבלאות dense, אין sparse Adagrad ב-CUDA). dim=300 על CPU נופל ל-27k זוגות/s.
- טרם אושר שהמשתמש הבין את מנגנון skip-gram ואת משמעות ה-baseline/PC1.

## היכן עצרנו / הצעד הבא

**הפרויקט סגור.** המודל הזוכה הוא `w2_e3` (window 2, dim 100, K=5, בלי subsampling,
3 מעברים = 150M זוגות), מועתק ל-`outputs/`. הגיבויים: `scratch/backups/embeddings_old_48828.npy`
(המודל הקודם) ו-`scratch/runs/` (כל 17 הריצות + `runs.zip`).

תוצאות הגריד (17 ריצות): חלון 2 מנצח (margin 0.295, king/queen 0.836, dog/cat 0.793),
subsampling מפסיד בקורפוס הזה, dim 200/300 מזיק, K=10/15 עולה זמן ולא מוסיף.
הלקח המרכזי: **ה-margin מוטה** — `dim 50` קיבל את ה-margin הגבוה (0.306) ואת התוצאה
הגרועה ביותר (8/16), ולכן הבחירה נעשתה לפי טבלת שכנים ולא לפי argmax של מדד.
ממצא נוסף: אנלוגיות exact-match עיוורות לנרדפות (`lady` במקום `woman`, `bounded` במקום
`jumped`) — 11/16 מול 15/16 הוא ברובו מזל של איזו נרדפת.

המחברת `walkthrough.ipynb` (59 תאים, רצה נקי ב-84s על CPU) נבנתה מחדש מול המודל החדש,
כל מספר בטקסט תואם לפלט. `figures/` מכיל ארבעה גרפים, `README.md` נכתב.

**מה לא עולה ל-git:** `outputs/embeddings.npy` ו-`vocab.txt` (המודל), `runs/`, `scratch/`,
`data/`. כן עולים: המחברת, הקוד, `outputs/embedding_map.html`, `outputs/metrics.json`,
`figures/`, ה-README וה-Kaggle notebook.

הצעד הבא: commit (19 קבצים, 5.6MB) ו-push, אחרי קריאת ה-runbook של האימות ב-
`/root/.opencode/skills/git-workflow/SKILL.md`.
