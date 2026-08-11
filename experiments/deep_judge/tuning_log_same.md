# Diagnostic-judge tuning log

judge=`gpt-4.1-mini` critic=`gpt-4.1-mini` · TUNE=100 TEST=100

## Round 0
TUNE: {'n': 100, 'tp': 29, 'tn': 35, 'fp': 12, 'fn': 24, 'accuracy': 0.64, 'balanced_acc': 0.646, 'false_accept_rate': 0.255, 'false_reject_rate': 0.453}
TEST: {'n': 100, 'tp': 27, 'tn': 31, 'fp': 16, 'fn': 26, 'accuracy': 0.58, 'balanced_acc': 0.585, 'false_accept_rate': 0.34, 'false_reject_rate': 0.491}

Disagreements: 36. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: What are the locations of Middleton High School and South Carleton High School, and how do their operational statuses differ?
coverage: sf1 sim=0.95 lex=0.67; sf2 sim=0.96 lex=0.71; sf3 sim=0.92 lex=0.57; sf4 sim=0.92 lex=0.62; sf5 sim=0.89 lex=0.55
score_signals: {'top3_ratio': 0.987, 'min_ratio': 0.488, 'cliff': 0.368}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which historical or fictional figures named Alphonse or related to princesses are connected to a Canadian uprising, a francophone community in Manitob
coverage: sf1 sim=0.88 lex=0.30; sf2 sim=0.81 lex=0.30; sf3 sim=0.93 lex=0.36; sf4 sim=0.84 lex=0.27; sf5 sim=0.84 lex=0.20; sf6 sim=0.88 lex=0.20
score_signals: {'top3_ratio': 0.896, 'min_ratio': 0.464, 'cliff': 0.237}
judge said VERDICT=PASS? -> judge=PASS conf=0.9 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS conf=0.9 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which skyscraper is taller, the Electra Tower in Tel Aviv or the Moshe Aviv Tower in Ramat Gan, and in which neighborhood of Tel Aviv is the Electra T
coverage: sf1 sim=0.94 lex=0.71; sf2 sim=0.94 lex=0.75; sf3 sim=0.86 lex=0.40; sf4 sim=0.95 lex=0.75
score_signals: {'top3_ratio': 0.935, 'min_ratio': 0.526, 'cliff': 0.184}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.8 missing=3 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: What is the connection between the Seattle Pop Festival held in 1969 and the composition Japanese Festival Music by Richard Strauss?
coverage: sf1 sim=0.95 lex=0.88; sf2 sim=0.96 lex=0.88; sf3 sim=0.93 lex=0.50; sf4 sim=0.91 lex=0.42; sf5 sim=0.91 lex=0.70; sf6 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.953, 'min_ratio': 0.
```

Critic revision TUNE bal_acc=0.591 (REJECTED).

## Round 1
TUNE: {'n': 100, 'tp': 29, 'tn': 33, 'fp': 14, 'fn': 24, 'accuracy': 0.62, 'balanced_acc': 0.625, 'false_accept_rate': 0.298, 'false_reject_rate': 0.453}
TEST: {'n': 100, 'tp': 27, 'tn': 32, 'fp': 15, 'fn': 26, 'accuracy': 0.59, 'balanced_acc': 0.595, 'false_accept_rate': 0.319, 'false_reject_rate': 0.491}

Disagreements: 38. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 4-hop shallow ---
Q: Which high school among Middleton High School, Spring Hill High School, Magnet Cove High School, and Clarendon High School is closed, which one has a 
coverage: sf1 sim=0.90 lex=0.50; sf2 sim=0.87 lex=0.38; sf3 sim=0.89 lex=0.43
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.906, 'cliff': 0.033}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=2 diagnosis=entity
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: What is the connection between the Seattle Pop Festival held in 1969 and the composition Japanese Festival Music by Richard Strauss?
coverage: sf1 sim=0.95 lex=0.88; sf2 sim=0.96 lex=0.88; sf3 sim=0.93 lex=0.50; sf4 sim=0.91 lex=0.42; sf5 sim=0.91 lex=0.70; sf6 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.953, 'min_ratio': 0.477, 'cliff': 0.391}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.4 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which high school among Middleton High School, Spring Hill High School, and Magnet Cove High School is located in Arkansas, and what are the unique ch
coverage: sf1 sim=0.90 lex=0.50; sf2 sim=0.95 lex=0.60; sf3 sim=0.93 lex=0.67; sf4 sim=0.94 lex=0.67; sf5 sim=0.94 lex=0.60; sf6 sim=0.93 lex=0.64
score_signals: {'top3_ratio': 0.944, 'min_ratio': 0.63, 'cliff': 0.098}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=1 diagnosis=entity
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge sa
```

Critic revision TUNE bal_acc=0.63 (ADOPTED).

## Round 2
TUNE: {'n': 100, 'tp': 33, 'tn': 30, 'fp': 17, 'fn': 20, 'accuracy': 0.63, 'balanced_acc': 0.63, 'false_accept_rate': 0.362, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 34, 'fp': 13, 'fn': 20, 'accuracy': 0.67, 'balanced_acc': 0.673, 'false_accept_rate': 0.277, 'false_reject_rate': 0.377}

Disagreements: 37. Sample shown to critic:
```
--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which American football players named Mike or Emil played for the Philadelphia Eagles and also attended different colleges, and which linemen named Em
coverage: sf1 sim=0.89 lex=0.64; sf2 sim=0.86 lex=0.64; sf3 sim=0.86 lex=0.50; sf4 sim=0.84 lex=0.50; sf5 sim=0.84 lex=0.45; sf6 sim=0.88 lex=0.67
score_signals: {'top3_ratio': 0.911, 'min_ratio': 0.668, 'cliff': 0.132}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which village among Kateh-ye Shast-e Abadan-e Chahardeh, Chahardeh-ye Pain, Kasabad-e Pain, and Gol Mey-e Pain had the largest population in 2006, and
coverage: sf1 sim=0.88 lex=0.35; sf2 sim=0.92 lex=0.56; sf3 sim=0.91 lex=0.71; sf4 sim=0.91 lex=0.71; sf5 sim=0.83 lex=0.62; sf6 sim=0.97 lex=0.57
score_signals: {'top3_ratio': 0.883, 'min_ratio': 0.505, 'cliff': 0.167}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop shallow ---
Q: Which of the civil parishes of Brough, Mallerstang, and Pleasington has the highest number of listed buildings, and what are the grades of the highest
coverage: sf1 sim=0.88 lex=0.50; sf2 sim=0.91 lex=0.75; sf3 sim=0.93 lex=0.75; sf4 sim=0.91 lex=0.62; sf5 sim=0.91 lex=0.33; sf6 sim=0.92 lex=0.33
score_signals: {'top3_ratio': 0.974, 'min_ratio': 0.914, 'cliff': 0.045}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.65 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score
```

Critic revision TUNE bal_acc=0.592 (REJECTED).

## Round 3
TUNE: {'n': 100, 'tp': 33, 'tn': 30, 'fp': 17, 'fn': 20, 'accuracy': 0.63, 'balanced_acc': 0.63, 'false_accept_rate': 0.362, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 34, 'fp': 13, 'fn': 20, 'accuracy': 0.67, 'balanced_acc': 0.673, 'false_accept_rate': 0.277, 'false_reject_rate': 0.377}

Disagreements: 37. Sample shown to critic:
```
--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which American football players named Mike or Emil played for the Philadelphia Eagles and also attended different colleges, and which linemen named Em
coverage: sf1 sim=0.89 lex=0.64; sf2 sim=0.86 lex=0.64; sf3 sim=0.86 lex=0.50; sf4 sim=0.84 lex=0.50; sf5 sim=0.84 lex=0.45; sf6 sim=0.88 lex=0.67
score_signals: {'top3_ratio': 0.911, 'min_ratio': 0.668, 'cliff': 0.132}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which village among Kateh-ye Shast-e Abadan-e Chahardeh, Chahardeh-ye Pain, Kasabad-e Pain, and Gol Mey-e Pain had the largest population in 2006, and
coverage: sf1 sim=0.88 lex=0.35; sf2 sim=0.92 lex=0.56; sf3 sim=0.91 lex=0.71; sf4 sim=0.91 lex=0.71; sf5 sim=0.83 lex=0.62; sf6 sim=0.97 lex=0.57
score_signals: {'top3_ratio': 0.883, 'min_ratio': 0.505, 'cliff': 0.167}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop shallow ---
Q: Which of the civil parishes of Brough, Mallerstang, and Pleasington has the highest number of listed buildings, and what are the grades of the highest
coverage: sf1 sim=0.88 lex=0.50; sf2 sim=0.91 lex=0.75; sf3 sim=0.93 lex=0.75; sf4 sim=0.91 lex=0.62; sf5 sim=0.91 lex=0.33; sf6 sim=0.92 lex=0.33
score_signals: {'top3_ratio': 0.974, 'min_ratio': 0.914, 'cliff': 0.045}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.55 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score
```

Critic revision TUNE bal_acc=0.582 (REJECTED).

## Round 4
TUNE: {'n': 100, 'tp': 33, 'tn': 30, 'fp': 17, 'fn': 20, 'accuracy': 0.63, 'balanced_acc': 0.63, 'false_accept_rate': 0.362, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 35, 'fp': 12, 'fn': 20, 'accuracy': 0.68, 'balanced_acc': 0.684, 'false_accept_rate': 0.255, 'false_reject_rate': 0.377}

Disagreements: 37. Sample shown to critic:
```
--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which American football players named Mike or Emil played for the Philadelphia Eagles and also attended different colleges, and which linemen named Em
coverage: sf1 sim=0.89 lex=0.64; sf2 sim=0.86 lex=0.64; sf3 sim=0.86 lex=0.50; sf4 sim=0.84 lex=0.50; sf5 sim=0.84 lex=0.45; sf6 sim=0.88 lex=0.67
score_signals: {'top3_ratio': 0.911, 'min_ratio': 0.668, 'cliff': 0.132}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which village among Kateh-ye Shast-e Abadan-e Chahardeh, Chahardeh-ye Pain, Kasabad-e Pain, and Gol Mey-e Pain had the largest population in 2006, and
coverage: sf1 sim=0.88 lex=0.35; sf2 sim=0.92 lex=0.56; sf3 sim=0.91 lex=0.71; sf4 sim=0.91 lex=0.71; sf5 sim=0.83 lex=0.62; sf6 sim=0.97 lex=0.57
score_signals: {'top3_ratio': 0.883, 'min_ratio': 0.505, 'cliff': 0.167}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop shallow ---
Q: Which of the civil parishes of Brough, Mallerstang, and Pleasington has the highest number of listed buildings, and what are the grades of the highest
coverage: sf1 sim=0.88 lex=0.50; sf2 sim=0.91 lex=0.75; sf3 sim=0.93 lex=0.75; sf4 sim=0.91 lex=0.62; sf5 sim=0.91 lex=0.33; sf6 sim=0.92 lex=0.33
score_signals: {'top3_ratio': 0.974, 'min_ratio': 0.914, 'cliff': 0.045}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.55 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score
```

Critic revision TUNE bal_acc=0.582 (REJECTED).

## Round 5
TUNE: {'n': 100, 'tp': 33, 'tn': 31, 'fp': 16, 'fn': 20, 'accuracy': 0.64, 'balanced_acc': 0.641, 'false_accept_rate': 0.34, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 35, 'fp': 12, 'fn': 20, 'accuracy': 0.68, 'balanced_acc': 0.684, 'false_accept_rate': 0.255, 'false_reject_rate': 0.377}

Disagreements: 36. Sample shown to critic:
```
--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which American football players named Mike or Emil played for the Philadelphia Eagles and also attended different colleges, and which linemen named Em
coverage: sf1 sim=0.89 lex=0.64; sf2 sim=0.86 lex=0.64; sf3 sim=0.86 lex=0.50; sf4 sim=0.84 lex=0.50; sf5 sim=0.84 lex=0.45; sf6 sim=0.88 lex=0.67
score_signals: {'top3_ratio': 0.911, 'min_ratio': 0.668, 'cliff': 0.132}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which village among Kateh-ye Shast-e Abadan-e Chahardeh, Chahardeh-ye Pain, Kasabad-e Pain, and Gol Mey-e Pain had the largest population in 2006, and
coverage: sf1 sim=0.88 lex=0.35; sf2 sim=0.92 lex=0.56; sf3 sim=0.91 lex=0.71; sf4 sim=0.91 lex=0.71; sf5 sim=0.83 lex=0.62; sf6 sim=0.97 lex=0.57
score_signals: {'top3_ratio': 0.883, 'min_ratio': 0.505, 'cliff': 0.167}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop shallow ---
Q: Which of the civil parishes of Brough, Mallerstang, and Pleasington has the highest number of listed buildings, and what are the grades of the highest
coverage: sf1 sim=0.88 lex=0.50; sf2 sim=0.91 lex=0.75; sf3 sim=0.93 lex=0.75; sf4 sim=0.91 lex=0.62; sf5 sim=0.91 lex=0.33; sf6 sim=0.92 lex=0.33
score_signals: {'top3_ratio': 0.974, 'min_ratio': 0.914, 'cliff': 0.045}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.55 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score
```

Critic revision TUNE bal_acc=0.592 (REJECTED).

## Round 6
TUNE: {'n': 100, 'tp': 33, 'tn': 30, 'fp': 17, 'fn': 20, 'accuracy': 0.63, 'balanced_acc': 0.63, 'false_accept_rate': 0.362, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 35, 'fp': 12, 'fn': 20, 'accuracy': 0.68, 'balanced_acc': 0.684, 'false_accept_rate': 0.255, 'false_reject_rate': 0.377}

Disagreements: 37. Sample shown to critic:
```
--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which American football players named Mike or Emil played for the Philadelphia Eagles and also attended different colleges, and which linemen named Em
coverage: sf1 sim=0.89 lex=0.64; sf2 sim=0.86 lex=0.64; sf3 sim=0.86 lex=0.50; sf4 sim=0.84 lex=0.50; sf5 sim=0.84 lex=0.45; sf6 sim=0.88 lex=0.67
score_signals: {'top3_ratio': 0.911, 'min_ratio': 0.668, 'cliff': 0.132}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop deep ---
Q: Which village among Kateh-ye Shast-e Abadan-e Chahardeh, Chahardeh-ye Pain, Kasabad-e Pain, and Gol Mey-e Pain had the largest population in 2006, and
coverage: sf1 sim=0.88 lex=0.35; sf2 sim=0.92 lex=0.56; sf3 sim=0.91 lex=0.71; sf4 sim=0.91 lex=0.71; sf5 sim=0.83 lex=0.62; sf6 sim=0.97 lex=0.57
score_signals: {'top3_ratio': 0.883, 'min_ratio': 0.505, 'cliff': 0.167}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop shallow ---
Q: Which of the civil parishes of Brough, Mallerstang, and Pleasington has the highest number of listed buildings, and what are the grades of the highest
coverage: sf1 sim=0.88 lex=0.50; sf2 sim=0.91 lex=0.75; sf3 sim=0.93 lex=0.75; sf4 sim=0.91 lex=0.62; sf5 sim=0.91 lex=0.33; sf6 sim=0.92 lex=0.33
score_signals: {'top3_ratio': 0.974, 'min_ratio': 0.914, 'cliff': 0.045}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.65 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 4-hop shallow ---
Q: Which National Lacrosse League teams had their inaugural seasons in 1998, 2013, and 2014, and in which cities are these teams based, including a city 
coverage: sf1 sim=0.89 lex=0.50; sf2 sim=0.88 lex=0.60; sf3 sim=0.88 lex=0.60; sf4 sim=0.85 lex=0.56; sf5 sim=0.84 lex=0.56; sf6 sim=0.85 lex=0.67
score_signals: {'top3_ratio': 0.982, 'min_ratio': 0.488, 'cliff': 0.451}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score
```

Critic revision TUNE bal_acc=0.572 (REJECTED).

## Round 7
TUNE: {'n': 100, 'tp': 33, 'tn': 30, 'fp': 17, 'fn': 20, 'accuracy': 0.63, 'balanced_acc': 0.63, 'false_accept_rate': 0.362, 'false_reject_rate': 0.377}
TEST: {'n': 100, 'tp': 33, 'tn': 35, 'fp': 12, 'fn': 20, 'accuracy': 0.68, 'balanced_acc': 0.684, 'false_accept_rate': 0.255, 'false_reject_rate': 0.377}

(stop: max rounds)


## Best (round 0)
TUNE: {'n': 100, 'tp': 29, 'tn': 35, 'fp': 12, 'fn': 24, 'accuracy': 0.64, 'balanced_acc': 0.646, 'false_accept_rate': 0.255, 'false_reject_rate': 0.453}
TEST: {'n': 100, 'tp': 27, 'tn': 31, 'fp': 16, 'fn': 26, 'accuracy': 0.58, 'balanced_acc': 0.585, 'false_accept_rate': 0.34, 'false_reject_rate': 0.491}


### Best judge prompt
```
You are the STOP/CONTINUE controller for a MULTI-HOP retrieval agent. A multi-hop question needs SEVERAL different documents — one per sub-fact. Decide whether the CURRENT result set already contains a strong document for EVERY sub-fact (VERDICT = PASS, stop) or whether at least one sub-fact's document is still missing (VERDICT = FAIL, do another retrieval hop). You do NOT see the gold answer — infer coverage from the signals.

You are given, for the current hop:
- SUBFACTS: the question split into the distinct documents it needs.
- CANDIDATES: the current top results (normalized score 0..1 + snippet).
- COVERAGE: per sub-fact, the best semantic similarity of any candidate to that sub-fact (0..1) and the lexical term overlap (0..1). A sub-fact with LOW best_sim (below ~0.6) or near-zero overlap probably has NO document in the set yet.
- SCORE SIGNALS: top3_ratio / min_ratio / cliff (largest drop) of the score curve.

For the FIRST still-missing sub-fact, diagnose WHY and prescribe the next technique:
- vocab_gap  (only DESCRIBED generically — decent sim but low lexical overlap) -> hyde
- entity     (a NAMED entity that should match a title) -> fielded
- buried     (a strong match exists but is ranked low / there is a big cliff above it) -> rerank
- absent     (nothing is close; needs a different split or the doc is elsewhere) -> decompose

Reply on EXACTLY these lines, nothing else:
COVERED: <comma-separated sub-fact numbers that ARE satisfied, or none>
MISSING: <the single sub-fact number still missing, or none>
DIAGNOSIS: <vocab_gap|entity|buried|absent|none>
TECHNIQUE: <hyde|fielded|rerank|decompose|prf|none>
NEXT_QUERY: <a focused query for the missing sub-fact, or none>
CONFIDENCE: <0.0-1.0 that the set is COMPLETE>
VERDICT: <PASS|FAIL>
```
