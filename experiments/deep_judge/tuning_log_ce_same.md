# Diagnostic-judge tuning log

judge=`gpt-4.1-mini` critic=`gpt-4.1-mini` · TUNE=100 TEST=100

## Round 0
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.723 (REJECTED).

## Round 1
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.76 (REJECTED).

## Round 2
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.3 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.658 (REJECTED).

## Round 3
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.706 (REJECTED).

## Round 4
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.713 (REJECTED).

## Round 5
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.7 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.725 (REJECTED).

## Round 6
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

Disagreements: 24. Sample shown to critic:
```
--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 2-hop shallow ---
Q: Which event involved a suicide car bombing in a Shia Muslim district while a French president was present, and which film directed by Angelina Jolie w
coverage: sf1 sim=0.92 lex=0.56; sf2 sim=0.72 lex=0.00; sf3 sim=0.89 lex=0.36
score_signals: {'top3_ratio': 0.951, 'min_ratio': 0.492, 'cliff': 0.394}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=2 diagnosis=absent
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: Which radio station licensed to Charlottesville, Virginia, serves Albemarle County with a News/Talk format and who owns it, and how does its format an
coverage: sf1 sim=0.92 lex=0.69; sf2 sim=0.89 lex=0.64; sf3 sim=0.89 lex=0.50; sf4 sim=0.87 lex=0.38; sf5 sim=0.84 lex=0.50; sf6 sim=0.84 lex=0.45
score_signals: {'top3_ratio': 0.873, 'min_ratio': 0.39, 'cliff': 0.316}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=5 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: Which Dorothea, born in the 18th century, was known for literary translation and which one held a significant governmental position in Denmark?
coverage: sf1 sim=0.86 lex=0.36; sf2 sim=0.91 lex=0.31; sf3 sim=0.86 lex=0.29; sf4 sim=0.90 lex=0.25
score_signals: {'top3_ratio': 0.821, 'min_ratio': 0.515, 'cliff': 0.416}
judge said VERDICT=PASS? -> judge=PASS conf=1.0 missing=none diagnosis=none
ORACLE TRUTH: INCOMPLETE (FAIL)

--- FALSE_REJECT (judge FAIL, truly COMPLETE) | 3-hop deep ---
Q: What is the connection between the title of Queen's fourth studio album, the anniversary celebrated by the box set that includes this album, and the f
coverage: sf1 sim=0.88 lex=0.71; sf2 sim=0.89 lex=0.42; sf3 sim=0.90 lex=0.88; sf4 sim=0.87 lex=0.35
score_signals: {'top3_ratio': 0.825, 'min_ratio': 0.434, 'cliff': 0.257}
judge said VERDICT=FAIL? -> judge=FAIL conf=0.6 missing=4 diagnosis=vocab_gap
ORACLE TRUTH: COMPLETE (PASS)

--- FALSE_ACCEPT (judge PASS, truly INCOMPLETE) | 3-hop shallow ---
Q: What are the differences in the thematic focus and production background between the 2006 films titled 'Goya's Ghosts' and 'Ghosts', and how does the 
coverage: sf1 sim=0.93 lex=0.56; sf2 sim=0.89 lex=0.50; sf3 sim=0.94 lex=0.56; sf4 sim=0.90 lex=0.50; sf5 sim=0.90 lex=0.40; sf6 sim=0.92 lex=0.40
score_signals: {'top3_ratio': 0.89, 'min_ratio': 0.492, 'cliff': 0.329}
judge said VERDICT=PASS? -> judge=PASS 
```

Critic revision TUNE bal_acc=0.771 (ADOPTED).

## Round 7
TUNE: {'n': 100, 'tp': 40, 'tn': 37, 'fp': 10, 'fn': 13, 'accuracy': 0.77, 'balanced_acc': 0.771, 'false_accept_rate': 0.213, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 35, 'fp': 12, 'fn': 16, 'accuracy': 0.72, 'balanced_acc': 0.721, 'false_accept_rate': 0.255, 'false_reject_rate': 0.302}

(stop: max rounds)


## Best (round 7)
TUNE: {'n': 100, 'tp': 40, 'tn': 37, 'fp': 10, 'fn': 13, 'accuracy': 0.77, 'balanced_acc': 0.771, 'false_accept_rate': 0.213, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 35, 'fp': 12, 'fn': 16, 'accuracy': 0.72, 'balanced_acc': 0.721, 'false_accept_rate': 0.255, 'false_reject_rate': 0.302}


### Best judge prompt
```
You are the STOP/CONTINUE controller for a MULTI-HOP retrieval agent. A multi-hop question requires SEVERAL distinct documents—one per sub-fact. Your task is to decide whether the CURRENT result set already contains a sufficiently strong document for EVERY sub-fact (VERDICT = PASS, stop) or whether at least one sub-fact's document is still missing (VERDICT = FAIL, continue retrieval). You do NOT see the gold answer—use the provided signals to infer coverage.

Input per current hop:
- SUBFACTS: the question decomposed into distinct sub-facts.
- CANDIDATES: current top retrieval results with normalized scores (0..1) and snippets.
- COVERAGE: for each sub-fact, three signals about its BEST candidate:
    * ce = CROSS-ENCODER relevance score (PRIMARY signal). Calibrated as follows:
        - ce > 0.1: strong evidence the sub-fact is covered.
        - ce between -0.5 and 0.1: borderline coverage; treat cautiously.
        - ce < -1.5: strong evidence the sub-fact is missing.
        - ce between -1.5 and -0.5: weak negative, consider other signals.
    * sim = bi-encoder cosine similarity (0..1). This is a weak, saturated signal; use only to break ties or support borderline cases.
    * lex = lexical overlap (0..1). Use as secondary evidence, especially to detect vocabulary gaps.
- SCORE SIGNALS: top3_ratio, min_ratio, cliff (largest score drop) of the candidate score curve.

Decision guidelines:
- FAIL if ANY sub-fact has ce < -1.5 (very negative), indicating no candidate answers it.
- PASS only if ALL sub-facts have ce > 0.1 (strong positive).
- For sub-facts with borderline ce (-1.5 to 0.1), consider lex and sim:
    * If lex < 0.2 and ce < 0, likely vocab_gap → treat as missing.
    * If lex ≥ 0.2 or sim ≥ 0.85, consider sub-fact covered despite borderline ce.
- If multiple sub-facts are borderline or weak, allow up to one sub-fact with borderline coverage before FAILing.
- Use score signals to detect buried documents:
    * If a sub-fact’s best candidate has ce > 0 but is ranked below a large cliff (>0.3) or top3_ratio < 0.85, diagnose buried.
- When deciding PASS, require confidence ≥ 0.85; otherwise, FAIL with appropriate diagnosis.

For the FIRST missing or borderline sub-fact (lowest ce or lex), diagnose WHY and prescribe the next technique:
- vocab_gap: some relevance but low lexical overlap → hyde
- entity: presence of a named entity that should match a title → fielded
- buried: strong match exists but ranked low or large score cliff → rerank
- absent: ce very negative for all candidates → decompose

Output EXACTLY these lines, nothing else:
COVERED: <comma-separated sub-fact numbers that are confidently satisfied, or none>
MISSING: <the single sub-fact number still missing or borderline, or none>
DIAGNOSIS: <vocab_gap|entity|buried|absent|none>
TECHNIQUE: <hyde|fielded|rerank|decompose|prf|none>
NEXT_QUERY: <a focused query for the missing sub-fact, or none>
CONFIDENCE: <0.0-1.0 confidence the set is COMPLETE>
VERDICT: <PASS|FAIL>
```
