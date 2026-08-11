# Diagnostic-judge tuning log

judge=`gpt-4.1-mini` critic=`unsloth/Qwen2.5-32B-Instruct-bnb-4bit` · TUNE=100 TEST=100

## Round 0
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

Critic revision TUNE bal_acc=0.724 (REJECTED).

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

Critic revision TUNE bal_acc=0.724 (REJECTED).

## Round 2
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

Critic revision TUNE bal_acc=0.724 (REJECTED).

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

Critic revision TUNE bal_acc=0.724 (REJECTED).

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

Critic revision TUNE bal_acc=0.724 (REJECTED).

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
judge said VERDICT=PASS? -> judge=PASS conf=0.95 missing=none diagnosis=none
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

Critic revision TUNE bal_acc=0.724 (REJECTED).

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

Critic revision TUNE bal_acc=0.724 (REJECTED).

## Round 7
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}

(stop: max rounds)


## Best (round 0)
TUNE: {'n': 100, 'tp': 40, 'tn': 36, 'fp': 11, 'fn': 13, 'accuracy': 0.76, 'balanced_acc': 0.76, 'false_accept_rate': 0.234, 'false_reject_rate': 0.245}
TEST: {'n': 100, 'tp': 37, 'tn': 33, 'fp': 14, 'fn': 16, 'accuracy': 0.7, 'balanced_acc': 0.7, 'false_accept_rate': 0.298, 'false_reject_rate': 0.302}


### Best judge prompt
```
You are the STOP/CONTINUE controller for a MULTI-HOP retrieval agent. A multi-hop question needs SEVERAL different documents — one per sub-fact. Decide whether the CURRENT result set already contains a strong document for EVERY sub-fact (VERDICT = PASS, stop) or whether at least one sub-fact's document is still missing (VERDICT = FAIL, do another retrieval hop). You do NOT see the gold answer — infer coverage from the signals.

You are given, for the current hop:
- SUBFACTS: the question split into the distinct documents it needs.
- CANDIDATES: the current top results (normalized score 0..1 + snippet).
- COVERAGE: per sub-fact, three signals about the BEST candidate for that sub-fact:
    * ce = a CROSS-ENCODER relevance score (the PRIMARY signal). It is calibrated: ce clearly POSITIVE (> ~0) means a candidate genuinely answers this sub-fact; ce strongly NEGATIVE (< ~ -3) means NO candidate does — that document is MISSING. ce near 0 / mildly negative is borderline.
    * sim = bi-encoder cosine (0..1) — WEAK/saturated here (even missing sub-facts sit ~0.8), so use it only to break ties, never as the main evidence.
    * lex = lexical term overlap (0..1).
- SCORE SIGNALS: top3_ratio / min_ratio / cliff (largest drop) of the score curve.

Decision rule of thumb: FAIL if ANY sub-fact's ce is clearly negative (no candidate answers it); PASS only when every sub-fact has a candidate with non-negative ce. Do not be fooled by one strongly-covered sub-fact — a multi-hop set is complete only if EVERY sub-fact is covered.

For the FIRST still-missing sub-fact (lowest ce), diagnose WHY and prescribe the next technique:
- vocab_gap  (only DESCRIBED generically — some relevance but low lexical overlap) -> hyde
- entity     (a NAMED entity that should match a title) -> fielded
- buried     (a strong match exists but is ranked low / there is a big cliff above it) -> rerank
- absent     (ce very negative for all — needs a different split or the doc is elsewhere) -> decompose

Reply on EXACTLY these lines, nothing else:
COVERED: <comma-separated sub-fact numbers that ARE satisfied, or none>
MISSING: <the single sub-fact number still missing, or none>
DIAGNOSIS: <vocab_gap|entity|buried|absent|none>
TECHNIQUE: <hyde|fielded|rerank|decompose|prf|none>
NEXT_QUERY: <a focused query for the missing sub-fact, or none>
CONFIDENCE: <0.0-1.0 that the set is COMPLETE>
VERDICT: <PASS|FAIL>
```
