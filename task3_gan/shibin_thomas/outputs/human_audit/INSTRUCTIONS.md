# Blinded human audit (30 fixed samples, 2 raters)

Predictions from run `cyclegan_v1_20261001-201801`. Open `images/S01.png` ... `S30.png` (or the overview `contact_sheet.jpg`). Each panel shows the INPUT (left) and the model OUTPUT (right). Do **not** open `audit_key.csv`.

Fill in your own sheet (`rater1.csv` or `rater2.csv`) independently -- do not discuss scores before both are done. Integers 1-5:

| Criterion | 1 | 5 |
|---|---|---|
| style | output does not look like the target domain | indistinguishable from a real Monet / real photo |
| content | scene of the input lost | layout, objects and shapes fully preserved |
| artifacts | severe checkerboard / blotches / smearing / noise | clean |

Then run `python task3_gan/shibin_thomas/src/human_audit.py score`.
