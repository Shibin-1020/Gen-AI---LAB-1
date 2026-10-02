# Task 2.2.4 — Manual review of 20 errors · Shibin Thomas

**Model reviewed:** `exp2_bigru_attn`, the best of my three models by **validation** macro-F1 (0.9522); run
`yelp3_20261001-182849`.
**Data:** official Yelp polarity test split (n = 38,000), threshold 0.5. The model makes **1,681 errors**
(4.42%): 971 false positives and 710 false negatives.
**Selection** (`src/error_analysis.py`, then reviewed by hand):

| Group | Count | Rule |
|---|---|---|
| Confident false positives | 5 | true negative, highest P(positive) |
| Confident false negatives | 5 | true positive, lowest P(positive) |
| Near-threshold errors | 5 | misclassified, P(positive) closest to 0.5 |
| Slice-specific failures | 5 | most confident errors in the slice with the highest error rate |

The slice with the highest error rate is **short reviews (≤ 50 words): 5.01%**. The other slices are:
has contrast 4.87%, has negation 4.48%, long 4.45%, no negation & no contrast 4.16%, medium 4.09%,
exclamation-heavy 2.84%.

I read every review and assigned the error type myself. The script's keyword-based suggestion was wrong for
several of them: for example, #2 and #17 are positive texts with a negative label, not "mixed sentiment".
Labels in Yelp polarity come from star ratings (1–2 stars = negative, 4–5 stars = positive), not from the
text, so a mismatch between text and stars is a real error source.

| # | Group | Test idx | True | P(pos) | Error type (manual) | What happened |
|---|---|---|---|---|---|---|
| 1 | Confident FP | 23815 | neg | 1.000 | Mixed sentiment (aspect-level) | long praise of the bakery ("tops", "really well made", "huge brownies") dominates a review that is negative overall |
| 2 | Confident FP | 29330 | neg | 1.000 | **Label noise** | "Wow love the place … very clean and new! Great place … worth a try!" is unambiguously positive; the star rating does not match the text |
| 3 | Confident FP | 408 | neg | 1.000 | Faint praise / comparative | "Maharani was fine enough … Copper is definitely still my place": lukewarm verdict wrapped in positive details ("food came very quickly") |
| 4 | Confident FP | 22663 | neg | 1.000 | Mixed + implicit verdict | "very tasty", "fantastic", "great" outnumber "average", "over cooked"; the real verdict is implicit ("more for show than content", recommends competitors) |
| 5 | Confident FP | 8426 | neg | 1.000 | Mixed sentiment (aspect-level) | the brunch is described as "reasonably priced" and "cheap", the mimosas "comped"; the negative judgement comes later in a 231-token review |
| 6 | Confident FN | 22807 | pos | 0.000 | **Label noise (edited review)** | "EDIT: … Horrible service … rethinking that": the text was rewritten after a positive rating; the model is right about the text |
| 7 | Confident FN | 22451 | pos | 0.001 | Label noise / outside-text rating | "The food is crap … horrible … worst nachos": the body is strongly negative although the review mentions 4 stars; nothing in the visible text supports a positive label |
| 8 | Confident FN | 27334 | pos | 0.001 | Mixed sentiment | "brilliance of your product" vs a long complaint about "dis-interested" staff; the complaint gets more words and wins |
| 9 | Confident FN | 21215 | pos | 0.001 | **Sentiment-target confusion** | a story about other customers' children misbehaving; the reviewer sides with the bartender (positive about the bar), but the words describe a negative event |
| 10 | Confident FN | 11808 | pos | 0.001 | Mixed (complaints first) | "This place is TINY! Narrow walk ways … I do hope that all of the animals … are ok": criticism comes first, the positive verdict later |
| 11 | Near-threshold | 20397 | pos | 0.499 | Truncation of a long narrative | 361 tokens > max_len 256: head+tail truncation drops the middle of a story-style review with little explicit opinion |
| 12 | Near-threshold | 18529 | pos | 0.498 | Mixed (item-by-item) | price-by-item sushi review ("average quality", "not once has it had a fishy smell"); mixed judgements per dish |
| 13 | Near-threshold | 28403 | pos | 0.498 | Mixed + humour | "alcohol selection sucks … dreamcrushed my cocktail dreams. However, the burger was nice": the positive turn after "however" is short and the complaint is long and humorous |
| 14 | Near-threshold | 29624 | neg | 0.502 | Implicit negativity / domain shift | a **concert** review: criticism via "overly sentimental", "too many covers", "weird duet", with positive words ("blessed", "elite performer") and no restaurant vocabulary |
| 15 | Near-threshold | 14124 | pos | 0.497 | **Idiom misread** | "the cookie's are to die for", plus "to my door in 15 minutes" and "good deals"; "die" pulls toward negative, and an idiom cannot be learned compositionally from stems |
| 16 | Slice (short) | 20439 | neg | 0.999 | Faint praise | "Standard take out and it's cheap. Service is fast and friendly. I go here when I'm too lazy …": positive words, but the place is a fallback (2-star "meh") |
| 17 | Slice (short) | 12480 | neg | 0.999 | **Label noise** | "omelette that was good … bacon was great. Our server was awesome!": mostly positive text with a negative label |
| 18 | Slice (short) | 26683 | pos | 0.002 | **Complex negation + star mention** | "despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review": "not", "disrespect" and "2 star" all point negative, but the sentence means "better than 2 stars" |
| 19 | Slice (short) | 33573 | pos | 0.002 | **Sarcasm / irony** | "For being a DUMP … Flys, stink, garbage, dirt … Keepin it real dumpy!": affectionate irony about a dive bar, literally all negative words |
| 20 | Slice (short) | 20058 | neg | 0.998 | Label noise / faint praise | "Pretty decent food. Great service. Divey.": reads positive; at most a lukewarm 2-star text |

## The 20 reviews

**1. Confident false positive** - true negative, P(pos) = 1.000, 106 tokens after preprocessing - *Mixed sentiment (aspect-level praise in a negative review)*

> This is the neighborhood Foodland that has the bare necessities needed to sustain a pantry or for when the next snowstorm of the century is a day away and you only have minutes to get TP, bread and milk. The bakery here is tops, small selection, but really well made specialities. Huge brownies, iced and moist. They are easily 5 inch squares, topped with things like Oreo cookies, nuts, sprinkles, a ...

**2. Confident false positive** - true negative, P(pos) = 1.000, 19 tokens after preprocessing - *Label noise (text clearly positive)*

> Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012

**3. Confident false positive** - true negative, P(pos) = 1.000, 62 tokens after preprocessing - *Faint praise / comparative review*

> Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago. Copper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly, which is rare. Usually indian food, good indian good, takes at least 30-45 minutes. We got our order in like ...

**4. Confident false positive** - true negative, P(pos) = 1.000, 97 tokens after preprocessing - *Mixed sentiment + implicit verdict (recommends competitors)*

> About average so far as steakhouses go. My rib eye was very tasty but a little over cooked. I didn't complain because the flavor was still fantastic. I thought the process were quite out of order. I've had better for sell. Over all I'd say they are more for show than content. If you want to have a great high end steak I'd recommend Ruth Chris, LG's, Flemings, or Morton's. If you want to show off h ...

**5. Confident false positive** - true negative, P(pos) = 1.000, 231 tokens after preprocessing - *Mixed sentiment (aspect-level praise in a negative review)*

> Saturday / Sunday AYCE brunch In true las vegas fashion, you get a flat rate to eat your heart out. For Strip food, main menu items seem reasonably priced and the brunch is cheap at $29.99. There are a few different flavors to choose from for the All-You-Can-Drink Bottomless mimosas. They're $5 per person; but on multiple occasions, the $5 was comped for me and my dining partner. Maybe its a local ...

**6. Confident false negative** - true positive, P(pos) = 0.000, 49 tokens after preprocessing - *Label noise (review edited after the rating)*

> EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.' Sorry Bella Nott ...

**7. Confident false negative** - true positive, P(pos) = 0.001, 113 tokens after preprocessing - *Label noise / rating driven by something outside the text*

> The food is crap. I'm not trying to be mean, but it really is horrible. I'd rather eat one of those Tornado things from Circle K for dinner. Also, I don't appreciate the waitress telling me everything is great when everything is absolutely not great. What kind of disgusting excuse for food must she live off of if the nachos get her stamp of approval? They were probably the worst nachos I've ever h ...

**8. Confident false negative** - true positive, P(pos) = 0.001, 75 tokens after preprocessing - *Mixed sentiment (great product, bad staff)*

> A note to the owners of Cupcrazed: Your staff are overriding the brilliance of your product. For a company that has something great to brag about as this place does, the employees seem dis-interested and completely unmotivated. Having arrived two hours before closing, and similar to other reports, the chairs were upside down on the tables sending a clear message that closing can't come fast enough ...

**9. Confident false negative** - true positive, P(pos) = 0.001, 84 tokens after preprocessing - *Sentiment-target confusion (negative story, positive about the business)*

> Last night several parents came in with over 15 children to celebrate their 9 year olds 4th grade graduation at 9:15 pm. The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue if anything were to happen to them on their premise. The children were running in and out of the bar while the parents continued to drink upstairs claim ...

**10. Confident false negative** - true positive, P(pos) = 0.001, 108 tokens after preprocessing - *Mixed sentiment (complaints first, praise later)*

> It was Anniversary time! But we didn't' want to spend a ton of money on food or booze. Plus, were weren't interested in going to a show at the time. So, what to do? Aquarium! Don't get me wrong: This place is TINY! Narrow walk ways and close quarters in general. Even the exhibits are small. I do hope that all of the animals in there are ok. They do seem treated well but I wonder if they would be h ...

**11. Near-threshold error** - true positive, P(pos) = 0.499, 361 tokens after preprocessing - *Truncation of a long narrative review*

> Went straight from work to check out Mekong Plaza last night after chatting with coworker ""JJ"" about the good deals she found there. She was able to find her fave brand of rice, grapes, garlic, and other items. She was lamenting that she only had twenty minutes to look around; she therefore allowed herself to buy those items she knew cost less here than at Lee Lee. After she'd listed off what sh ...

**12. Near-threshold error** - true positive, P(pos) = 0.498, 175 tokens after preprocessing - *Mixed sentiment (item-by-item review)*

> As with all my reviews, my review of Teharu will compare the food relative to price. Who cares about service right? Go to China and tell me about service. Food is first and foremost. Nigiri. $1 - Salmon: average quality for being priced at $1, wait how can I say that since I've never had $1 salmon nigiri???? its $1 and throughout my 4 years of going here, not once has it had a fishy smell. No fish ...

**13. Near-threshold error** - true positive, P(pos) = 0.498, 33 tokens after preprocessing - *Mixed sentiment + humour*

> Their alcohol selection sucks, as in they ran out of everything they said they had. I had to have a Sapphire tonic...SAPPHIRE, PEOPLE!!! The waiter said she had Hendricks and then dreamcrushed my cocktail dreams. My other friend wanted a beer and 8 of the beers he wanted they were ""out of"". However, the burger was nice, very juicy and rare. Plus, the more you drink the better everything tastes,  ...

**14. Near-threshold error** - true negative, P(pos) = 0.502, 77 tokens after preprocessing - *Implicit negativity / domain shift (concert, not restaurant)*

> Overly sentimental, non stop references to her kids, how blessed she is, and her husband.... too many covers from elite performers like Stevie Wonder.... Billy Joel...Tina Turner. Why is she not just playing her music.. as an elite performer herself? A weird duet sung between herself and a computer generation of herself... and then back to back with another one...a computer generated Stevie Wonder ...

**15. Near-threshold error** - true positive, P(pos) = 0.497, 22 tokens after preprocessing - *Idiom misread ("to die for")*

> Wow,I placed an order here last night and the pizza was to my door in 15 minutes! I got a pepperoni pizza with a chocolate chip cookie and the cookie's are to die for. Also if you order online there are a lot of good deals going on.

**16. Slice-specific failure (short (<=50 words))** - true negative, P(pos) = 0.999, 16 tokens after preprocessing - *Faint praise (a fallback option)*

> Standard take out and it's cheap. Service is fast and friendly. I go here when I'm too lazy to walk to Zaw's or pressed for time (I've never had to wait longer than 15 minutes).

**17. Slice-specific failure (short (<=50 words))** - true negative, P(pos) = 0.999, 12 tokens after preprocessing - *Label noise (text mostly positive)*

> my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!

**18. Slice-specific failure (short (<=50 words))** - true positive, P(pos) = 0.002, 16 tokens after preprocessing - *Complex negation + star mention*

> I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review.

**19. Slice-specific failure (short (<=50 words))** - true positive, P(pos) = 0.002, 14 tokens after preprocessing - *Sarcasm / irony*

> For being a DUMP, should expect much more. Flys, stink, garbage, dirt, and everything that comes with. Salt River... Keepin it real dumpy!

**20. Slice-specific failure (short (<=50 words))** - true negative, P(pos) = 0.998, 8 tokens after preprocessing - *Label noise / faint praise ("pretty decent")*

> Meat and two place. Pretty decent food. Great service. Divey.

## Error types found

| Error type | Count | Reviews | Can the model fix it? |
|---|---|---|---|
| Mixed sentiment (aspect-level / complaints vs praise) | 7 | 1, 4, 5, 8, 10, 12, 13 | partly: which clause carries the verdict is learnable |
| Label noise (text contradicts the star label, incl. edited review) | 5 | 2, 6, 7, 17, 20 | no: the model's prediction matches the text |
| Faint praise / implicit or comparative negativity | 3 | 3, 14, 16 | partly: needs world knowledge ("fine enough", "fallback") |
| Complex negation + explicit rating | 1 | 18 | yes, in principle (compositional) |
| Sentiment-target confusion | 1 | 9 | hard: needs to know who the sentiment is about |
| Truncation of a long review | 1 | 11 | yes: longer context |
| Idiom | 1 | 15 | partly: more data / sub-word or phrase features |
| Sarcasm / irony | 1 | 19 | hard without pragmatic context |

**Observations**
* **A quarter of the reviewed errors are not model errors.** Five of the 20 (#2, #6, #7, #17, #20) have
  texts that contradict their star-derived label. This matches the global picture: **932 test reviews are
  misclassified by all three of my models**, and 544 of the BiGRU's 1,681 errors are made with more than 90%
  confidence. A noisy core like that caps the achievable accuracy below 100%.
* **Confident errors are mostly "the words point the wrong way".** Every confident false positive or
  negative has many sentiment words on the opposite side of the label: praise in a negative review (#1, #4,
  #5) or a negative story in a positive review (#9, #10). The BiGRU is order-aware, but it still
  weights the amount of sentiment-bearing text heavily.
* **Near-threshold errors are genuinely ambiguous.** Mixed item-by-item reviews (#12, #13), an off-domain
  concert review (#14), an idiom (#15) and a truncated narrative (#11). Here the model's uncertainty
  (P ≈ 0.5) is the right behaviour.
* **Short reviews are the hardest slice** because there is no redundancy. One phrase decides the
  label, so faint praise (#16), sarcasm (#19), complex negation (#18) and noisy labels (#17, #20) are
  decisive.

## Proposed testable fix

The most frequent model-fixable error is **mixed sentiment**: 7 of 20, plus #18, where the clause after
"despite / but" carries the verdict. **Fix: contrast-aware clause marking.** In preprocessing, prefix every
token that follows the last contrast word ("but", "however", "although", "though", "despite", "yet") with a
marker, e.g. `but` → `POST_good`. Alternatively, feed the post-contrast clause as a second segment. Either
way, the model learns that what comes after the turn is the verdict.

**How to test it:**
1. Retrain `exp2_bigru_attn` with only this change: same data split, seed and hyperparameters.
2. Compare the **has-contrast slice** (currently macro-F1 0.9509, error rate 4.87%) and overall macro-F1.
3. Run a paired McNemar test against the current BiGRU on the same 38,000 test reviews.
4. The fix is supported if the has-contrast error rate drops significantly while the no-contrast slice does
   not get worse.

**Secondary fixes, each testable on its own slice:**
* negation marking (`not good` → `NOT_good`) for #18-type errors, measured on the has-negation slice;
* a larger `max_len` (512) for long reviews (#11), measured on the long-review slice;
* estimating label noise: relabel a random sample of 200 confident errors by hand, to quantify the ceiling.
