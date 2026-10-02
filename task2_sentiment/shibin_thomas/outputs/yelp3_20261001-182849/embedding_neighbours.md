# Nearest neighbours in the learned embeddings (cosine, 20,000 most frequent stems)

Run `yelp3_20261001-182849`, computed from the committed `best_model.pt` files.

| Query (stem) | Baseline (mean-pool) neighbours | BiGRU neighbours |
|---|---|---|
| great | notch, drawback, bonus, trifecta, happili, downsid | alexandra, wander, shenanigan, groov, chester, unadulter |
| terribl | tasteless, worst, overpr, mediocr, overr, horribl | tasteless, garish, uncook, u00a37, overpr, apollo |
| delici | downsid, fantast, perfect, drawback, notch, awesom | sunken, guava, hubcap, newbi, oregon, sniff |
| rude | overpr, downhil, ined, underwhelm, unimpress, worst | yucca, overpr, cannelloni, polic, ownership, checkbook |
| not | clump, unappet, eh, limbo, horribl, guess | limbo, u00f4tel, sushimon, wont, forbid, waistlin |
| recommend | brizza, mike, wintertim, definit, porchetta, eccentr | mike, porchetta, brizza, sightse, eccentr, wintertim |
| never | att, enrag, bite, idea, elara, lubric | enrag, idea, broad, lie, att, wondrous |
| amaz | gem, outstand, downsid, perfect, happili, impecc | velout, gem, resourc, juiciest, samoa, goooood |
| disappoint | meh, bare, underwhelm, bland, tasteless, meager | embarrass, expier, dissatisfi, haggard, meh, underwhelm |
| worst | tasteless, meh, bland, disgust, aw, downhil | tasteless, prey, aw, disgust, craptast, sham |
