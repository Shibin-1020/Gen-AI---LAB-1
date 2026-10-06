# Task 2 failure analysis - Denisha Ketan Tank

This review uses the actual records in `outputs/error_review.json`. The labels are Yelp
Polarity labels, where `0` is negative and `1` is positive. The main failure patterns are
context-poor short reviews, negation and sarcasm, and long reviews containing both positive
and negative aspects.

## Confident false positives

| Example | Evidence | Human interpretation | Testable fix |
|---|---|---|---|
| 1 | “Save your $300 initial consultation and go to a Macy's personal shopper--they are free.” | The model focused on “free” and “personal shopper,” although the review advises the reader to avoid the service. | Add advice/negation examples and preserve phrase-level negation. |
| 2 | “Yup” | The one-word review has almost no sentiment context. The confident positive prediction is unsupported by the text. | Treat extremely short reviews as a separate robustness slice and add an abstention analysis. |
| 3 | The mountain-bike review says the trail was formerly excellent but has been damaged by rangers. | “Very best” is positive, but the overall conclusion is negative. The model overweighted an earlier phrase. | Use sequence or attention pooling that emphasizes the final conclusion. |
| 4 | “rice galore!” | The phrase could be positive or sarcastic, but has no surrounding context. | Add ambiguous short-review examples and inspect confidence calibration. |
| 5 | “I love this place... great help... great quality...” | The model predicts positive, while the dataset label is negative. This may be annotation noise or a context mismatch. | Audit neighboring records and test robustness to suspected label noise. |

## Confident false negatives

| Example | Evidence | Human interpretation | Testable fix |
|---|---|---|---|
| 6 | “Not expensive..” | The model likely treated “not” and “expensive” independently and missed the favorable meaning. | Preserve negation scope and add negation-aware features. |
| 7 | “Convenient & OK Mex” | The compressed phrase and abbreviation provide little ordinary sentiment context. | Add abbreviation normalization and test a character/subword tokenizer. |
| 8 | “False advertising! John ISN'T grouchy!” | The model likely relied on “false advertising” without resolving the following denial. | Add contrast and negation examples. |
| 9 | The writer says the guitarist “sucked” and was “not sorry.” | Strong informal negative language was missed in a short colloquial review. | Add slang augmentation and retain emphasis/punctuation features. |
| 10 | “avoided the horrible buffet crowds... boo-yah.” | “Horrible” describes crowds the writer avoided, while “boo-yah” is positive slang. | Add contrastive examples where negative words occur in positive reviews. |

## Near-threshold errors

These examples had probability approximately 0.5, so the model was appropriately uncertain.

| Example | Evidence | Human interpretation | Testable fix |
|---|---|---|---|
| 11 | Massage was nice, but the manicure/pedicure caused problems and the reviewer will not return. | Mixed sentiment across services; the negative conclusion is easy to miss. | Use aspect-level or sentence-level aggregation. |
| 12 | A long diner review praises some fries and service but criticizes the shake, onion rings, burger, and overall experience. | Genuine mixed sentiment with many competing aspects. | Add hierarchical sentence pooling and test long-review truncation. |
| 13 | The soup was delicious, but the main dish was too sweet and poorly seasoned. | One positive item competes with several negative judgments. | Add aspect-aware attention or sentence-level labels. |
| 14 | French review describing food as reheated and too spicy. | The English-trained tokenizer has limited support for non-English text. | Detect language and test multilingual or language-specific tokenization. |
| 15 | Club review mentions an attractive layout but concludes that the wait and crowd ruined the visit. | Positive and negative evidence are mixed, with the conclusion appearing late. | Increase long-context coverage and test conclusion-aware pooling. |

## Slice-specific failures

| Example | Slice | Evidence | Human interpretation | Testable fix |
|---|---|---|---|---|
| 16 | Medium | Service and ambience are positive, but the reviewer rejects the hot dog and plans to visit another place. | The model overweights opening praise and misses the product judgment. | Add more mixed-aspect reviews and conclusion-aware pooling. |
| 17 | Medium | The review is excited about a celebrity sighting and cheap food, but also says the food was poor and the writer unhappy. | Event excitement distracts from the actual food evaluation. | Separate event/entity excitement from review sentiment. |
| 18 | Medium | The review praises spicy chicken, rice, sauce, speed, and price, but the dataset label is negative. | This looks like a possible label-noise or annotation-mismatch case rather than a simple model failure. | Audit the source label and compare neighboring examples before changing the model. |
| 19 | Medium | Service is described as rushed and pushy, but individual employees and the food are praised. | Mixed aspect-level sentiment makes the overall label difficult to infer. | Use aspect-level attention or sentence-level evidence aggregation. |
| 20 | Medium | The reviewer praises a promoter and room layout but criticizes crowding and the value of a table. | Positive entity-specific language competes with a negative overall experience. | Add entity/aspect separation and evaluate the final overall judgment. |

## Overall findings

The errors are systematic rather than random. Short reviews lack context, while long
reviews are difficult when positive and negative aspects coexist. Negation, sarcasm, slang,
multilingual text, and late overall conclusions recur across the examples.

The updated review file contains all 20 required unique records: five confident false
positives, five confident false negatives, five near-threshold errors, and five
slice-specific errors. The cases are drawn from the actual test predictions and are ready
for inclusion in the report.
