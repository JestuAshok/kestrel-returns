## Github Repo URL
https://github.com/JestuAshok/kestrel-returns
Public as instructed by the organisers. No raw customer data is in the repo or its
history (data/ is git-ignored). It holds predictions, aggregate outputs, a trained model
(weights only), and a few sample customer and order IDs in logs and demos.

## What did you build, and what business decision does it support? State the number and the rupees.
A returns-risk scorer, with a web screen and API. The decision is to phone customers on
orders scoring 12% or more before dispatch (about 30% of orders, about 210 a month)
instead of holding them. At 700 orders a month the net gain is about Rs 11,000 to 11,500
(interval Rs 9,700 to 13,300). A return costs Rs 1,150, a call costs Rs 45, and a call
prevents 35% of returns per the policy pilot.

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.
ROC-AUC, because the file is a ranking score. I expect about 0.77, range 0.73 to 0.81.
Estimated with eight monthly out-of-sample tests (Nov 2025 to Jun 2026), each trained
only on earlier months. Pooled ROC-AUC 0.776; months ranged 0.739 to 0.805. PR-AUC about
0.38 (range 0.30 to 0.46). On the test set, 28.1% of orders score 0.12 or more, against
30.3% in the backtest, so little drift.

## How do you know it works?
Time-based split only: fit before Apr 2026, validation Apr to Jun 2026 (ROC-AUC 0.788,
bootstrap 95% 0.755 to 0.816), plus the rolling backtest. Accuracy is 89.5% against 88.7%
for "never returns", so 95% is not reachable. The top 10% of scores had 42% actual
returns. Precision at the 12% cut-off is about 25%, so about 3 in 4 calls go to orders
that would not have returned. Weak spots: Robot Vacuum (ROC-AUC 0.70), and the model
under-predicts Shield members (16% average score vs 20.5% actual). Calibration is good
except the top decile (predicted 36%, actual 42%).

## Did you change, narrow, or push back on the client's ask?
Yes. I pushed back on "flag and hold, 95% accuracy". Accuracy is a poor measure at an 11%
return rate, and holding loses money because about 12% of held customers cancel. I
recommended calling instead (the spring pilot Meenal mentioned) and kept Shield members to
calls only. I used Finance's Rs 1,150, not Rs 600. I decided this during analysis, after
seeing the rupee tables.

## What is wrong with what you are handing us, or with the data we handed you?
- The 35% call effect comes from a pilot I cannot verify. Below about 16% the saving is zero.
- The 15% margin in the hold test is my assumption; the pack gives none.
- The 12% threshold was picked on the same out-of-sample data it is scored on, though the curve is flat from 8% to 15%.
- Customer history columns are not a clean running count: about half of repeat customers have a later order with fewer prior orders. I kept them because they predict consistently, but only partly trust them.
- 19% of rows have a signup date after the order date, so I dropped those features.
- Shield status comes from one snapshot, so it may partly reflect past returns.
- The 3+ prior returns group has only 124 orders.
- Pickup and service-event columns leak the outcome and were dropped. October order values were 100 times too large; I divided them by 100. 651 duplicate orders were removed.
- Reasons show observed past rates, not causes. Scores above 60% display as "60% or more".
- Tested on Python 3.14 on Windows only.
- Input validation was weak at first (a garbage pincode scored); I found it by hand, fixed it and added tests.
- Clean-machine tests found two README bugs (missing data/ folder, a hidden file dependency); both fixed and re-tested.
- An agent script mislabelled the test period "Jul to Aug"; it is Jul to Sep.
- Logs contain a few sample order and customer IDs and a local Windows path.

## What does one prediction cost, and what would a month cost at about 700 orders?
Rs 0. The model runs locally with no API or paid call. 700 x Rs 0 = Rs 0. The confirmation
calls cost Rs 45 each (about Rs 9,500 a month), already counted in the savings above.

## What did you deliberately leave out, and why?
Hold policies beyond testing them, because they lose money. Delivery-note text, because it
showed no difference between returned and non-returned orders. City and state in reasons
(noise). Hyperparameter search and gradient boosting, since the simpler model won. A paid
LLM, since nothing needed one. Call scheduling and capacity planning.

## Anything you built or found that nobody asked for?
The call-versus-hold analysis, sensitivity tests (call effect, completion rate, return
cost, month by month), the leak and unit-error findings, a contradiction filter on the
reasons, input validation with plain-English errors, and four clean-machine test runs
with logs.

## What did you use AI for?
Tools: Antigravity with Gemini (listed as "2.8 medium"), free; Claude chat, free. Total
cost Rs 0.
Helped: boilerplate, the API and page, spotting leakage and the October error.
Misled me: the agent wrote "plain-English interpretations" before seeing results, with
wrong odds ratios (COD 2.45 vs actual 1.90). It invented causes ("buyer remorse", "guest
checkouts"). Its first ablation ignored my spec. It said files existed that it had not
created. It recommended options that would have chosen cost assumptions for me, which I
declined. Claude chat wrongly called the 90-98% scores "not credible" until the data
showed customers with 4+ prior returns come back 76% of the time; it also skipped git
init and missed the pincode validation gap.
Discarded: gradient boosting, signup features, city reasons, the hardcoded explanation
block, the first reasons version.
Video: [YOUR DRIVE LINK]

## Your Public Google Drive Link
[https://drive.google.com/file/d/1oOu_zxjhZrCwKHnBvPh2i3cxfgKbU4Lw/view?usp=sharing]

## Someone picks this up on Monday and you are unreachable. The three things they need to know.
1. The decision is to call at a score of 0.12 or more, never hold. The rupee figure depends on the 35% call effect, so the pilot should measure it first.
2. To rebuild, follow the README (data is not in the repo). models/model.joblib and predictions.csv are in it; retrain with python src/final_model.py. Evidence is indexed in docs/EVIDENCE.md.
3. Do not add pickup or service-event columns (they leak). Treat customer history, Shield and signup dates as unreliable until Tanmay confirms how they are produced.
