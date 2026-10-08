# Evidence index

| Question | Where to look |
|---|---|
| What is in the data, and what leaked? | docs/audit_output.txt, docs/prepare_output.txt |
| How good is the model, and how often is it wrong? | docs/train_output.txt (validation, calibration, slices), docs/backtest_lr_output.txt (8 monthly out-of-sample tests, pooled ROC-AUC 0.776) |
| What is calling vs holding worth in rupees? | docs/economics_output.txt, docs/economics.png |
| What if our assumptions are wrong? | docs/economics_sensitivity_output.txt |
| Are the extreme scores real? | docs/diagnose_output.txt |
| Is predictions.csv valid, and does test drift? | docs/final_model_output.txt |
| Are the reasons shown to staff accurate? | docs/final_output.txt |
| Does the service work? | docs/test_output.txt (8 passed), docs/api_example.txt, docs/screenshot_high_risk.png, docs/screenshot_bad_input.png |
| Does it start on a clean machine? | docs/clean_machine_test_run1_FAILED.txt, docs/clean_machine_test_run2_FAILED.txt, docs/clean_machine_test.txt (run 3), docs/clean_machine_test_run4.txt (final code, passed) |

Key results
- Model: logistic regression, time-based validation only.
- Pooled out-of-sample ROC-AUC 0.776, PR-AUC 0.384, 8 months, all between 0.74 and 0.81.
- Best accuracy any threshold reaches: 89.5%. "Never returned" scores 88.7%.
- Calling orders scoring 12% or more: about +Rs 11,465 per month (bootstrap 95%: Rs 9,724 to 13,259). Worst month +Rs 8,650.
- Holding orders: negative at every threshold up to 40%, about zero at best.
- Weakest assumption: the 35% effect of a call. The call still breaks even down to about 16%.