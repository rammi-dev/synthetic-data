uv run papermill sample_visualiser.ipynb sample_visualiser_executed.ipynb \
  -p duration_days 365 \
  -p decoy_freq_per_day 2.0 \
  -p run_short_scenarios true \
  -p run_long_series true \
  -p seed 42 \
  --progress-bar 2>&1