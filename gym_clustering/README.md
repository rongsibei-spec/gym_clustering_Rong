# Gym member clustering

Use Python 3.11 or newer. From this directory, run:

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The course data describe 330 gym members over the past three months. The script uses a random 70/30 split with seed 42, excludes member ID from modeling, and fits the scaler and every k-means model on the 231 training members only. It scores the 99 held-out members against the training centers. Both spread curves flatten substantially after four groups.

Segment names summarize the training centroids in original units. Counts use all 330 members. The offer decision gives the credit to a segment only if the average member-level net is positive on training members in that segment. A member spending below $20 nets $5 with the credit. Anyone spending at least $20 nets −$10 with the credit. Members not receiving the credit contribute $0.

The script reports the new member's segment and offer. It scores the monthly net for both policies across all 330 members. The estimates use the stated finance assumptions.

Generated outputs are under `results/`: elbow chart, training/test spread, centroids, member assignments, and summary. The included `analysis_results.txt` records the reproducible results.
