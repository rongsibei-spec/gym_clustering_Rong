"""Fit gym member segments and estimate monthly offer value."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
            "months_member", "monthly_extras_spend"]
SEED = 42
K = 4
NEW_MEMBER = dict(zip(FEATURES, [30, 4, 42, 1, 5, 10]))


def credit_net(extras_spend):
    """Per-member net under the assignment's finance assumptions."""
    return np.where(extras_spend < 20, 25 * 0.60 - 10, -10)


def member_offer(member, scaler, model, names, eligibility):
    """Return the fitted segment and whether that segment gets credit."""
    row = pd.DataFrame([member], columns=FEATURES)
    cluster = int(model.predict(scaler.transform(row))[0])
    return names[cluster], bool(eligibility.loc[cluster])


def plot_elbow(spread, output):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(spread.index, spread["Train"], "o-", label="Training (231)", color="#086f72")
    ax.plot(spread.index, spread["Test"], "s-", label="Held out (99)", color="#556b83")
    ax.axvline(K, color="#84939e", linestyle="--", linewidth=1)
    ax.set(xlabel="Number of segments (k)", ylabel="Sum of squared distances",
           xticks=spread.index, ylim=(0, None))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "elbow.png", dpi=200)
    plt.close(fig)


def main():
    members = pd.read_csv(ROOT / "gym_members.csv")
    x = members[FEATURES]
    train, test = train_test_split(x, test_size=0.30, random_state=SEED)
    scaler = StandardScaler().fit(train)
    train_scaled, test_scaled = scaler.transform(train), scaler.transform(test)
    rows = []
    models = {}
    for k in range(1, 9):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(train_scaled)
        test_distance = model.transform(test_scaled).min(axis=1)
        rows.append({"k": k, "Train": model.inertia_, "Test": np.square(test_distance).sum()})
        models[k] = model
    spread = pd.DataFrame(rows).set_index("k")
    model = models[K]
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    order = centers["visits_per_month"].sort_values().index
    names = dict(zip(order, ["Light users", "Class regulars", "Long-time members", "Frequent trainers"]))
    train_net = pd.Series(credit_net(train["monthly_extras_spend"]), index=train.index)
    # Choose one action for the entire segment using its training-only expected net.
    eligibility = train_net.groupby(model.labels_).mean().gt(0)
    assignments = members.copy()
    assignments["cluster"] = model.predict(scaler.transform(x))
    assignments["segment"] = assignments["cluster"].map(names)
    assignments["gets_credit"] = assignments["cluster"].map(eligibility)
    assignments["baseline_net"] = credit_net(assignments["monthly_extras_spend"])
    assignments["targeted_net"] = assignments["baseline_net"].where(assignments["gets_credit"], 0)
    profiles = centers.assign(
        segment=pd.Series(names),
        members=assignments.groupby("cluster").size(),
        credit=eligibility,
    ).loc[order]
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    spread.to_csv(output / "spread.csv")
    profiles.to_csv(output / "centers.csv", index_label="cluster")
    assignments.to_csv(output / "assignments.csv", index=False)
    plot_elbow(spread, output)
    baseline = float(assignments["baseline_net"].sum())
    targeted = float(assignments["targeted_net"].sum())
    name, credit = member_offer(NEW_MEMBER, scaler, model, names, eligibility)
    summary = {
        "members": len(members), "train": len(train), "test": len(test),
        "average_extras_spend": float(x["monthly_extras_spend"].mean()),
        "chosen_k": K, "baseline_test_spread": float(spread.loc[1, "Test"]),
        "v1_test_spread": float(spread.loc[K, "Test"]),
        "test_spread_reduction_pct": float(100 * (1 - spread.loc[K, "Test"] / spread.loc[1, "Test"])),
        "new_member_segment": name, "new_member_credit": credit,
        "baseline_monthly_net": baseline, "v1_monthly_net": targeted,
        "monthly_improvement": targeted - baseline,
        "credit_recipients": int(assignments["gets_credit"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nWithin-group spread\n" + spread.round(2).to_string())
    print("\nCentroids in original units and full-data counts\n" + profiles.round(2).to_string())
    print("\nMember-level offer economics by segment\n" +
          assignments.groupby("segment")[["baseline_net", "targeted_net"]].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
