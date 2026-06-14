"""Optional Streamlit dashboard for live demo transition logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_LOG = Path("logs/state_transitions.csv")


def load_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_epoch",
                "timestamp_iso",
                "previous_state",
                "new_state",
                "drowsiness_score",
                "distraction_score",
            ]
        )
    df = pd.read_csv(path)
    if "timestamp_iso" in df:
        df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="Driver Monitoring Logs", layout="wide")
    st.title("Driver Monitoring Logs")
    st.caption(str(args.log))

    df = load_log(args.log)
    if df.empty:
        st.info("No state transitions logged yet. Run the realtime demo first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Transitions", len(df))
    col2.metric("Latest state", str(df.iloc[-1]["new_state"]))
    col3.metric("Critical events", int((df["new_state"] == "CRITICAL").sum()))

    counts = df["new_state"].value_counts().reset_index()
    counts.columns = ["state", "count"]
    st.plotly_chart(px.bar(counts, x="state", y="count", title="Transitions by State"), use_container_width=True)

    timeline = df.dropna(subset=["timestamp_iso"])
    if not timeline.empty:
        st.plotly_chart(
            px.scatter(
                timeline,
                x="timestamp_iso",
                y="new_state",
                color="new_state",
                size="drowsiness_score",
                hover_data=["previous_state", "drowsiness_score", "distraction_score"],
                title="State Transition Timeline",
            ),
            use_container_width=True,
        )

    st.dataframe(df.sort_values("timestamp_epoch", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
