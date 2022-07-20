import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np

import altair as alt

import experiments as ex
import util

col1, col2 = st.columns(2)

with col1:
    st.title("Multi-Armed Bandit Value Calculator")
    st.markdown("""
            The purpose of this calculator is to help you to determine the benefit of using multi 
            armed bandits to compare different models for a given use case. 

            The calculator determines the financial benefit of running a MAB experiment versus a naive
            approach of simply splitting incoming traffic evenly between all model instances.

            You will fill in the following parameters to allow the calculator to do its magic:
            - **Cost of Failure**: This is the financial impact of your model misclassifying a data point.
            - **Unit of Time**: This is the unit of time over which your model receives requests. e.g. a
            web scale recommendation system is likely responding to 
            - **Length of Time**: The number of time steps you would like your MAB experiment to run for.
            e.g. with a Unit of Time of "weeks", and setting Length of Time to 52, you would like
            your experiment to run for 52 weeks.
            - **Number of Reward Requests**: 
            This field refers to the number of reward requests which are fedback to the deployment within the selected time window. 
            A correct prediction gets a reward of 1, while an incorrect prediction receives a reward of 0. These rewards
            are not available at prediction time, and are instead fedback to the deployment after the fact.  e.g. if the number of reward requests 
            is set to 50, and the unit of time is weeks, this implies that each week your deployment receives
            50 updated requests with rewards. 
            - **Number of Requests**: The number of inference requests sent to the model within the selected time window. 
            - **Number of Models**: The number of different model instances to be compared. Sliders to set the
            accuracy of each will be displayed as they are added.
            - **Routing Algorithm**: The algorithm being used to determine the traffic splitting for the 
            MAB. Currently only Thompson Sampling is supported. 
            """)

with col2:
    st.title(" ")
    COST_OF_FAILURE = st.number_input("Cost of Failure:", min_value=1, value=25)
    TIME_UNIT = st.selectbox("Unit of Time:",
                            options=["seconds", "minutes", "hours", "days", "weeks", "months", "years"])
    TIME_SERIES_LENGTH = st.number_input("Length of Time:", min_value=1, value=100)
    NO_OF_REWARDS = st.number_input("Number of Reward Requests:", min_value=1, value=50)
    NO_OF_REQUESTS = st.number_input("Number of Requests", min_value=1, value=1000)
    NO_OF_MODELS = st.number_input("Number of Models:", min_value=2, max_value=10)

    MODEL_ACCURACIES = []
    for i in range(NO_OF_MODELS):
        accuracies = st.slider(f"Model {i+1} accuracy:", min_value=0.0, max_value=1.0, value=0.9)
        MODEL_ACCURACIES.append(accuracies)

    ROUTING_ALGO = st.selectbox("Routing Algorithm:", options=["Thompson Sampling"])

# Generating Thompson Sampling experiment data
ts_bandit_selected = ex.generate_ts_time_series(MODEL_ACCURACIES, TIME_SERIES_LENGTH, NO_OF_REWARDS)
ts_bandit_selected = util.format_as_dataframe(ts_bandit_selected)
ts_misclassifications = util.create_misclassification_df(ts_bandit_selected, MODEL_ACCURACIES) * ((NO_OF_REQUESTS // NO_OF_MODELS) // NO_OF_REWARDS)

# Generating control experiment data
control_bandit_selected = ex.generate_control_time_series(MODEL_ACCURACIES, TIME_SERIES_LENGTH, NO_OF_REWARDS)
control_bandit_selected = util.format_as_dataframe(control_bandit_selected)
control_misclassifications = util.create_misclassification_df(control_bandit_selected, MODEL_ACCURACIES) * ((NO_OF_REQUESTS // NO_OF_MODELS) // NO_OF_REWARDS)

# Translating misclassifications to the financial cost between control and bandit
total_ts_misclassifications = sum(ts_misclassifications.iloc[-1, :])
total_control_misclassifications = sum(control_misclassifications.iloc[-1, :])
misclassification_diff = total_control_misclassifications - total_ts_misclassifications
cost_incurred = misclassification_diff * COST_OF_FAILURE


# Cumulative cost of misclassification plot
ts_cumsum = ts_misclassifications.sum(axis=1).iloc[:-1].cumsum() * COST_OF_FAILURE
control_cumsum = control_misclassifications.sum(axis=1).iloc[:-1].cumsum() * COST_OF_FAILURE

ts_cumsum = pd.concat([ts_cumsum, pd.Series(["Thompson Sampling" for i in range(len(ts_cumsum))])], axis=1).reset_index()
control_cumsum = pd.concat([control_cumsum, pd.Series(["Control" for i in range(len(control_cumsum))])], axis=1).reset_index()
total_cumsum = pd.concat([ts_cumsum, control_cumsum])
total_cumsum.columns = [f"Time step ({TIME_UNIT})", "Cost of failure", "Algorithm"]

with col1:
    st.title(f"Cumulative Financial Impact: ${cost_incurred}")

    fig = alt.Chart(total_cumsum).mark_line().encode(
        x=f"Time step ({TIME_UNIT})",
        y='Cost of failure',
        color='Algorithm',
        strokeDash='Algorithm'
    ).interactive()

    st.altair_chart(fig, use_container_width=True)

    # Tabular information that we care about
    model_misclassifications = pd.concat([control_misclassifications.sum(), ts_misclassifications.sum()], axis=1)
    model_misclassifications.index = model_misclassifications.index + 1
    model_misclassifications.columns = ["Control", "Thompson Sampling"]
    st.markdown("The table below displays the number of misclassifications made by each model in the control and MAB experiment respectively.")
    st.write(model_misclassifications)

    totals = pd.Series(model_misclassifications.sum(numeric_only=True, axis=0), name="Totals")
    st.markdown(f"In total the control experiment misclassifies **{totals[0]}** predictions, while the MAB misclassifies **{totals[1]}**.\
                  The MAB reduces the total number of errors by **{totals[0] - totals[1]}**.")