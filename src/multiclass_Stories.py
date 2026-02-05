import pandas as pd
import shap
import numpy as np


class SHAPstory():
    """
      A class to generate SHAPstories, narratives that explain AI predictions based on precomputed SHAP values.

      Attributes:
      -----------
      data_df : DataFrame
          A DataFrame containing precomputed features, SHAP values, survival times, and other relevant data.
      feature_desc_df : DataFrame
          A DataFrame containing descriptions for each feature.
      dataset_description : str
          A brief description of the dataset.
      input_description : str
          Description of the input features.
      target_description : str
          Description of the target variable.
      """

    def __init__(self, data_df, feature_desc_df, dataset_description,
                 input_description, event_description, shap_cutoff, event):
        """Initializes the SHAPstory class with necessary parameters."""

        self.data_df = data_df
        self.feature_desc_df = feature_desc_df
        self.dataset_description = dataset_description
        self.input_description = input_description
        self.event_description = event_description
        self.shap_cutoff = shap_cutoff
        self.event = event

    def generate_prompt(self, sample_index):
        """
        Generates the prompt for LLMs to generate a narrative.

        Parameters:
        -----------
        sample_index : int
            The index of the sample for which the prompt is generated.

        Returns:
        --------
        str
            The generated prompt.
        """

        # setting a cutoff and filter features with shap value
        event = self.event
        group = self.data_df[self.data_df['sample_index'] == sample_index].copy()
        group = group[group["variable_value"] .notna()]  # filter feature without value
        group = group[group["shap_value"].abs() >= self.shap_cutoff]
        if group.empty:
            return f"[Sample {sample_index}] No features remained after filtering. Possibly all SHAP values < {self.shap_cutoff}."

        # group = group.sort_values(by="shap_value", ascending=False)
        group = group.reindex(group["shap_value"].abs().sort_values(ascending=False).index)

        shap_lines = []
        desc_lines = []
        for _, row in group.iterrows():
            name = row["variable_name"]
            val = row["variable_value"]
            shap = row["shap_value"]
            # direction = "increases" if shap > 0 else "decreases"
            shap_lines.append(f"{name}: {val} (SHAP {shap:+.4f}) ")
            desc_row = self.feature_desc_df[self.feature_desc_df["feature_name"] == name]
            desc_text = desc_row["feature_desc"].values[0] if not desc_row.empty else "Description not available."
            desc_lines.append(f"{name}: {desc_text}")

        shap_block = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(shap_lines))
        desc_block = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(desc_lines))

        y_event = group["y_event"].iloc[0]
        y_time = group["y_time"].iloc[0]
        median_time = group["median_time"].iloc[0]
        if event == 'osa':
            if y_event == 1:
                event_text = f"This patient's actual survival time was {y_time:.1f} months (the event occurred)."
            else:
                event_text = f"This patient was still alive at the last follow-up, with a recorded survival time of {y_time:.1f} months."


            prompt_string = f"""

            You are a data scientist interpreting a machine learning model’s prediction using computed SHAP values. You need to explain the role and the relevance of each feature used by the model to predict the survival of each patient by translating the numeric SHAP values into a narrative explanation. This explanation must be targeted to a medical expert, and you should assume that they are unfamiliar with machine learning. 
            This for example may include explaining what positive and negative SHAP values represent. 
            Your narrative explanation of how the features led to each patient survival prediction must focus on why the model made this prediction, given the SHAP values and the input features, *without judging or estimating the model performance in general*.  If the SHAP values of one or more features appear to conflict with established medical knowledge,  mention about the oddity, without trying to reconcile them. 


            Here I provide you with the background information for the patient case we are analyzing:

            A survival model was used to predict {self.dataset_description}.
            The input features describe the following aspects {self.input_description}.
            The target variables include the event indicator (whether the event occurred or was censored) and the time to the event, representing the predicted duration until {self.event_description}.

            The predicted result are the following:
            The survival model predicted a median survival time of {median_time:.1f} months. The actual, measured patient outcome is the following: {event_text}

            The model was interpreted by extracting SHAP values, which are shown in this table:
            {shap_block} 

            The table above includes every feature along with its value for that instance, and the SHAP value assigned to it. A positive SHAP value indicates that the corresponding feature increased the survival possibility (a more favorable outcome), while a negative value indicates a less favorable outcome.


            If you need, this is an additional clarification of the features:
            {desc_block}

            """

        elif event == 'efsa':
            if y_event == 1:
                event_text = f"The patient experienced a relapse or related event at {y_time:.1f} days."
            else:
                event_text = f"No relapse event occurred during the follow-up period of {y_time:.1f} days."

            prompt_string = f"""
                        You are a data scientist explaining an AI model’s prediction using computed SHAP values. You need to interpret the feature attributions for a medical expert and explain what positive and negative SHAP values represent.


                        Here's the background information for the patient case we are analyzing:


                        An AI survival model was used to predict {self.dataset_description}.
                        The input features of the data include data about {self.input_description}.
                        The target variables include the event indicator (whether the event occurred or was censored) and the time to the event, representing the predicted duration until {self.event_description}.


                        The predicted result and the actual target variables of a certain instance are shown below:
                        The model predicted that the patient would remain relapse-free for a median duration of {median_time:.1f} days.
                        {event_text} Please ensure your explanation focuses solely on *why the model made this prediction* and *does not* judge or estimate the model's performance in this case.


                        The provided SHAP table was generated to help us understand this outcome. It includes every feature along with its value for that instance, and the SHAP value assigned to it. For this explanation, a positive SHAP value indicates that a given variable has increased the survival possibility (a more favorable outcome), while a negative value indicates a decrease (a less favorable outcome).


                        The goal of SHAP is to explain the prediction of an instance by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory. The feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly distribute the “payout” (= the prediction) among the features.  If the interpretation of a feature's SHAP value appears to conflict with established medical facts, provide a concise explanation that emphasizes *what the model learned from its specific data patterns*. Your explanation should be accurate to the model's behavior and convincing, without misleading the audience by distorting medical facts.


                        Now, your task is to come up with a plausible, fluent, and correct story to explain why the model predicted this outcome. Focus on the features with the highest absolute SHAP values, and explain what positive and negative SHAP values indicate. In your story, try to explain the most important feature values and potential interactions that fit the narrative. If the interpretation of a feature's SHAP value appears to conflict with established medical facts, provide a concise explanation that emphasizes *what the model learned from its specific data patterns*. Your explanation should be accurate to the model's behavior and convincing, without misleading the audience by distorting medical facts. There is no need to enumerate individual features outside of the story. Conclude with a short summary of why this predicted survival time may have resulted. Limit your answer to 10-14 sentences.




                        SHAP table:
                        {shap_block}
                        Additional clarification of the features:
                        {desc_block}


                               """


        return prompt_string