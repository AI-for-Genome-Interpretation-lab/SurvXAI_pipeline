import pandas as pd
from generate_stories_mm import generate_narratives

def main():
    # METABRIC dataset
    df = pd.read_csv("../outputs/metabric_sample.csv")

    #select some examples to see how the LLM works
    selected_samples = [49,58,72,88,121]
    sample_subset = df[df["sample_index"].isin(selected_samples)]

    feature_names = sample_subset["variable_name"].unique().tolist()

    #create another window to run vLLM and generate responses
    #model = the same name as the LLM on vLLM, osa = event of death
    generate_narratives(sample_subset, feature_names, shap_cutoff=0.01,model="openai/gpt-oss-120b", event='osa', dataset="metabric")


if __name__ == '__main__':
    main()