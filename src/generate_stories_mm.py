import pandas as pd
import requests
from multiclass_Stories import SHAPstory


def call_ollama_llm(prompt, model="deepseek-r1:14b"):
    response = requests.post(
        "http://localhost:11434/api/generate",  # default Ollama API
        json={
            "model": model,  
            "prompt": prompt,
            "stream": False  # False to get complete response
        }
    )
    return response.json()["response"]  


# vLLM API call function
def call_vllm_llm(prompt, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]


def get_prompt_config(dataset="mm"):
    """
    Returns prompt config based on dataset name.

    Parameters:
    -----------
    dataset : str
        Name of the dataset. Options: "flchain", "mm", "mm_molecular", "gbsg2"
    """

    if dataset == "mm":
        # ----------------------------description for mm----------------------------
        feature_descriptions = [
            "Âge: Patient's age at diagnosis (years). Older age is generally associated with poorer prognosis.",
            "Stade_SD: Durie-Salmon stage (1 = IA to 6 = IIIB). Higher stages indicate more advanced multiple myeloma and worse prognosis.",
            "Stade_ISS: International Staging System stage (0 = unknown, 1 = I, 2 = II, 3 = III). Higher stage reflects greater tumor burden.",
            "poucentage_CD138_Moelle: Percentage of CD138+ plasma cells in bone marrow aspirate. Normal <5%. Values >10% suggest abnormal plasma cell proliferation; >60% strongly supports myeloma diagnosis.",
            "NPC_par_mm3: Total number of plasma cells per mm³ of bone marrow. Normal range: 5,000–20,000/mm³. Higher values indicate increased plasma cell burden.",
            "MMC_par_mm3 (Moelle): Number of malignant plasma cells per mm³ of bone marrow. Normal: 0. Any detectable malignant cells are pathological.",
            "Pourcentage_Phase S PC tumoraux: Percentage of plasma tumor cells in the S-phase (DNA synthesis). Normal <1%. Higher percentages suggest rapid proliferation and worse prognosis.",
            "Ostéolyse: Presence of osteolytic bone lesions (1 = present, 0 = absent). Presence indicates advanced disease.",
            "Tx_Sg Prot MonoCl: Serum monoclonal protein (M-protein) concentration. Normal: 0 g/L. Levels >30 g/L suggest active myeloma; >60 g/L indicates high tumor burden.",
            "Taux IgG g/L: Immunoglobulin G concentration. Normal: 7–16 g/L. Abnormal levels may reflect monoclonal gammopathy.",
            "Taux IgA g/L: Immunoglobulin A concentration. Normal: 0.7–4 g/L. Elevated levels may indicate IgA myeloma subtype.",
            "Taux IgM g/L: Immunoglobulin M concentration. Normal: 0.4–2.3 g/L. Typically low in myeloma due to immune suppression.",
            "Calcémie mmol/L: Serum calcium. Normal: 2.15–2.55 mmol/L. Hypercalcemia (>2.6) suggests bone resorption and tumor activity.",
            "protéinurie g/24h: 24-hour urinary protein excretion. Normal <0.15 g/24h. Higher values indicate renal involvement.",
            "Hémoglobine g/dL: Hemoglobin concentration. Normal: 13–17 (male), 12–15 (female). Low values indicate anemia, common in myeloma.",
            "Béta-2M mg/L: Beta-2 microglobulin. Normal <2.5 mg/L. Elevated levels correlate with tumor burden and poor prognosis.",
            "CRP mg/L: C-reactive protein. Normal <5 mg/L. High levels reflect systemic inflammation and may indicate active disease.",
            "LDH IU/L: Lactate dehydrogenase. Normal <250 IU/L (lab-dependent). High levels reflect tissue turnover and poor prognosis.",
            "Albumine g/L: Serum albumin. Normal: 35–50 g/L. Low levels may indicate malnutrition, inflammation, or advanced disease.",
            "Créatinine µmol/L: Serum creatinine. Normal: 44–106 µmol/L. Elevated values suggest impaired renal function.",
            "%CD138 au myélogramme: Percentage of CD138+ plasma cells in bone marrow smear. Normal <5%. Higher percentages indicate abnormal infiltration.",
            "Nb greffes: Number of hematopoietic stem cell transplants received. May reflect treatment intensity and response.",
            "IgH_BJ: Presence of Bence-Jones (urinary light chain) myeloma subtype. Indicates kidney involvement risk.",
            "IgH_IgA: Presence of IgA heavy chain subtype. Associated with specific biological features of myeloma.",
            "IgH_IgD: Presence of IgD heavy chain subtype. Rare and often associated with aggressive disease.",
            "IgH_IgG: Presence of IgG heavy chain subtype. The most common myeloma subtype.",
            "IgH_IgG,IgA: Presence of both IgG and IgA heavy chains (biclonal myeloma). Rare but clinically relevant.",
            "IgH_NS: Non-specified immunoglobulin heavy chain subtype.",
            "IgL_Kappa: Presence of kappa light chain subtype. Important for myeloma classification.",
            "IgL_Kappa,Lambda: Presence of both kappa and lambda light chains (biclonal light chain involvement).",
            "IgL_Lambda: Presence of lambda light chain subtype. Important for myeloma classification."
        ]

        dataset_description = (
            "a dataset on patients diagnosed with multiple myeloma,  "
            "containing diagnostic information, biomarkers, treatment history, and clinical outcomes"
        )

        input_description = "the complete clinical profile of a patient with multiple myeloma"
        event_description_osa = "death from multiple myeloma"
        event_description_efsa = "multiple myeloma relapse"



    elif dataset == "gbsg2":
        # ------------------------------------------gbsg2--------------------------------------------
        feature_descriptions = [
            "age: Patient age at diagnosis (years). Older age is generally associated with worse prognosis.",
            "estrec: Estrogen receptor level (fmol/L). Higher levels indicate hormone-sensitive tumors and are associated with better response to endocrine therapy.",
            "progrec: Progesterone receptor level (fmol/L). Like ER, higher values suggest hormone sensitivity and may be favorable.",
            "pnodes: Number of positive axillary lymph nodes. More positive nodes indicate higher risk and poorer prognosis.",
            "tsize: Primary tumor size in millimeters. Larger tumors are associated with increased risk of recurrence.",
            "tgrade: Histologic tumor grade (1 = well differentiated, 2 = moderately differentiated, 3 = poorly differentiated). Higher grade reflects more aggressive disease.",
            "menostat: Menopausal status at diagnosis (0 = premenopausal, 1 = postmenopausal). Prognosis and treatment may differ between groups.",
            "horTh: Receipt of adjuvant hormone therapy (0 = no, 1 = yes). Endocrine therapy can improve outcomes in hormone receptor–positive disease."
        ]

        dataset_description = (
            "a breast cancer cohort from the German Breast Cancer Study Group (GBSG2), "
            "including clinical and hormone receptor variables with time-to-event outcomes for survival analysis"
        )

        input_description = "the clinical and receptor profile of a breast cancer patient (GBSG2)"

        event_description_osa = "locoregional recurrence, distant metastases, or death from breast cancer"
        event_description_efsa = "locoregional recurrence, distant metastases, or death from breast cancer"


    elif dataset == "flchain":
        # -------------------------------------------flchain----------------------------------------------------------
        feature_descriptions = [
            "age: Age of the participant at baseline (years). Older age may be associated with higher mortality risk.",
            "sex: Biological sex of the participant (0 = female, 1 = male). Some survival differences may exist between sexes.",
            "sample.yr: Calendar year of blood sample collection. Later years may reflect changes in diagnostics or treatment practices.",
            "kappa: Serum free kappa light chain concentration (mg/L). Abnormal levels may indicate plasma cell disorders.",
            "lambda: Serum free lambda light chain concentration (mg/L). Abnormal levels may indicate plasma cell disorders.",
            "creatinine: Serum creatinine concentration (mg/dL). Elevated levels can indicate impaired kidney function, which is relevant for prognosis.",
            "mgus: Indicator for monoclonal gammopathy of undetermined significance (0 = no, 1 = yes). MGUS is a precursor condition to multiple myeloma and related disorders."
        ]

        dataset_description = (
            "a cohort from the US general population (NHANES-linked), "
            "including demographics, kidney function, and serum free light chain measurements with survival follow-up data (flchain dataset)."
        )

        input_description = "the demographic, laboratory, and clinical profile of a study participant from the flchain cohort"
        event_description_osa = "death from any cause (overall survival)"
        event_description_efsa = "death from any cause (overall survival)"

    elif dataset == "metabric":
        # -------------------------------------------metabric----------------------------------------------------------
        feature_descriptions = [
            "Age at Diagnosis: Patient age in years at initial diagnosis. Older age is generally associated with poorer prognosis.",
            "Tumor Size: Primary tumor size (typically in mm). Larger tumors are linked to higher recurrence risk.",
            "Lymph nodes examined positive: Count of positive regional lymph nodes. More positive nodes indicate higher stage and worse outcomes.",
            "Nottingham prognostic index: Composite score from tumor size, nodal status, and grade; higher NPI indicates worse prognosis.",
            "Mutation Count: Total number of somatic coding mutations detected. Often correlates with genomic instability.",
            "TMB (nonsynonymous): Nonsynonymous tumor mutational burden (mutations/Mb). Higher TMB may reflect neoantigen load; prognostic value varies by subtype.",
            "Tumor Stage: Overall AJCC/clinical stage; higher stage denotes more advanced disease and poorer prognosis.",
            "Neoplasm Histologic Grade: Nottingham histologic grade (1–3). Higher grade reflects poorer differentiation and more aggressive biology.",

            "Chemotherapy: Receipt of systemic chemotherapy (1 = yes, 0 = no).",
            "Hormone Therapy: Receipt of adjuvant endocrine therapy (1 = yes, 0 = no).",
            "Radio Therapy: Receipt of radiotherapy (1 = yes, 0 = no).",

            "ER Status: Estrogen receptor status (1 = positive, 0 = negative). ER-positive tumors tend to benefit from endocrine therapy.",
            "PR Status: Progesterone receptor status (1 = positive, 0 = negative). PR positivity often supports endocrine sensitivity.",
            "HER2 Status: HER2 receptor status (1 = positive, 0 = negative). HER2-positive disease may benefit from anti-HER2 therapy.",

            "Pam50 + Claudin-low subtype_BASAL: Indicator for Basal-like subtype (1 = yes, 0 = no); typically more aggressive, often ER-/PR-/HER2-.",
            "Pam50 + Claudin-low subtype_CLAUDIN-LOW: Indicator for Claudin-low subtype; characterized by low cell–cell adhesion gene expression and immune/stem-like features.",
            "Pam50 + Claudin-low subtype_HER2: Indicator for HER2-enriched subtype; often higher proliferation and sensitivity to anti-HER2 therapy.",
            "Pam50 + Claudin-low subtype_LUMA: Indicator for Luminal A subtype; generally best prognosis with endocrine responsiveness.",
            "Pam50 + Claudin-low subtype_LUMB: Indicator for Luminal B subtype; more proliferative than Luminal A, intermediate prognosis.",
            "Pam50 + Claudin-low subtype_NORMAL: Indicator for Normal-like subtype; reference-like profile with variable clinical behavior.",

            "3-Gene classifier subtype_ER+/HER2- HIGH PROLIF: Indicator for ER+/HER2− high-proliferation class.",
            "3-Gene classifier subtype_ER+/HER2- LOW PROLIF: Indicator for ER+/HER2− low-proliferation class.",
            "3-Gene classifier subtype_ER-/HER2-: Indicator for ER−/HER2− (triple-negative–like) class.",
            "3-Gene classifier subtype_HER2+: Indicator for HER2+ class.",
            "3-Gene classifier subtype_UNKNOWN: Indicator for subtype not assigned/unknown.",

            "Integrative Cluster_1: Indicator for Integrative Cluster (IntClust) 1 membership.",
            "Integrative Cluster_10: Indicator for IntClust 10 membership.",
            "Integrative Cluster_2: Indicator for IntClust 2 membership.",
            "Integrative Cluster_3: Indicator for IntClust 3 membership.",
            "Integrative Cluster_4ER+: Indicator for IntClust 4 ER-positive membership.",
            "Integrative Cluster_4ER-: Indicator for IntClust 4 ER-negative membership.",
            "Integrative Cluster_5: Indicator for IntClust 5 membership.",
            "Integrative Cluster_6: Indicator for IntClust 6 membership.",
            "Integrative Cluster_7: Indicator for IntClust 7 membership.",
            "Integrative Cluster_8: Indicator for IntClust 8 membership.",
            "Integrative Cluster_9: Indicator for IntClust 9 membership.",

            "Type of Breast Surgery: Binary coding (1 = MASTECTOMY, 0 = BREAST CONSERVING); missing/other treated as NA.",
            "Inferred Menopausal State: Binary coding (1 = POST, 0 = PRE); missing/other treated as NA.",
            "Primary Tumor Laterality: Binary coding (1 = RIGHT, 0 = LEFT); missing/other treated as NA."
        ]
        dataset_description = (
            "a METABRIC breast cancer cohort with clinicopathologic features, treatments, and molecular subtypes, "
            "used to study prognosis and treatment response"
        )

        input_description = "the complete clinico-genomic profile of a patient with breast cancer"

        event_description_osa = "death from any cause (overall survival event)"
        event_description_efsa = "breast cancer relapse/progression or related event"


    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported datasets: 'flchain', 'mm',‘mm_grouped', 'mm_molecular', 'gbsg2','metabric', 'metabric_grouped'")

    # Create a mapping dictionary for feature descriptions
    feature_desc_dict = {}
    for desc in feature_descriptions:
        # Extract feature name from the description (everything before the first colon)
        feature_name = desc.split(':')[0].strip()
        feature_desc_dict[feature_name] = desc

    return feature_desc_dict, dataset_description, input_description, event_description_osa, event_description_efsa


def generate_narratives(median_shap_df, feature_names, shap_cutoff=0.005, model="Qwen/Qwen1.5-32B-Chat", event='osa',
                        dataset="flchain"):
    # Get descriptions and dataset summary
    feature_desc_dict, dataset_description, input_description, event_description_osa, event_description_efsa = get_prompt_config(
        dataset)

    # Build description DataFrame
    feature_df = pd.DataFrame({
        "feature_name": feature_names,
        "feature_desc": [feature_desc_dict.get(name, "Description not found") for name in feature_names]
    })

    # Select the appropriate event description for clarity
    if event == 'osa':
        event_description = event_description_osa
    elif event == 'efsa':
        event_description = event_description_efsa
    else:
        raise ValueError(f"Unsupported event type: {event}")

    # SHAPstory no longer needs the model or raw data, just the processed dataframe
    story_generator = SHAPstory(
        data_df=median_shap_df,
        feature_desc_df=feature_df,
        dataset_description=dataset_description,
        input_description=input_description,
        event_description=event_description
        , shap_cutoff=shap_cutoff,
        event=event
    )

    # For each instance in median_shap_df, generate prompt (no API call here)

    sample_indices = median_shap_df["sample_index"].unique()

    results = []

    # once
    for sample_idx in sample_indices:
        prompt = story_generator.generate_prompt(sample_idx)
        print(f"Calling local LLM for sample {sample_idx}...")
        response_text = call_vllm_llm(prompt, model=model)
        # response_text = call_ollama_llm(prompt,model="deepseek-r1:8b")
        # print(f"Sample{sample_idx}:\n")
        # print(f"Prompt: {prompt}\n")
        # print(f"Response: {response_text}")
        results.append({
            "sample_idx": sample_idx,
            "prompt": prompt,
            "response_text": response_text
        })

    # multiple times
    # for sample_idx in sample_indices:
    #     prompt = story_generator.generate_prompt(sample_idx)
    #     print(f"Calling local LLM for sample {sample_idx}...")
    #     print(f"Sample {sample_idx} ")
    #     print(f"Prompt: {prompt}\n")
    #
    #     for i in range(3):  # generate 3 responses per sample
    #         response_text = call_vllm_llm(prompt, model=model)
    #         print(f"Response {i + 1}:\n")
    #         print(f"Prompt: {prompt}\n")
    #         print(f"Response: {response_text}\n")
    #
    #         results.append({
    #             "sample_index": sample_idx,
    #             "response_number": i + 1,
    #             "response": response_text
    #         })
    #

    result_df = pd.DataFrame(results)[["sample_idx", "prompt", "response_text"]]
    safe_model = model.replace("/", "_").replace(":", "_")
    filename = f"fixed_responses_{dataset}_{safe_model}.csv"
    # store the responses as csv files
    result_df.to_csv(filename, index=False)
    return result_df
