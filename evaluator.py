import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

client = OpenAI()

SPECIALTIES = [
    "Surgery",
    "Cardiovascular / Pulmonary",
    "Orthopedic",
    "Radiology",
    "Neurology"
]

def evaluate_single_prediction(text: str, true_label: str, predicted_label: str) -> dict:
    """Use GPT to evaluate a single classifier prediction."""

    # truncate very long notes to keep costs low
    truncated_text = text[:1500] if len(text) > 1500 else text

    prompt = f"""You are an expert clinical NLP evaluator assessing a medical text classifier. 

    A classifier predicted the medical specialty of the following clinical note.
    Your job is to evaluate whether the prediction is correct and explain your reasoning.

    Clinical note:
    {truncated_text}

    True specialty: {true_label}
    Classifier prediction: {predicted_label}

    Available specialties: {", ".join(SPECIALTIES)}

    Respond in valid JSON with exactly these fields:
    {{
        "is_correct": true or false,
        "confidence": a number between 0 and 1 indicating how confident you are,
        "reasoning": "2-3 sentences explaining why the prediction is correct or incorrect",
        "key_clinical_terms": ["list", "of", "3-5", "terms", "that", "indicate", "the", "specialty"],
        "suggested_label": "the specialty you would predict, which may differ from both"
    }}

    Return only valid JSON. No preamble, no explanation outside the JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role":"user","content":prompt}]
        )

        raw = response.choices[0].message.content.strip()

        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw=raw[4:]
        
        result = json.loads(raw)
        result['text'] = truncated_text
        result['true_label'] = true_label
        result['predicted_label'] = predicted_label
        
        return result
    
    except Exception as e:
        return{
            "is_correct": None,
            "confidence": None,
            "reasoning": f"Evaluation failed: {str(e)}",
            "key_clinical_terms": [],
            "suggested_label": None,
            "text": truncated_text,
            "true_label": true_label,
            "predicted_label": predicted_label,
        }
    
def evaluate_test_set(test_csv: str = "data/test_results.csv", 
                      sample_size: int = 30) -> pd.DataFrame:
    """Evaluate a sample of test predictions and return results."""

    df = pd.read_csv(test_csv)

    # Sample evenly across correct and incorrect predictions
    # Focus more on misclassifications since those are more interesting
    correct = df[df['label'] == df['predicted']].sample(
        min(10, len(df[df['label'] == df['predicted']])),
        random_state=42
    )
    incorrect = df[df['label'] != df['predicted']].sample(
        min(20, len(df[df['label'] != df['predicted']])),
        random_state=42
    )

    sample = pd.concat([correct, incorrect]).reset_index(drop=True)
    print(f"Evaluating {len(sample)} predictions ({len(incorrect)} misclassifications, {len(correct)} correct)...")

    results = []
    for i, row in sample.iterrows():
        print(f"    Evaluating {i+1}/{len(sample)}...")
        result = evaluate_single_prediction(
            text=row['text'],
            true_label=row['label'],
            predicted_label=row['predicted']
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)

    # save results
    results_df.to_csv('data/evaluation_result.csv', index=False)
    print(f"\nEvaluation complete. Results saved to data/evaluation_result.csv")

    # Print summary
    valid = results_df[results_df["is_correct"].notna()]
    llm_accuracy = valid["is_correct"].mean()

    # Fix: calculate classifier accuracy correctly from test_results.csv
    test_df = pd.read_csv(test_csv)
    real_classifier_accuracy = (test_df["label"] == test_df["predicted"]).mean()

    print(f"\n--- Evaluation Summary ---")
    print(f"Classifier accuracy on full test set: {real_classifier_accuracy:.1%}")
    print(f"LLM agreement with true labels: {llm_accuracy:.1%}")
    print(f"Average LLM confidence: {valid['confidence'].mean():.2f}")
    print(f"\nMisclassification breakdown:")
    wrong = results_df[results_df["is_correct"] == False].copy()
    print(wrong[["true_label", "predicted_label", "reasoning"]].to_string())

    # Identify boundary cases — wrong but clinically reasonable
    boundary_keywords = ["reasonable", "subspecialty", "both", "while", "although", "however"]
    wrong["is_boundary"] = wrong["reasoning"].str.lower().apply(
        lambda x: any(word in x for word in boundary_keywords)
    )
    boundary_count = wrong["is_boundary"].sum()
    print(f"\nBoundary/ambiguous cases: {boundary_count}/{len(wrong)} misclassifications")
    print(f"Adjusted accuracy (excluding boundary cases): {(valid['is_correct'].sum() + boundary_count) / len(valid):.1%}")
    return results_df

if __name__ == "__main__":
    evaluate_test_set()