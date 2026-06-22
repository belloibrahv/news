import sys
import os

# Add src to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from predictor import predict_category

# Predefined 10 manual test cases from Section 12.1
TEST_CASES = [
    {
        "id": "TC-01",
        "text": "The National Assembly passed the 2025 Appropriation Bill yesterday after a long session.",
        "expected": "Politics"
    },
    {
        "id": "TC-02",
        "text": "Super Eagles defeated Cameroon 2-1 in the AFCON qualifier match and qualified for the next stage.",
        "expected": "Sports"
    },
    {
        "id": "TC-03",
        "text": "Flutterwave announced a new API integration for African fintech companies to ease cross-border payments.",
        "expected": "Technology"
    },
    {
        "id": "TC-04",
        "text": "Nollywood actress wins award at the Africa Magic Viewers Choice ceremony held in Lagos.",
        "expected": "Entertainment"
    },
    {
        "id": "TC-05",
        "text": "The Central Bank of Nigeria raised the benchmark interest rate to 27.5% to stabilize the economy.",
        "expected": "Business"
    },
    {
        "id": "TC-06",
        "text": "Dangote Refinery begins petrol export to West African markets, aiming to boost regional trade.",
        "expected": "Business"
    },
    {
        "id": "TC-07",
        "text": "Lagos State Government announces new broadband infrastructure plan to boost connection speeds.",
        "expected": "Technology"  # Spec warns: "expected fail — Politics/Technology boundary"
    },
    {
        "id": "TC-08",
        "text": "Afrobeats artist Burna Boy wins Grammy for Global Impact, gaining international recognition.",
        "expected": "Entertainment"
    },
    {
        "id": "TC-09",
        "text": "Nigeria inflation rate hits 34.2% — NBS report 2025 shows rising consumer prices.",
        "expected": "Business"
    },
    {
        "id": "TC-10",
        "text": "Moses Simon scores hat-trick as Nantes beats PSG 3-2 in an astonishing Ligue 1 display.",
        "expected": "Sports"
    }
]

def run_system_tests():
    print("=" * 90)
    print("RUNNING FUNCTIONAL SYSTEM TESTS (10 Manual Test Cases)")
    print("=" * 90)
    
    passed_count = 0
    results = []
    
    # Header format
    row_format = "{:<8} | {:<40} | {:<13} | {:<13} | {:<8}"
    print(row_format.format("Test ID", "Article Extract (First 40 Chars)", "Expected", "Predicted", "Result"))
    print("-" * 90)
    
    for case in TEST_CASES:
        text = case["text"]
        expected = case["expected"]
        test_id = case["id"]
        
        # Get preview
        preview = text[:37] + "..." if len(text) > 40 else text
        
        try:
            prediction = predict_category(text)
            predicted = prediction["category"]
            
            if predicted == expected:
                result = "PASS"
                passed_count += 1
            else:
                result = "FAIL"
                
            results.append((test_id, preview, expected, predicted, result))
        except Exception as e:
            results.append((test_id, preview, expected, "ERROR", "FAIL"))
            print(f"Error on {test_id}: {str(e)}")
            
    for r in results:
        print(row_format.format(*r))
        
    print("=" * 90)
    pass_rate = (passed_count / len(TEST_CASES)) * 100
    print(f"RESULTS SUMMARY: {passed_count}/{len(TEST_CASES)} passed ({pass_rate:.1f}%)")
    print(f"Target Threshold: 90.0% (minimum 9/10 passed)")
    print("=" * 90)
    
    if pass_rate >= 90.0:
        print("SYSTEM TEST RUN: SUCCESSFUL\n")
        return True
    else:
        print("SYSTEM TEST RUN: FAILED (Below 90% threshold)\n")
        return False

if __name__ == "__main__":
    success = run_system_tests()
    sys.exit(0 if success else 1)
