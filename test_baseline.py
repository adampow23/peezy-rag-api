#!/usr/bin/env python3
"""
Peezy AI Baseline Test
======================

This script tests the Peezy AI system by:
1. Querying the RAG API for knowledge
2. (Optionally) Calling the full Firebase function
3. Recording results for baseline measurement

Run after deploying the RAG API to get your baseline metrics.

Usage:
  python test_baseline.py --rag-url https://your-railway-url.up.railway.app
  
  # With Firebase testing (requires auth):
  python test_baseline.py --rag-url https://... --firebase-url https://us-central1-your-project.cloudfunctions.net
"""

import argparse
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any

# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # Basic questions
    {
        "id": "basic_1",
        "message": "What should I do first?",
        "context": {},
        "category": "general",
        "expected_topics": ["planning", "start", "first"]
    },
    {
        "id": "basic_2", 
        "message": "How much will movers cost?",
        "context": {"move_distance": "local"},
        "category": "cost",
        "expected_topics": ["cost", "price", "quote"]
    },
    {
        "id": "basic_3",
        "message": "When should I book movers?",
        "context": {},
        "category": "timeline",
        "expected_topics": ["book", "schedule", "weeks"]
    },
    
    # Pet-related
    {
        "id": "pet_1",
        "message": "I have a dog, what do I need to know?",
        "context": {"has_pets": True},
        "category": "pets",
        "expected_topics": ["pet", "dog", "health"]
    },
    {
        "id": "pet_2",
        "message": "Does my dog need a health certificate?",
        "context": {"has_pets": True, "move_distance": "long_distance"},
        "category": "pets",
        "expected_topics": ["health", "certificate", "interstate"]
    },
    
    # Specialty items
    {
        "id": "specialty_1",
        "message": "I have a piano, how do I move it?",
        "context": {},
        "category": "specialty",
        "expected_topics": ["piano", "specialty", "mover"]
    },
    {
        "id": "specialty_2",
        "message": "What about my pool table?",
        "context": {},
        "category": "specialty", 
        "expected_topics": ["pool", "table", "disassembly"]
    },
    {
        "id": "specialty_3",
        "message": "I have antiques, should I be worried?",
        "context": {},
        "category": "specialty",
        "expected_topics": ["antique", "valuable", "insurance"]
    },
    
    # Apartments
    {
        "id": "apartment_1",
        "message": "I live in an apartment on the 5th floor",
        "context": {"dwelling_type": "apartment"},
        "category": "logistics",
        "expected_topics": ["elevator", "stairs", "floor"]
    },
    {
        "id": "apartment_2",
        "message": "Do I need to reserve the elevator?",
        "context": {"dwelling_type": "apartment"},
        "category": "logistics",
        "expected_topics": ["elevator", "reserve", "building"]
    },
    
    # Timeline/Urgency
    {
        "id": "timeline_1",
        "message": "I'm moving in 2 weeks, is that enough time?",
        "context": {"days_until_move": 14},
        "category": "timeline",
        "expected_topics": ["urgent", "time", "priority"]
    },
    {
        "id": "timeline_2",
        "message": "I'm moving next week, what do I do?",
        "context": {"days_until_move": 7},
        "category": "timeline",
        "expected_topics": ["urgent", "immediate", "priority"]
    },
    
    # Emotional/Stress
    {
        "id": "emotional_1",
        "message": "I'm so stressed about this move",
        "context": {},
        "category": "emotional",
        "expected_topics": ["stress", "overwhelm", "help"]
    },
    {
        "id": "emotional_2",
        "message": "I'm overwhelmed and don't know where to start",
        "context": {},
        "category": "emotional",
        "expected_topics": ["overwhelm", "start", "step"]
    },
    {
        "id": "emotional_3",
        "message": "This is all too much",
        "context": {},
        "category": "emotional",
        "expected_topics": ["support", "help", "break"]
    },
    
    # Risks/Concerns  
    {
        "id": "risk_1",
        "message": "What if the movers damage my stuff?",
        "context": {},
        "category": "risk",
        "expected_topics": ["damage", "insurance", "protection"]
    },
    {
        "id": "risk_2",
        "message": "How do I avoid getting scammed?",
        "context": {},
        "category": "risk",
        "expected_topics": ["scam", "verify", "dot"]
    },
    {
        "id": "risk_3",
        "message": "What's a binding estimate?",
        "context": {},
        "category": "knowledge",
        "expected_topics": ["binding", "estimate", "price"]
    },
    
    # Address changes
    {
        "id": "address_1",
        "message": "What addresses do I need to update?",
        "context": {},
        "category": "administrative",
        "expected_topics": ["address", "update", "change"]
    },
    {
        "id": "address_2",
        "message": "When should I forward my mail?",
        "context": {},
        "category": "administrative",
        "expected_topics": ["mail", "forward", "usps"]
    },
    
    # Complex/Compound
    {
        "id": "complex_1",
        "message": "I have a piano and two dogs, moving cross country next month",
        "context": {"has_pets": True, "move_distance": "cross_country"},
        "category": "complex",
        "expected_topics": ["piano", "pet", "interstate"]
    },
]

# ============================================================================
# RAG API TESTING
# ============================================================================

def test_rag_api(rag_url: str, test_case: Dict) -> Dict:
    """Test a single query against the RAG API."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{rag_url}/query",
            json={
                "query": test_case["message"],
                "context": test_case["context"],
                "max_results": 8
            },
            timeout=5
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "elapsed_ms": elapsed_ms,
                "node_count": 0,
                "formatted_context": ""
            }
        
        data = response.json()
        
        return {
            "success": data.get("success", False),
            "node_count": data.get("node_count", 0),
            "matched_nodes": data.get("matched_nodes", []),
            "formatted_context": data.get("formatted_context", ""),
            "elapsed_ms": elapsed_ms
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout",
            "elapsed_ms": 5000,
            "node_count": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "node_count": 0
        }

def run_rag_tests(rag_url: str) -> Dict:
    """Run all test cases against the RAG API."""
    print("\n" + "="*70)
    print("PEEZY RAG API BASELINE TEST")
    print("="*70)
    print(f"RAG URL: {rag_url}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("-"*70)
    
    results = []
    passed = 0
    failed = 0
    total_time = 0
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test['id']}: \"{test['message'][:50]}...\"")
        
        result = test_rag_api(rag_url, test)
        result["test_id"] = test["id"]
        result["category"] = test["category"]
        result["message"] = test["message"]
        
        total_time += result.get("elapsed_ms", 0)
        
        if result["success"] and result["node_count"] > 0:
            passed += 1
            status = "✅"
        elif result["success"]:
            # Success but no nodes matched - might be expected for emotional queries
            if test["category"] == "emotional":
                passed += 1
                status = "⚠️ (no nodes, but emotional query)"
            else:
                failed += 1
                status = "⚠️ (no nodes)"
        else:
            failed += 1
            status = f"❌ ({result.get('error', 'unknown error')})"
        
        print(f"   {status} {result['node_count']} nodes, {result['elapsed_ms']}ms")
        if result.get("matched_nodes"):
            print(f"   Nodes: {result['matched_nodes'][:3]}")
        
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(TEST_CASES)} ({100*passed/len(TEST_CASES):.1f}%)")
    print(f"Failed: {failed}/{len(TEST_CASES)}")
    print(f"Avg response time: {total_time/len(TEST_CASES):.0f}ms")
    
    # By category
    print("\nBy Category:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["success"] and (r["node_count"] > 0 or r["category"] == "emotional"):
            categories[cat]["passed"] += 1
    
    for cat, stats in sorted(categories.items()):
        pct = 100 * stats["passed"] / stats["total"]
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "rag_url": rag_url,
        "total_tests": len(TEST_CASES),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(TEST_CASES),
        "avg_response_ms": total_time / len(TEST_CASES),
        "results": results,
        "by_category": categories
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Peezy AI Baseline Test")
    parser.add_argument("--rag-url", required=True, help="URL of the RAG API")
    parser.add_argument("--output", default="baseline_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run RAG tests
    results = run_rag_tests(args.rag_url)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Print baseline metrics
    print("\n" + "="*70)
    print("BASELINE METRICS")
    print("="*70)
    print(f"RAG Pass Rate: {100*results['pass_rate']:.1f}%")
    print(f"Avg Response Time: {results['avg_response_ms']:.0f}ms")
    print("\nThese are your baseline numbers. After improvements, run this again to measure progress.")

if __name__ == "__main__":
    main()
