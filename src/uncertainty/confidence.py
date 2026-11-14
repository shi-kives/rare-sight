import math

def compute_margin(all_class_scores):
    if not all_class_scores:
        raise ValueError("all_class_scores cannot be empty.")

    scores = [float(score) for score in all_class_scores.values()]

    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Class scores must be finite.")

    scores.sort(reverse=True)

    if len(scores) < 2:
        return 1.0
    
    return float(scores[0] - scores[1])

def distance_flag(min_distance, threshold):
    min_distance = float(min_distance)
    threshold = float(threshold)

    if not math.isfinite(min_distance) or not math.isfinite(threshold):
        raise ValueError("Distance and threshold must be finite.")

    if threshold < 0:
        raise ValueError("Distance threshold cannot be negative.")
    
    return min_distance > threshold

def compute_retrieval_agreement(predicted_class, similar_cases):
    if not similar_cases:
        return 0.0

    matches = sum(1 for case in similar_cases if str(case['class']) == str(predicted_class))
    return float(matches / len(similar_cases))

def overall_uncertainty(confidence, margin, distance_flag, retrieval_agreement, agrees):
    weak_signals = []

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    if not 0 <= retrieval_agreement <= 1:
        raise ValueError("Retrieval agreement must be between 0 and 1.")

    if confidence < 0.5:
        weak_signals.append(f"low confidence: {confidence:.1%}")

    if margin < 0.2:
        weak_signals.append(f"low margin: {margin:.3f}. model is uncertain between top 2 classes.")

    if distance_flag:
        weak_signals.append("query too far from all prototypes. possibly a new class.")

    if retrieval_agreement < 0.4:
        weak_signals.append(f"mixed retrieval: {retrieval_agreement:.0%}")

    if not agrees:
        weak_signals.append("prototype and retrieval engine disagree on class prediction.")

    n_weak = len(weak_signals)

    if n_weak == 0:
        level = 'HIGH'

    elif n_weak == 1:
        level = 'MEDIUM'

    else:
        level = 'LOW'

    return {
        'level': level,
        'reasons': weak_signals,
        'score': n_weak
    }