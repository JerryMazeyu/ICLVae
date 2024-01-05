from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def get_attitude(answer_list):
    """
    Get attitude(Positive / Negative) from n answers.
    """
    positive_score = 0
    negative_score = 0
    for ans in answer_list:
        if "positive" in ans.lower() and "negative" in ans.lower():
            continue
        elif "positive" in ans.lower():
            positive_score += 1
        elif "negative" in ans.lower():
            negative_score += 1
        else:
            continue
    if positive_score + negative_score == 0:
        return -1
    else:
        return 1 if positive_score >= negative_score else 0

def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0, 1]

def get_attitude_score(answer_list):
    """
    Get attitude score from n answers.
    """
    positive_score = 0
    negative_score = 0
    for ans in answer_list:
        if "positive" in ans.lower() and "negative" in ans.lower():
            continue
        elif "positive" in ans.lower():
            positive_score += calculate_cosine_similarity(ans, "This comment is positive.")
        elif "negative" in ans.lower():
            negative_score += calculate_cosine_similarity(ans, "This comment is negative.")
    return negative_score, positive_score





