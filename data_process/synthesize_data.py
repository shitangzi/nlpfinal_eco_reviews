import pandas as pd
import random
import numpy as np
import contractions

# Step 1: load corpus
def load_and_group_corpus(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=["Review Text", "Rating"])

    positive_reviews = data[data["Rating"] >= 4]["Review Text"].tolist()
    negative_reviews = data[data["Rating"] <= 2]["Review Text"].tolist()
    neutral_reviews = data[data["Rating"] == 3]["Review Text"].tolist()
    
    # filter out reviews fewer than 3 words, because we use 2-dim Markov Chian
    positive_reviews = [review for review in positive_reviews if len(review.split()) >= 3]
    negative_reviews = [review for review in negative_reviews if len(review.split()) >= 3]
    neutral_reviews = [review for review in neutral_reviews if len(review.split()) >= 3]
    return positive_reviews, negative_reviews, neutral_reviews

# Step 2: build a Markov Chain for each sentiment
def build_markov_chain(corpus):
    markov_chain = {}
    for review in corpus:
        words = review.split()
        for i in range(len(words) - 2):
            bigram = (words[i], words[i + 1])
            next_word = words[i + 2]
            if bigram not in markov_chain:
                markov_chain[bigram] = []
            markov_chain[bigram].append(next_word)
    return markov_chain

# Step 3: generate a sentence using the Markov Chain
def generate_sentence(markov_chain, max_length=25):
    if not markov_chain:
        return "No data available"
    start_word = random.choice(list(markov_chain.keys()))
    sentence = list(start_word)
    
    for _ in range(max_length - 2):
        if start_word not in markov_chain or not markov_chain[start_word]:
            break
        next_word = random.choice(markov_chain[start_word])
        sentence.append(next_word)
        start_word = (start_word[1], next_word)
    return " ".join(sentence)

# Step 4: generate reviews for a specific sentiment
def generate_reviews(markov_chain, num_reviews=10, max_length=20):
    reviews = []
    for _ in range(num_reviews):
        reviews.append(generate_sentence(markov_chain, max_length))
    return reviews

# Step 5: generate synthetic data
def generate_synthetic_data(file_path, total_samples=5000):
    positive_reviews, negative_reviews, neutral_reviews = load_and_group_corpus(file_path)
    positive_chain = build_markov_chain(positive_reviews)
    negative_chain = build_markov_chain(negative_reviews)
    neutral_chain = build_markov_chain(neutral_reviews)
    num_positive = total_samples // 2
    num_negative = total_samples // 2
    num_neutral = int(total_samples * 0.05)  

    data = []

    # generate positive reviews
    for _ in range(num_positive):
        review_text = generate_sentence(positive_chain)
        rating = random.choice([4, 5])
        recommended = 1
        feedback_count = np.random.randint(0, 20)
        age = np.random.randint(18, 70)
        data.append({
            "Age": age,
            "Review Text": review_text,
            "Rating": rating,
            "Recommended IND": recommended,
            "Positive Feedback Count": feedback_count
        })
    print("finish generating positive reviews")

    # generate negative reviews
    for _ in range(num_negative):
        review_text = generate_sentence(negative_chain)
        rating = random.choice([1, 2])
        recommended = 0
        feedback_count = np.random.randint(0, 20)
        age = np.random.randint(18, 70)
        data.append({
            "Age": age,
            "Review Text": review_text,
            "Rating": rating,
            "Recommended IND": recommended,
            "Positive Feedback Count": feedback_count
        })
    print("finish generating negative reviews")

    # generate neutral reviews
    for _ in range(num_neutral):
        review_text = generate_sentence(neutral_chain)
        rating = 3
        recommended = random.choice([0, 1])  # natural reviews may or may not recommend
        feedback_count = np.random.randint(0, 20)
        age = np.random.randint(18, 70)
        data.append({
            "Age": age,
            "Review Text": review_text,
            "Rating": rating,
            "Recommended IND": recommended,
            "Positive Feedback Count": feedback_count
        })
    print("finish generating neutral reviews")

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
    return df

# Step 6: expand contractions
def expand_contractions(text):
    return contractions.fix(text)
# balanced_data["Review Text"] = balanced_data["Review Text"].apply(expand_contractions)
# print("---Finish step 4")

# Step 7: remove unnecessary punctuation, exclude .,!?, and lower letter 
punctuation = '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'  # 
def clean_text(text):
    if isinstance(text, str):
        clean_list = [x for x in text if x not in punctuation]
        cleaned_text = ''.join(clean_list).replace('\n', '').replace('\r', '')
        return cleaned_text.lower()
    return text
# balanced_data["Review Text"] = balanced_data["Review Text"].apply(clean_text)
# print("---Finish step 5")

# Main execution
if __name__ == "__main__":
    file_path = "E-Commerce_Reviews.csv"  
    synthetic_data = generate_synthetic_data(file_path, total_samples=5000)
    synthetic_data["Review Text"] = synthetic_data["Review Text"].apply(expand_contractions)
    synthetic_data["Review Text"] = synthetic_data["Review Text"].apply(clean_text)
    synthetic_data.insert(0, "", range(0, len(synthetic_data))) 
    output_file = "synthetic_reviews.csv"
    synthetic_data.to_csv(output_file, index=False)
    print(f"Synthetic data saved to '{output_file}'")
