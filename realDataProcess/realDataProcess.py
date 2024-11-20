import pandas as pd
import string
import contractions

# Step 1: Read the data
file_path = "E-Commerce_Reviews.csv"  # Replace with your data file path
data = pd.read_csv(file_path)

# Step 2: Remove rows with missing values
data = data.dropna()
print("---Finish step 2")

# Step 3: Select 4000 samples with a 1:1 balance between recommended and not recommended
recommended = data[data["Recommended IND"] == 1]
not_recommended = data[data["Recommended IND"] == 0]
recommended_sample = recommended.sample(n=2000, random_state=42)
not_recommended_sample = not_recommended.sample(n=2000, random_state=42)
balanced_data = pd.concat([recommended_sample, not_recommended_sample])
balanced_data = balanced_data.sample(frac=1, random_state=42)
print("---Finish step 3")

# Step 4: Expand contractions
def expand_contractions(text):
    return contractions.fix(text)
balanced_data["Review Text"] = balanced_data["Review Text"].apply(expand_contractions)
print("---Finish step 4")

# Step 5: Remove unnecessary punctuation, exclude .,!?, and lower letter 
punctuation = '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'  # 
def clean_text(text):
    if isinstance(text, str):
        clean_list = [x for x in text if x not in punctuation]
        cleaned_text = ''.join(clean_list).replace('\n', '').replace('\r', '')
        return cleaned_text.lower()
    return text
balanced_data["Review Text"] = balanced_data["Review Text"].apply(clean_text)
print("---Finish step 5")

# Step 6: Keep only necessary columns
columns_to_keep = ["Age", "Review Text", "Rating", "Recommended IND", "Positive Feedback Count"]
cleaned_data = balanced_data[columns_to_keep]
print("---Finish step 6")

# Step 7: Save the processed data to a new CSV file
output_file = "processed_real_data.csv"
cleaned_data.to_csv(output_file, index=False)
print("---Finish step 7")

print(f"Data processing completed. The cleaned data is saved to {output_file}")
