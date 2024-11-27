# GRU-based Sentiment Analysis for E-commerce Reviews README
**Author**: Sitang Gong (sg669), Jieying Zhang (z450), Chelsea Lyu (jl1230) 
## Project Overview
The goal of this project is to analyze user reviews and classify their sentiment to determine if a
product is recommended by users. We built a GRU-based model to analyze e-commerce reviews and determine whether
users recommend a product. By using product ratings as additional input and training the model
on a larger dataset, we improved its ability to capture patterns in the reviews. 

## Data Process Folder
### CSV file:
* **E-Commerce_Reviews.csv**: The raw dataset consisting of e-commerce product reviews. This is the starting point for data processing.
* **processed_real_data.csv**:
A smaller processed dataset with 5,000 records derived from the raw e-commerce review data. 
* **processed_real_data_large.csv**: A larger processed dataset with 10,000 records derived from the raw e-commerce review data.
* **synthetic_reviews.csv**: A dataset of synthetically generated reviews.
* **20randomcases.csv**: A small dataset containing 20 cases randomly sampled from raw e-commerce reviews for quick testing and validation.

### Python Scripts:
* **real_data_process.py**: The script for processing and cleaning real data.
* **synthesize_data.py**: The script for synthesizing training data.
* **generate_test_cases.py**: The scirpt for selecting 20 random cases from raw e-commerce review data, and saving to 20randomcases.csv.

### Usage Instruction:
Run `python3 real_data_process.py` to process the raw E-Commerce_Reviews.csv file, and generate small/large dataset. User can change `n` number in line 16-17 to control the number of the Recommended and Not Recommended records in processed csv file.

Use `python3 synthesize_data.py` to create synthetic datasets.

Run `python3 generate_test_cases.py` to create the 20randomcases.csv file.

## Train Folder
### ipynb file:
* **train generate GRU**: Training on synthesized data without additional features.
* **train generate GRU rate**: Training on synthesized data with rating as an auxiliary input.
* **train real GRU**: Training on the smaller real dataset without rating.
* **train real GRU large**: Training on the larger real dataset without rating.
* **train real GRU rate**: Training on the smaller real dataset with rating.
* **train real GRU rate large**: Training on the larger real dataset with rating.

### Usage Instruction
We used Google Colab's GPU for training, so we ran out code on the Google Colab platform. Our approach was to upload all the `.ipynb` files to Colab and place all the CSV files generated and processed by `data_process` in the same folder as the code files. 

Then, change the file path in the third cell to make sure that the code can read the cvs file, and simply click `Run All` in each `ipynb` file to execute the entire code.

