# NIQ
# NIQ
NIQ: Nutritional Metadata Classification on Open Food Facts Dataset
This project develops a machine learning pipeline for food product classification and nutritional analysis using the Open Food Facts dataset. It focuses on four tasks: rule-based token tagging, food category classification, nutritional entity tagging, and attribute prediction, leveraging text features (ingredients_text) and structured data for health-aware applications.
Project Overview
The pipeline processes the Open Food Facts dataset to:

Analyze missing values and visualize metadata distributions.
Tag ingredients with semantic labels (e.g., NUT, QTY).
Classify food categories (pnns_groups_1, pnns_groups_2) using an LSTM model.
Extract nutritional entities (e.g., "Protein: 10 g") via BIO tagging.
Predict numerical (e.g., energy_100g) and categorical (e.g., nutriscore_grade) attributes.

The project includes data preprocessing, model training, and evaluation, with results documented in main2 (1).tex.
File Descriptions and Functionality

cleaned_openfoodfacts_subset.csvA subset of the Open Food Facts dataset used as input, containing product metadata, nutritional values, and ingredient lists.

clean_openfoodfacts_dynamic_columns_with_types.pyGenerates value count analyses for key metadata columns (categories, brands, countries, main_category) and saves a 2x2 grid plot as top_10_distribution.png.

clean_openfoodfacts_dynamic_columns_with_types2.pyExtends the above script to include label distribution analysis for pnns_groups_1 and pnns_groups_2, saving plots as top_10_metadata_distribution.png and label_distributions.png.

imputation3.pyPerforms missing value imputation on numerical (median) and categorical (most frequent) columns, saving summaries (summary_before_imputation.csv, summary_after_imputation.csv) and the imputed dataset (imputed_openfoodfacts_median_most_frequent_.csv).

entityTagging2.pyImplements rule-based token tagging on ingredients_text, assigning semantic tags (e.g., NUT, FLA) using lexicons and regex. Outputs tagged data to tagged_ingredient_output_with_pairs.csv.

LSTM2.pyTrains an LSTM model (LSTMG1) for food category classification, using GloVe embeddings and tag counts. Outputs a classification report with accuracy and F1-scores.

NutritionalEntityClassifier.pyTrains a BiLSTM model for nutritional entity tagging, converting tags to BIO format and predicting entities like "Protein: 10 g". Outputs F1-score and classification report.

NutritionalEntityClassifier1.pyTrains a BiLSTM model for nutritional attribute prediction, predicting numerical (e.g., energy_100g) and categorical (e.g., nutriscore_grade) attributes. Outputs RMSE for numerical predictions and F1-score for categorical predictions.

main2 (1).texA LaTeX report documenting the project, including data analysis, preprocessing, model details, evaluation metrics, and conclusions.

README.mdThis file, providing project overview, file descriptions, and execution instructions.


Sequence of Execution
Follow this sequence to run the pipeline:

Data Analysis and Visualization  

Run clean_openfoodfacts_dynamic_columns_with_types.py to generate metadata distribution plots.
Run clean_openfoodfacts_dynamic_columns_with_types2.py to generate metadata and label distribution plots.


Data Imputation  

Run imputation3.py to impute missing values in the dataset, generating imputed_openfoodfacts_median_most_frequent_.csv.


Rule-Based Token Tagging  

Run entityTagging2.py to tag ingredients, producing tagged_ingredient_output_with_pairs.csv.


Food Category Classification  

Run LSTM2.py to train an LSTM model for classifying pnns_groups_1 and pnns_groups_2.


Nutritional Entity Tagging  

Run NutritionalEntityClassifier.py to train a BiLSTM model for BIO tagging of nutritional entities.


Nutritional Attribute Prediction  

Run NutritionalEntityClassifier1.py to train a BiLSTM model for predicting numerical and categorical attributes.


Documentation  

Compile main2 (1).tex using a LaTeX compiler (e.g., pdflatex) to generate the project report.



Execution Instructions
Prerequisites

Python 3.8+ and required libraries:  pip install pandas numpy seaborn matplotlib torch spacy sklearn nltk


spaCy Model:  python -m spacy download en_core_web_trf


NLTK Data: The scripts automatically download punkt to /cluster/datastore/aniketag/allData/nltk_data. Ensure this path exists or modify the path in the scripts.
GloVe Embeddings: Download glove.6B.300d.txt from Stanford NLP and place it at /cluster/datastore/aniketag/allData/NIQ/glove.6B.300d.txt.
Dataset: Ensure cleaned_openfoodfacts.csv is at /cluster/datastore/aniketag/NIQ/. The provided subset (cleaned_openfoodfacts_subset.csv) can be used for testing.
LaTeX Compiler: Install a LaTeX distribution (e.g., TeX Live) to compile main2 (1).tex.

Steps to Run

Clone the Repository  
git clone <repository-url>
cd <repository-name>


Set Up Paths  

Update CSV_PATH in all Python scripts to point to your cleaned_openfoodfacts.csv location.
Ensure glove_path in LSTM2.py, NutritionalEntityClassifier.py, and NutritionalEntityClassifier1.py points to glove.6B.300d.txt.
Create a genData directory for outputs:  mkdir genData
mkdir genImages




Run Scripts in Sequence  

Data Analysis:  python clean_openfoodfacts_dynamic_columns_with_types.py
python clean_openfoodfacts_dynamic_columns_with_types2.py


Imputation:  python imputation3.py


Tagging:  python entityTagging2.py


Classification:  python LSTM2.py


Entity Tagging:  python NutritionalEntityClassifier.py


Attribute Prediction:  python NutritionalEntityClassifier1.py




Compile the Report  
pdflatex main2\(1\).tex
pdflatex main2\(1\).tex  # Run twice to resolve references



Notes

Ensure sufficient disk space for intermediate files (e.g., genData/tagged_ingredient_output_with_pairs.csv).
Scripts may require GPU access for faster training; modify device in scripts if needed.
The LaTeX report (main2 (1).tex) references image files (e.g., top_10_metadata_distribution.png). Ensure these are generated or update paths accordingly.

License
This project is licensed under the MIT License. See the LICENSE file for details.
