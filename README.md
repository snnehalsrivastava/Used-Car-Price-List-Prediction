# Used-Car-Price-List-Prediction

This project utilizes **Machine Learning** to **predict used car listing prices** based on historical data from Craigslist.org. It aims to resolve the ambiguity faced by both sellers and buyers in determining fair market value, thereby promoting a transparent and efficient pricing process.

## Project Description

The core objective of this study was to develop a tool capable of accurately predicting used car listing prices. This involved **intensive exploratory data analysis (EDA)** and comprehensive **data cleaning**. Various machine learning models were applied, with **XGBoost achieving the highest prediction accuracy at 89.15%** after hyperparameter tuning.

## Business Problem

The rapidly expanding used car market frequently suffers from a **lack of pricing transparency**.
*   **Sellers** often struggle to determine a fair market value, which can lead to overpricing (vehicles remaining on the market) or underpricing (selling for less than their worth). This also consumes significant time and effort.
*   **Buyers** face challenges in evaluating vehicle conditions, finding suitable options within their budget, and ensuring fair prices, potentially leading to costly repairs or overpayment. This tool aims to increase pricing transparency and reduce fraudulent activities.

A **valuation tool** that provides accurate and objective price assessments, considering factors like mileage, age, condition, and market trends, can mitigate these issues, leading to a more transparent and efficient buying and selling process. Similar efforts leveraging machine learning for used car valuation have been undertaken by companies like CarGurus, Edmunds, Vroom, and Autotrader, showing promising results with high accuracy rates.

## Dataset

The dataset used for this project was **scraped from Craigslist.org** and consists of historical listing prices of used cars.
*   **Size**: Initially comprised **426,880 rows and 26 features**.
*   **Features**: These include 14 categorical, 7 numerical, and 5 text data types. Prominent features include paint color, manufacturer, number of cylinders, and odometer readings.
*   **Timeframe**: The data spans from 1900 to 2022, though considerable data for analysis begins from 1990 onwards.
*   **Download**: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

## Data Preprocessing & Exploratory Data Analysis (EDA)

The project involved substantial data preprocessing due to the raw dataset's quality.
*   **Initial Analysis**: EDA was performed using data visualization techniques like the Missingno library for visualizing missing values and Sweetviz for generating detailed reports to understand data patterns, outliers, and relationships.
*   **Data Cleaning**:
    *   **Dropped irrelevant columns**: 'county' (no data), 'size' (67% missing values), 'URL', 'Region_URL', 'Image_URL', and 'VIN' were removed as they were not significant for price prediction or contained non-functional URLs.
    *   **Filtered data**: Rows with prices less than $500 were removed (considered unrealistic "dirty data"). Data listed before 1990 was also removed due to negligible records.
    *   **Missing Values**: Rows with null values present in more than 17 columns were dropped. Additionally, rows with null values for columns that initially had over 10% nulls (e.g., 'Type' after initial cleaning) were also removed. For electric cars, 0 was imputed for the 'cylinders' column where values were missing or non-zero. Despite attempts using algorithms like K-Prototype clustering and chi-squared tests, no clear pattern or relationship in the missingness of data could be found.
    *   **Outlier Handling**: Extreme price outliers (e.g., $3,736,928,711) and odometer values greater than 5,000,000 were identified and removed to enhance data quality.
    *   Attempts were made to extract information from the 'Description' column for imputing 'paint_color' and 'drive', but the randomness of the information led to unreliable imputation of "junk values".
*   **Data Split**: The dataset was divided into training, validation, and test sets using a 3-way split: 80% for train+validation and 20% for test, followed by splitting the train+validation set into 75% for training and 25% for validation.
*   **Feature Transformation**: A pipeline was defined to perform **one-hot encoding** on categorical columns, converting them into a numerical format suitable for machine learning models.

## Machine Learning Models

Several machine learning models were applied after data preprocessing to predict car listing prices.
*   **Linear Regression**: A statistical method modeling linear relationships.
*   **Decision Tree Regressor**: A model that recursively partitions data based on input features.
*   **Random Forest Regressor**: Combines multiple decision trees to improve accuracy and reduce overfitting.
*   **XGBoost (Extreme Gradient Boosting)**: An algorithm that sequentially adds decision trees, correcting errors of previous trees, incorporating regularization techniques for complexity.

## Analysis and Results

The performance of each model was evaluated using R-squared values and Root Mean Squared Error (RMSE).

| Machine Learning Model | RMSE Value | R2 Value |
| :--------------------- | :--------- | :------- |
| Linear Regression      | 6284.26    | 0.7114   |
| Decision Tree Regressor| 6124.77    | 0.8253   |
| Random Forest Regressor| 6756.71    | 0.7874   |
| **XGBoost**            | **4825.56** | **0.8915** |

*   **Linear Regression**: Achieved an R-squared value of **0.7114**.
*   **Decision Tree Regressor**: Showed signs of **overfitting**, with a high training R2 of 0.9999 but a lower validation R2 of **0.8253**.
*   **Random Forest Regressor**: Resulted in a validation R2 of **0.7874**. This model, too, did not handle outliers effectively.
*   **XGBoost Model**:
    *   Initial validation R2: **0.857**.
    *   After **hyperparameter tuning** using randomized search, XGBoost achieved the **highest accuracy** with a validation R2 of **0.8915** and an RMSE of 4825.56.

## Conclusion & Future Work

The developed ML model, particularly the **tuned XGBoost model with 89.15% accuracy**, can be transformed into a valuable **valuation tool**.
*   **Benefits**:
    *   **Platforms**: Enhances user experience, boosts engagement, and optimizes operations by standardizing valuation, reducing time spent negotiating prices.
    *   **Sellers**: Provides objective, market-driven suggested listing prices based on all relevant car features and historical data, saving time and effort.
    *   **Buyers**: Increases pricing transparency, reduces the risk of fraud, and facilitates more informed purchasing decisions.
*   **Implementation**: This tool can be integrated as a toolbar on websites or developed into a user-friendly mobile application.
*   **Limitations**: The study faced challenges with the raw dataset's "dirty data," a high number of null values, and numerous outliers which were minimized but not completely solved. Important features like MPG or number of doors were also missing. Some imputation methods may not be completely accurate.
*   **Improvements**: Future work could focus on utilizing **actual sales price data** instead of listing prices to further enhance prediction accuracy and reflect real transaction values.

## Technologies Used

*   **Python Libraries**:
    *   **Data Manipulation & Analysis**: Pandas, NumPy.
    *   **Data Visualization**: Seaborn, Matplotlib, Plotly.
    *   **Machine Learning**: Scikit-Learn (for model selection, preprocessing, metrics), XGBoost (for the high-performing model), NLTK (for text processing in 'description' column exploration, specifically `word_tokenize`, `stopwords`, `WordNetLemmatizer`, `string`).
    *   **Clustering**: KPrototypes (for missing values analysis).
    *   **Statistical Analysis**: Scipy.

## Contributors

*   Snnehal Srivastava
*   Rachit Jain
*   Riya Mangal
*   Dhruv
*   Chirag Murali
