🏠 I Built an ML Model to Predict Apartment Prices in Egypt — Achieved R² = 0.9730!

For my Machine Learning final project at the Faculty of Computers & Artificial Intelligence, Menoufia National University (under Dr. Heba Omara), I built a complete end-to-end regression pipeline on a real Egyptian real estate dataset.

The result? The winning model explains 97.3% of price variance 🎯

📊 Dataset:

Started with 1,008 apartment listings → cleaned down to 978 records
Features: Area (m²), Bedrooms, Property Age, Floor, Metro Distance, Garage, Garden, Bathrooms, and more
⚙️ Full Pipeline — 11 Stages:

1️⃣ EDA (Exploratory Data Analysis) Found that Area (m²) had a 0.98 correlation with price — by far the strongest predictor in the entire dataset.

2️⃣ Missing Value Imputation

Area: 30 missing values
Bedrooms: 25 missing values
Property Age: 20 missing values
Floor: 15 missing values
✅ Solution: Median Imputation applied per feature
3️⃣ Outlier Removal Applied IQR Method to detect and remove anomalous entries that would distort model learning.

4️⃣ Feature Scaling StandardScaler to normalize all numeric features to a common scale.

5️⃣ Train/Test Split 80/20 split for an objective, unbiased evaluation.

🤖 Core Phase: Training & Comparing 6 Models:

Model	R²	MAE
🏆 Lasso Regression	0.9730	~24k EGP
Linear Regression	0.9727	~24k EGP
Ridge (GridSearch)	0.9727	~24k EGP
Ridge Regression	0.9612	~45k EGP
ElasticNet (GridSearch)	0.1652	~267k EGP
ElasticNet	0.0007	~292k EGP
🔬 Hyperparameter Tuning with GridSearchCV:

Ridge GridSearch: Best Alpha = 0.1 — any higher value dramatically degrades performance (as shown clearly in the Alpha vs R² curve)
ElasticNet GridSearch: Completely failed even after tuning — R² = 0.1652 only
💡 Key Insight: Not every regularization technique works for every dataset. L1+L2 combined (ElasticNet) was overkill for this relatively linear dataset.

📈 Winner: Lasso Regression

✅ R² = 0.9730 — explains 97.3% of price variance
✅ MAE ≈ 24,000 EGP — average prediction error
✅ Why Lasso won: L1 regularization automatically zeroed out weak features, effectively performing built-in Feature Selection — no manual feature dropping needed!
📊 MLflow Experiment Tracking: Logged all 6 models with their metrics, parameters, and artifacts in MLflow — enabling clean reproducibility and scientific model comparison.

bash
$ mlflow ui
# Open: http://localhost:5000
🌐 Interactive Web Application: Beyond the model, I built a fully interactive web app in HTML/CSS/JavaScript that runs entirely in the browser — zero API, zero backend, zero server needed!

Features:

🎚️ Sliders for Area, Bedrooms, Floor, and Property Age
🗺️ City & finishing type selector
📊 Real-time model performance visualization
💡 Live feature importance display
🏠 Instant price prediction in Egyptian Pounds (EGP)
Deployable as a 100% static page — no infrastructure required.

🛠️ Tech Stack: Python | Pandas | NumPy | Scikit-Learn | Matplotlib | Seaborn | MLflow | HTML | CSS | JavaScript

🎓 Key Takeaways:

A 0.98 correlation for area proves that real insight starts with genuinely understanding your data
Lasso doesn't just improve accuracy — it performs automatic Feature Selection
ElasticNet needs more complex, multi-collinear datasets to shine
MLflow transforms experiment chaos into reproducible, organized science
