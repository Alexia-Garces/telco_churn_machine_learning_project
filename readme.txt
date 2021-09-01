# Telco Churn Classification Project

## Project Overview
- Use classification methodologies to create a supervised machine learning model that accurately predicts churn.
- Explore data to gain insight and provide recommendations related to drivers of churn.
- My initial hypotheses are that contract type and payment type are drivers of churn.
 
## Plan
- [x] Create project goals.
- [x] Task out Data Science Pipeline workflow in Trello to provide transparency of progress and workflow to key stakeholders.
- [x] Create Readme file to explain project contents, plan, goals, and workflow.
- [x] Create helper fucntion files for the acquire, prepare, and explore phases of the pipeline.
- [x] Acquire, Prepare, and clean data.
- [x] Explore data to find key insights and recommendations
- [x] Validate and test hypothesis using statistical testing
- [x] Train 3 classificiation models, and choose best preforming model for production
- [x] Provide final thoughts and conclusions and recommendations for next steps

To view my complete Data Science Pipeline Project Plan check out my Trello board https://trello.com/b/ueVajFPq/data-science-pipeline 

## Data dictionary
New Feature  | Description   | Data Type
--|--|--
autopay    | indicates if the customer has autopay as a payment type | int64



Categorical Features   | Description |	Data Type
--|--|--
senior_citizen|	indicates if the customer is a senior citizen	|int64
dependents|	    indicates if the customer has dependents	|int64
phone_service|	indicates if the customer has phone service with Telco	| int64
multiple_lines |	indicates if the customer with phone service has multiple lines	| int64
online_security|	indicates if the customer has online security services |	int 64
online_backup|	indicates if the customer has online backup services |	int64
device_protection	| indicates if the customer has device protection services |	int64
tech_support |  indicates if the customer has tech support services |	int64
streaming_tv |	indicates if the customer has tv streaming services |	int64
streaming_movies |	indicates if the customer has movie streaming services |	int64
payment_type    | indicates the type of payment method a customer is using | int64
internet_service_type |	indicates which internet service (if any) the customer has |	int64
gender	|   indicates the the customers' gender identity |	uint8
contract_type | 	indicates the type of contract the customer has with Telco |	int64
auto_bill_pay |	indicates if the customer is enrolled in auto bill pay or not |	int64

Continuous Features | Description | Data Type
--|--|--
monthly_charges | how much a customer pays per month in dollars| float64
total_charges   | how much a customer has paid over the course of their tenure | float64
tenure          | how many months the customer has been with the company| int64

Other   | Description   | Data Type
--|--|--
churn   | indicates whether or not a customer churned | int64
customer_id | customer id number                       | object

## Ideas and Hypotheses
### Key Findings
- Is contract type a driver of churn?
    - Yes, there is a clear correlation between contract type and churn
- Does payment type drive churn?
    - Accoridng to chi2 testing there's evidence to suggest that autopayment is not independent of churn

<hr style="border-top: 10px groove #8b0aa5; margin-top: 1px; margin-bottom: 1px"></hr>

### Models
- Created 3 classification models (Decision tree, KNN, and Random Forest)
- I found that Model 1 (a Random Forest) performed best, using most the features in our dataset
    - Accuracy on validate was 80%
    - Recall on validate was 52%
    - Accuracy on Test: 80% accuracy, which was only slighly lower than our accuracy on validate set
    - Recall on Test: 55%

### Goals for future exploration
- Feature selection and engineering as it may enhance model preformance:
 - Splitting up the data further and running tests on only the Month to Month customers
 - Looking at a subset of data for those customers that have only been with us for less than a year as they less tenure, the higher the churn rate.
 - Looking at the reverse side of this would have value as well.
 - Dive into phone service as a subset of the data since it is our most popular service.

### Reccomendations
 - Put the model into production and begin targeting the customers flagged to churn with targeted marketing.
    - From the data we can see a correlation between month to month customers and churn:
        - Send out offers to month to month customers offering a discounted contract rate.
 - With the correlation of autopay to churn, I would recommend providing an incentive to sign up for autopay.
    - A small $5 dollar monthly discount would be a nice way to incentivize this.

## How to Recreate this Project
You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook.

1. Read the README.md
2. Download the aquire.py, prepare.py, explore.py and telco_classification_presentation.ipynb files into your working directory, or clone this repository 
3. Add your own env file to your directory. (user, password, host)
4. Run the telco_classification_presentation.ipynb notebook

### Skills Required
- Python
    - Pandas
    - Seaborn
    - Matplotlib
    - Numpy -Sklearn

- SQL

- Statistical Analysis
    - Descriptive Stats
    - Hypothesis Testing
    - T-test
    - Chi^2 Test

- Classifcation Modeling
    - Logistical Regression
    - Random Forest
    - KNN
    - Baseline Accuracy


### See my initial Telco Churn Project Analyzing Month to Month Customer Churn
- [First telco churn project](https://www.canva.com/design/DAEl7y3tGCQ/share/preview?token=7tpRZ2eY4-JIkOJ1YruwxQ&role=EDITOR&utm_content=DAEl7y3tGCQ&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton) on Canva