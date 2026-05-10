# Singapore Talent Market Intelligence — Team 5, Persona C

This repository contains Team 5’s NumPy, Pandas and EDA hackathon project for the **Singapore Talent Market Intelligence** dataset.

Our analysis focuses on **Persona C: The Government Workforce Planner**. The objective is to identify sectors and job roles in Singapore that may face structural hiring difficulty, so workforce planners can better prioritise retraining grants and labour-market interventions.

## Project Focus

We analysed Singapore job posting data from October 2022 to May 2024 to answer the following questions:

1. Which sectors show the highest hiring difficulty?
2. Does position level affect hiring difficulty?
3. Which roles have high vacancies but low application response?
4. What are the top priority sectors for workforce intervention?

## Method Summary

The project uses Python-based exploratory data analysis, including:

- Data cleaning and validation
- Date conversion and missing-value checks
- Salary outlier handling
- Category and position-level analysis
- A custom hiring difficulty score using:
  - `metadata_repostCount`
  - `numberOfVacancies`
  - `metadata_totalNumberJobApplication`

Higher repost counts, more vacancies and lower application response contribute to a higher hiring difficulty score.

## Repository Contents

- `DS5 - NumPy Pandas EDA Hackathon - Team 5 - Persona C.ipynb`  
  Main analysis notebook containing the data cleaning, scoring logic and visualisations.

- `Team5 - workforce_hiring_difficulty_final.pptx`  
  Final presentation deck summarising the key insights and policy recommendations.

- `README.md`  
  Project overview and repository guide.

## Key Insight

The analysis suggests that hiring difficulty is not limited to senior roles. Some entry-level and non-executive positions also show signs of structural hiring challenges, highlighting the need for targeted workforce planning beyond traditional assumptions.

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook

## Project Context

This project was completed as part of the **DS5 NumPy, Pandas and EDA Hackathon** under the NTU Data Science and AI programme.
