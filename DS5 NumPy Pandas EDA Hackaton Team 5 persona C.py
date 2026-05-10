# %% [markdown]
# <span style="font-size:22px; font-weight:bold">DS5 - NumPy / Pandas / EDA Hackathon — Singapore Talent Market Intelligence</span>
# 
# **Team 5 members** Fiona Lim, Cheah Peng Huat (Caleb), Ng Zhenxian, Kevin Chua, Soh Shi Hui
# 
# **Persona C — The Government Workforce Planner**
# 
# "The Ministry of Manpower wants to know which sectors face structural hiring difficulty so we can target retraining grants."
# 
# ### Research questions
# - Which sectors have the highest hiring difficulty?
# - Does position level affect hiring difficulty?
# - Which roles have high vacancies but low applicants?
# - What are the top 5 hardest-to-fill sectors?
# 
# 
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# %%
# Load — this may take 10–15 seconds on the first run
df = pd.read_csv('/home/ssh/SGJobData/SGJobData.csv', low_memory=False)


# %%
# Sanity checks
print(df.shape)          # Expect (~1048585, 20)
print(df.dtypes)
print(df.head(3))


# %%
df['metadata_originalPostingDate'] = pd.to_datetime(df['metadata_originalPostingDate'])
df['metadata_newPostingDate']       = pd.to_datetime(df['metadata_newPostingDate'])
df['metadata_expiryDate']           = pd.to_datetime(df['metadata_expiryDate'])



# %%
# Shape, types, missing values at a glance
print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
print("\nNull counts:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print("\nBasic stats:\n", df[['salary_minimum','salary_maximum','average_salary',
                               'minimumYearsExperience','numberOfVacancies']].describe())


# %%
# Remove obvious outliers using the 99th percentile
p99 = np.percentile(df['average_salary'].dropna(), 99)
df_clean = df[(df['average_salary'] > 500) & (df['average_salary'] <= p99)].copy()

# Use NumPy to compute stats on the cleaned array
sal = df_clean['average_salary'].to_numpy()
print(f"Mean: {np.mean(sal):,.0f}  Median: {np.median(sal):,.0f}  Std: {np.std(sal):,.0f}")
print(f"25th pct: {np.percentile(sal, 25):,.0f}  75th pct: {np.percentile(sal, 75):,.0f}")


# %%
salary_by_level = (
    df_clean
    .groupby('positionLevels')['average_salary']
    .agg(['mean', 'median', 'count'])
    .rename(columns={'mean': 'avg_salary', 'median': 'median_salary', 'count': 'num_jobs'})
    .sort_values('median_salary', ascending=False)
)
print(salary_by_level)


# %%
import re

def extract_first_category(cat_str):
    """Extract the first category label from the JSON-like string."""
    if pd.isna(cat_str):
        return np.nan
    match = re.search(r'"category"\s*:\s*"([^"]+)"', str(cat_str))
    return match.group(1) if match else np.nan

df_clean['primary_category'] = df_clean['categories'].apply(extract_first_category)
print(df_clean['primary_category'].value_counts().head(10))


# %%
df_clean['year_month'] = df_clean['metadata_originalPostingDate'].dt.to_period('M')
monthly = df_clean.groupby('year_month').size().reset_index(name='postings')

plt.figure(figsize=(12, 4))
plt.plot(monthly['year_month'].astype(str), monthly['postings'])
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Job Postings — SGJobData')
plt.tight_layout()
plt.show()


# %%
agency_keywords = ['RECRUIT', 'HR ADVISORY', 'MANPOWER', 'STAFFING', 'CONSULT', 'TALENT']
pattern = '|'.join(agency_keywords)

direct_only = df_clean[
    (~df_clean['postedCompany_name'].str.upper().str.contains(pattern, na=False)) &
    (df_clean['metadata_isPostedOnBehalf'] == False)
]


# %%
num_cols = ['average_salary', 'minimumYearsExperience', 'numberOfVacancies',
            'metadata_repostCount', 'metadata_totalNumberJobApplication',
            'metadata_totalNumberOfView']
corr = df_clean[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix — Numerical Columns')
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean, x='positionLevels', y='average_salary')
plt.xticks(rotation=45, ha='right')
plt.title('Salary Distribution by Position Level')
plt.tight_layout()
plt.show()



# %%
#1. What is the shape of the DataFrame, the date range of metadata_originalPostingDate, and how many unique job post IDs are there?
print(df_clean.shape)
print(df_clean['metadata_originalPostingDate'].min(), df_clean['metadata_originalPostingDate'].max())
print(df_clean['metadata_jobPostId'].nunique())
# print("Unique Job Post IDs:", df_clean['metadata_jobPostId'].unique()) 
# this will print all unique job post IDs, which may be a very long list. 
#Consider printing only the count or a sample if there are too many.

# %%
#2. What are the top 10 most common primary_category values, and 
#what percentage of all jobs does each represent? (Use .value_counts(normalize=True))

counts=df_clean['primary_category'].value_counts().head(10)

# 2. Get the percentage (normalize=True gives a 0-1 decimal, so we multiply by 100)
percentages = df_clean['primary_category'].value_counts(normalize=True).head(10) * 100

Top_10_most_common_categories = pd.DataFrame({"Count": counts,
                                              "Percentage": percentages})
print(Top_10_most_common_categories)

# %%
#3. After removing salary outliers, what is the median average_salary broken down by positionLevels? 
# #Sort from highest to lowest.

salary_by_level = (
    df_clean
    .groupby('positionLevels')['average_salary']
    .agg(['mean', 'median', 'count'])
    .rename(columns={'mean': 'avg_salary', 'median': 'median_salary', 'count': 'num_jobs'})
    .sort_values('median_salary', ascending=False)
)
print(salary_by_level)


# %%
#4. Which 5 columns have the most missing values? What percentage of rows are null in each?
null_counts = df_clean.isnull().sum().sort_values(ascending=False).head(5)
null_percentages = (df_clean.isnull().mean() * 100).sort_values(ascending=False).head(5)
missing_data_summary = pd.DataFrame({
    'Null Count': null_counts,
    'Null Percentage': null_percentages
})
print(missing_data_summary)



# %%
# What is the distribution of minimumYearsExperience? Plot a histogram. Where does the bulk of demand sit?

df_filtered = df_clean[df_clean['minimumYearsExperience'] <= 30] #remove 80-year outliers from calculations entirely
plt.figure(figsize=(10, 5)) #the numbers 10 and 5 represent the width and height of plot in inches.
sns.histplot(df_filtered['minimumYearsExperience'].dropna(), bins=30, kde=False) 
# dropna() removes missing values, bins=30 creates 30 bars, kde=False turns off the density curve.

plt.xlim(0, 30) #sets the limits of the x-axis from 0 to 30 years of experience. Adjust as needed based on data.
plt.title('Distribution of Minimum Years of Experience') #sets the title of the plot.
plt.xlabel('Minimum Years of Experience') #sets the label for the x-axis.
plt.ylabel('Number of Job Postings') #sets the label for the y-axis.
plt.tight_layout() #adjusts the layout to prevent overlap of elements.
plt.show() #displays the plot on the screen.

# The bulk of demand sits between 0-5 years of experience, with a significant drop-off after 10 years. 

# %%
# Compute the correlation between average_salary, minimumYearsExperience, metadata_repostCount, and metadata_totalNumberJobApplication. 
# Which pair has the strongest relationship?

num_cols = ['average_salary', 'minimumYearsExperience', 'metadata_repostCount', 'metadata_totalNumberJobApplication']
corr = df_clean[num_cols].corr()
print(corr)


# %% [markdown]
# Persona C — The Government Workforce Planner
# "The Ministry of Manpower wants to know which sectors face structural hiring difficulty so we can target retraining grants."
# 
# Create a "difficulty score" using NumPy: combine metadata_repostCount, numberOfVacancies, and metadata_totalNumberJobApplication into a single normalised index (hint: (x - x.min()) / (x.max() - x.min())).
# 

# %% [markdown]
# **Qn 1 - Which sectors have the highest hiring difficulty based on repost rate, vacancies, and low applications?**
# 
# hiring difficulty score = (repost_norm + vacancies_norm + apps_norm) / 3
# 
# Results:
# Public / Civil Service 0.33509

# %%
# 1. Normalize the individual components (0 to 1 scale)
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

repost_norm = normalize(df_clean['metadata_repostCount'])
vacancies_norm = normalize(df_clean['numberOfVacancies'])
# Inverse normalization for applications: 1 = few apps (difficult), 0 = many apps (easy)
apps_norm = 1 - normalize(df_clean['metadata_totalNumberJobApplication'])

# 2. Combine into a single Difficulty Score (Average of the three)
# You can weight these if you think Reposts are more important than Vacancies
df_clean['hiring_difficulty_score'] = (repost_norm + vacancies_norm + apps_norm) / 3




# %% [markdown]
# **QN 2 - Does Position Level Affect Difficulty?**
# 
# Answer: Yes. Lower-level roles such as Non-executive and Fresh/entry level have higher difficulty scores than management roles. 
# 
# This suggests hiring difficulty is not only a senior-skills issue; it also affects frontline and entry-level manpower needs.

# %%
# Which categories and position levels have the highest difficulty scores?

# 3. View the sectors with the highest structural difficulty
difficulty_by_primary_category= df_clean.groupby('primary_category')['hiring_difficulty_score'].mean().sort_values(ascending=False)
difficulty_by_position_level = df_clean.groupby('positionLevels')['hiring_difficulty_score'].mean().sort_values(ascending=False)

print(difficulty_by_primary_category.head(10).round(4),"\n" + "="*50)
print(difficulty_by_position_level.head(10).round(4))



# %% [markdown]
# make the table more readable 
# 1. Convert the Series results into DataFrames
# df_sector_difficulty = difficulty_by_primary_category.head(10).reset_index()
# df_level_difficulty = difficulty_by_position_level.head().reset_index()
# 
# 2. Rename the columns so they look professional (optional)
# df_sector_difficulty.columns = ['Primary Category', 'Difficulty Score']
# df_level_difficulty.columns = ['Position Level', 'Difficulty Score']
# 
# 3. Print them (Pandas will now format them as tables)
# print("TOP 10 DIFFICULT SECTORS")
# print(df_sector_difficulty)
# print("\n" + "="*50 + "\n")
# 
# print("DIFFICULTY BY POSITION LEVEL")
# print(df_level_difficulty)

# %% [markdown]
# **Qn 3 - Are there roles with many vacancies but few applicants (supply–demand mismatch)?**
# General management

# %%
# Are there roles with high vacancy counts but very low application rates (applications ÷ views)? Visualise with a scatter plot.


# 1. Calculate Application Rate (Apps / Views). This metric tells us what percentage of people who saw the job actually clicked "Apply."
# We add a tiny epsilon or use fillna to avoid division by zero
df_clean['app_rate'] = df_clean['metadata_totalNumberJobApplication'] / df_clean['metadata_totalNumberOfView']

# Replace infinite values (from division by zero) or NaNs with 0
df_clean['app_rate'] = df_clean['app_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 2. Filter for high vacancy roles to keep the plot clean
# jobs with more than 100 vacancies
high_vacancy_df = df_clean[df_clean['numberOfVacancies'] > 100].copy()

# 3. Scatter plot: Vacancies vs Application Rate
plt.figure(figsize=(14,8))

# 1. Add hue to color dots by category
# We remove the 'color' argument because 'hue' will now control the colors
ax = sns.scatterplot(
    data=high_vacancy_df, 
    x='app_rate', 
    y='numberOfVacancies',
    hue='primary_category', 
    alpha=1,
    palette='tab20' # A professional color palette
)

# 2. Limit the view to the 'Problem Zone' (0% to 10% app rate & vacancies >100)
plt.xlim(-0.001, 0.10)
plt.ylim(0, high_vacancy_df['numberOfVacancies'].max()-100 + 5)

# 3. Move the legend outside the plot so it doesn't cover the data
plt.legend(title='Job Categories', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.title('Hiring Difficulty: Vacancies vs. App Rate by Sector', fontsize=15)
plt.xlabel('Application Rate (Apps / Views)')
plt.ylabel('Number of Vacancies')

plt.tight_layout() # Essential when moving legends outside
plt.show()

# %% [markdown]
# **Qn 4 - Top 5 Hardest to Fill Sectors**

# %% [markdown]
# Plot monthly metadata_repostCount trends by category for the top 5 hard-to-fill sectors.
# 
# The Strategy
# Identify the Top 5: We use the hiring_difficulty_score we calculated earlier.
# Filter & Resample: We filter the data and group it by Month.
# Plot: Use a line chart to show which sectors are seeing an increase in reposting (indicating growing structural difficulty).

# %%
# 1. Identify the Top 5 Hardest-to-Fill Sectors
top_5_sectors = (
    df_clean.groupby('primary_category')['hiring_difficulty_score']
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

# 2. Filter the dataframe for only these sectors
df_top_5 = df_clean[df_clean['primary_category'].isin(top_5_sectors)].copy()


# 3. Group by Sector AND Month, then sum the reposts
# 'MS' stands for Month Start
trends = (
    df_top_5.groupby(['primary_category', pd.Grouper(key='metadata_originalPostingDate', freq='MS')])
    ['metadata_repostCount']
    .sum()
    .reset_index()
)



# %%
# 4. Visualise with a Line Plot
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=trends, 
    x='metadata_originalPostingDate', 
    y='metadata_repostCount', 
    hue='primary_category', 
    marker='o',
    palette='tab10'
)

for x, y in zip(trends['metadata_originalPostingDate'], trends['metadata_repostCount']):
    plt.text(x, y, y, ha='center', va='bottom')


# Formatting for a professional report
plt.title('Monthly Repost Trends: Top 5 Hardest-to-Fill Sectors', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Number of Reposts', fontsize=12)
plt.legend(title='Sectors', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


