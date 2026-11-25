# UK Data Jobs Analysis with Python

## Project Overview

This project explores the **United Kingdom (UK) data job market** using Python and job-posting data. It investigates which skills employers request across selected data roles, how demand for Data Analyst skills changes month by month, how salaries vary by role and skill, and which Data Analyst skills combine relatively high demand with higher median salaries.

The project follows a question-led approach. Each research question is supported by the analysis method, selected Python code, the relevant visualization(s), and findings drawn from the results.

## Research Questions

1. What are the most demanded skills for the three most popular data roles in the UK?
2. How are in-demand skills trending for Data Analysts in the UK?
3. How well do data jobs and skills pay in the UK?
4. What are the most optimal skills for Data Analysts in the UK when considering both demand and salary?

## Tools and Technologies

- **Python** — programming language used for the analysis
- **Pandas** — data cleaning, filtering, grouping, merging, reshaping and aggregation
- **NumPy** — numerical operations
- **Matplotlib** — chart creation and formatting
- **Seaborn** — statistical visualizations
- **ast** — parsing list- and dictionary-like values stored as strings
- **Hugging Face Datasets** — loading the job-postings dataset
- **Jupyter Notebook and VS Code** — analysis environment

---

## Data Loading and Preparation

The dataset was loaded from the `lukebarousse/data_jobs` collection on Hugging Face. The posting-date column was converted to a datetime format, and the `job_skills` field was parsed into Python lists so that individual skills could be analysed.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from datasets import load_dataset

loaded_dataset = load_dataset("lukebarousse/data_jobs")
df = loaded_dataset["train"].to_pandas()

df["job_posted_date"] = pd.to_datetime(df["job_posted_date"])
df["job_skills"] = df["job_skills"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else x
)
```

For the UK analysis, records were filtered using the `job_country` column:

```python
df_UK = df[df["job_country"] == "United Kingdom"].copy()
```

For salary analysis, postings without an annual salary value were excluded. The dataset's annual salary field was converted from USD to GBP using the fixed multiplier used in the notebooks:

```python
df_UK = df_UK.dropna(subset=["salary_year_avg"]).copy()
df_UK["salary_year_avg_gbp"] = df_UK["salary_year_avg"] * 0.7454
```

**Currency note:** `0.7454` is a fixed conversion multiplier, not a live or date-specific exchange rate. GBP figures in this project are therefore estimates converted from the dataset's USD salary values.

---

## 1. What Are the Most Demanded Skills for the Three Most Popular Data Roles in the UK?

### Method

The analysis focused on the three roles selected in the notebook: **Data Scientist, Data Engineer and Data Analyst**. The UK postings were filtered, the skill lists were expanded so that each skill occupied its own row, and skill mentions were counted by job title. The counts were then compared with the total number of postings for each role to calculate the percentage of postings requesting each skill.

```python
df_UK = df[df["job_country"] == "United Kingdom"].copy()
df_skills = df_UK.explode("job_skills")

df_skills_count = (
    df_skills.groupby(["job_title_short", "job_skills"])
    .size()
    .reset_index(name="skills_count")
)
```

The percentage calculation used the skill count divided by the number of postings for the corresponding job title:

```python
df_skills_perc["skills_perc"] = (
    df_skills_perc["skills_count"] / df_skills_perc["job_total"]
) * 100
```

### Visualization

![Most requested skills across Data Scientist, Data Engineer and Data Analyst postings in the UK](3.%20Projects/images/skills-count.png)

### Key Insights

- **Data Scientists:** **Python** was the leading skill, appearing in **69%** of postings in the chart. **SQL** followed at **44%**, then **R** at **29%**.
- **Data Engineers:** **SQL** led at **60%**, closely followed by **Python** at **55%**. **Azure** appeared in **41%** of postings, while **AWS** accounted for **33%**.
- **Data Analysts:** **SQL (43%)** and **Excel (41%)** were the two most requested skills. **Power BI** followed at **27%**, then **Python (20%)** and **Tableau (16%)**.
- **SQL** featured among the leading skills for all three roles, while the other leading tools varied by role. The chart therefore shows both shared foundations and role-specific differences in the UK postings analysed.

---

## 2. How Are In-Demand Skills Trending for Data Analysts in the UK?

### Method

This analysis filtered the dataset to UK Data Analyst postings, extracted the month from each posting date, and expanded the skill lists. A pivot table was used to count monthly skill mentions. The monthly counts were divided by the total number of Data Analyst postings in each month to calculate the share of postings requesting each skill.

```python
df_DA_UK = df[
    (df["job_title"] == "Data Analyst") &
    (df["job_country"] == "United Kingdom")
].copy()

df_DA_UK["job_posted_month_no"] = (
    df_DA_UK["job_posted_date"].dt.month
)

df_DA_UK_exploded = df_DA_UK.explode("job_skills")
```

The monthly skill counts were converted into percentages of that month's Data Analyst postings:

```python
df_DA_UK_Percent = (
    df_DA_UK_pivot.div(DA_Total, axis=0) * 100
)
```

### Visualization

![Monthly trend of the five leading Data Analyst skills in the UK](3.%20Projects/images/skills-trend.png)


### Key Insights

- **SQL** had the highest share throughout the chart, moving from approximately **44% in December** to a high of about **57% in September**.
- **Excel** remained the second most prominent skill in most months, ranging approximately from **37% to 46%**.
- **Power BI** varied more noticeably, reaching approximately **38% in July** and falling to around **25% in December**.
- **Python** and **Tableau** generally had lower shares than SQL and Excel. Python reached approximately **30% in October**, while Tableau was around **24% in October**.
- These percentages represent the share of monthly UK Data Analyst postings that mention each skill. They do not represent the total number of vacancies.

**Date-scope note:** The chart title includes “2023”, but the notebook code shown for this chart extracts the month without applying an explicit year filter. The chart is therefore presented as the monthly pattern produced by the notebook; the year label should be verified against the dataset's date coverage before making a year-specific claim.

---

## 3. How Well Do Data Jobs and Skills Pay in the UK?

This question was explored in two parts: salary distributions across six data job titles, and median salaries associated with skills in UK Data Analyst postings.

### 3.1 Salary Distribution Across Data Roles

#### Method

The six most frequently occurring job titles in the UK subset were selected. Their annual salary distributions were compared using a box plot. Only postings with a recorded annual salary were included, and salary values were converted to GBP using the fixed multiplier described above.

```python
job_titles = (
    df_UK["job_title_short"].value_counts().head(6).index.tolist()
)

df_UK_top6 = df_UK[
    df_UK["job_title_short"].isin(job_titles)
]

job_order = (
    df_UK_top6.groupby("job_title_short")["salary_year_avg_gbp"]
    .median()
    .sort_values(ascending=False)
    .index
)
```

#### Visualization

![Annual salary distributions for six data roles in the UK](3.%20Projects/images/salary-distribution.png)

#### Key Insights

- **Senior Data Engineer** has a median close to **£110K** in the plotted comparison, placing its median above the other five roles. Its box is very narrow, while several individual postings appear as lower and higher outliers.
- **Senior Data Scientist** has a median of approximately **£68K**, with the middle 50% of salaries extending from roughly **£67K to £118K**.
- **Data Engineer** and **Data Scientist** have medians in the region of **£80K–£83K**. Their boxes show a wider spread than the Senior Data Analyst box.
- **Senior Data Analyst** has a median around **£82K**, although its box is extremely narrow in this chart and several observations appear as outliers.
- **Data Analyst** has a median around **£65K**, with the middle half of the plotted salaries extending roughly from **£43K to £78K**.
- These are approximate readings from the displayed box plot. The box plot communicates the median, spread and outliers; it does not show the number of postings behind each role.

### 3.2 Highest-Paid and Most In-Demand Skills for Data Analysts

#### Method

The analysis filtered for UK Data Analyst postings with annual salary information, converted salary values to GBP, and expanded the skill lists. Median salary was calculated for each skill. Two rankings were then created: the ten skills with the highest median salaries and the ten most frequently occurring skills.

```python
df_DA_UK = df[
    (df["job_title_short"] == "Data Analyst") &
    (df["job_country"] == "United Kingdom")
].copy()

df_DA_UK["salary_year_avg_gbp"] = (
    df_DA_UK["salary_year_avg"] * 0.7454
)

df_DA_UK = df_DA_UK.dropna(
    subset=["salary_year_avg_gbp"]
)

df_DA_UK = df_DA_UK.explode("job_skills")
```

The median salary and number of observations were calculated by skill:

```python
df_DA_top_pay = (
    df_DA_UK.groupby("job_skills")["salary_year_avg_gbp"]
    .agg(["count", "median"])
    .sort_values(by="median", ascending=False)
    .head(10)
)
```

#### Visualization

![Highest-paid and most in-demand skills for Data Analysts in the UK](3.%20Projects/images/salary-skills.png)

#### Key Insights

- In the **highest-paid skills** panel, **C++**, **NumPy**, **TensorFlow**, **PyTorch** and **Pandas** appear at approximately **£132K** median annual salary. **Aurora, Kafka, AWS, MySQL and PostgreSQL** follow at approximately **£123K**.
- In the **most in-demand skills** panel, **Tableau** has a median close to **£75K**, followed by **SQL** at approximately **£73K** and **Looker** at approximately **£71K**.
- **Power BI** and **Python** are both around **£66K**, while **SAS** is close to **£60K** in the most in-demand group.
- The comparison shows that the skills with the highest median salaries are not necessarily the same skills that appear most frequently in postings.
- The plotted salary figures are medians among postings with salary information that mention each skill. They should not be interpreted as guaranteed salaries for someone who learns that skill.

*The salary amounts in this subsection are approximate readings from the chart. The exact values are not printed in the saved notebook output available for this project.*

---

## 4. What Are the Most Optimal Skills for Data Analysts in the UK?

### Method

The final analysis brought together two measures for each skill: the percentage of salary-reported UK Data Analyst postings requesting it and its median annual salary in GBP. The analysis retained skills with more than five mentions. The first scatter plot compares demand and salary using one colour. The second adds technology categories to provide another view of the same skills.

```python
df_DA_Skills = (
    df_DA_UK_exploded.groupby("job_skills")
    .agg(
        skills_count=("job_skills", "size"),
        median_salary_gbp=("salary_year_avg_gbp", "median"),
        median_salary=("salary_year_avg", "median")
    )
    .sort_values(by="skills_count", ascending=False)
)

DA_jobs_count = len(df_DA_UK)

df_DA_Skills["skills_percent"] = (
    df_DA_Skills["skills_count"] / DA_jobs_count * 100
)

df_DA_Skills_high_demand = df_DA_Skills[
    df_DA_Skills["skills_count"] > 5
]
```

### 4.1 Overall Demand vs. Median Salary

![Overall demand and median salary for Data Analyst skills in the UK](3.%20Projects/images/optimal-skills-overall.png)

### 4.2 Demand vs. Median Salary by Technology Category

![Demand and median salary for Data Analyst skills in the UK, grouped by technology category](3.%20Projects/images/optimal-skills.png)

### Key Results

The table below uses the values saved in the notebook's `df_DA_Skills_high_demand` output.

| Skill | Postings mentioning skill | Share of salary-reported Data Analyst postings | Median annual salary (GBP) |
|---|---:|---:|---:|
| **SQL** | 33 | **45.8%** | **£73,421.90** |
| **Excel** | 29 | **40.3%** | **£56,314.97** |
| **Python** | 25 | **34.7%** | **£66,415.14** |
| **Tableau** | 13 | **18.1%** | **£74,912.70** |
| **R** | 9 | **12.5%** | **£57,408.84** |
| **Go** | 9 | **12.5%** | **£52,923.40** |
| **SAS** | 8 | **11.1%** | **£59,637.22** |
| **Power BI** | 7 | **9.7%** | **£66,415.14** |
| **Looker** | 6 | **8.3%** | **£71,567.72** |
| **Outlook** | 6 | **8.3%** | **£40,145.01** |
| **Azure** | 6 | **8.3%** | **£68,912.23** |

### Key Insights

- **SQL** combines the highest demand in this group, appearing in **33 postings (45.8%)**, with a median annual salary of approximately **£73.4K**.
- **Excel** is also highly requested, appearing in **29 postings (40.3%)**, but its median salary of approximately **£56.3K** is lower than SQL, Python and Tableau.
- **Python** appears in **25 postings (34.7%)** and has a median salary of approximately **£66.4K**, making it another strong combination of demand and salary in this subset.
- **Tableau** has the highest median salary in the table at approximately **£74.9K**, despite appearing in **18.1%** of the salary-reported postings.
- **Power BI** appears in **9.7%** of postings and has a median salary of approximately **£66.4K**. Its demand share is lower than SQL, Excel and Python in this analysis.
- **Azure** appears in **8.3%** of postings and has a median salary of approximately **£68.9K**.
- The technology-coloured scatter plot groups the skills into categories such as **programming**, **analyst tools** and **cloud**. This adds context to the demand-and-salary comparison rather than replacing the overall scatter plot.

**How to interpret “optimal”:** In this project, “optimal” is a comparative label for skills that combine demand and median salary within this dataset subset. It is not a universal ranking or a guarantee that learning a particular skill will produce a particular salary.

---

## Overall Findings: What the UK Data Job Postings Show

Across the four research questions, several patterns stand out:

- **SQL is a recurring foundation** across the three data roles and is the most requested skill in the Data Analyst optimal-skills subset.
- **Excel remains highly relevant** to Data Analyst postings, appearing in **40.3%** of the salary-reported subset used in the final analysis.
- **Python is important across roles**: it leads the Data Scientist chart at **69%**, appears in **55%** of Data Engineer postings, and appears in **34.7%** of the salary-reported Data Analyst subset.
- **Power BI and Tableau** are both present in the Data Analyst results, but their demand and median salary figures differ.
- **Demand and salary do not always move together.** SQL and Excel have the highest demand shares in the final Data Analyst comparison, while Tableau has the highest median salary among the listed skills.
- Salary distributions differ across job titles, and individual postings can sit well above or below a role's median.

## Challenges and Learning

During this project, I applied and practised:

- Loading a real-world dataset and preparing it for analysis.
- Converting date columns and parsing list-like values stored as strings.
- Filtering records by country and job title.
- Exploding list columns to analyse individual skills.
- Grouping, aggregating, merging and pivoting DataFrames.
- Calculating percentages and median salaries.
- Creating and formatting bar charts, line charts, box plots and scatter plots with Matplotlib and Seaborn.
- Combining multiple measures to compare skill demand and salary.
- Interpreting results and communicating findings through visualizations.


## Conclusion

This project used Python to investigate four practical questions about the **UK data job market**: which skills are requested across data roles, how Data Analyst skill demand changes by month, how salaries vary across roles and skills, and which skills combine demand with median salary.

The results highlight **SQL, Excel and Python** as frequently requested Data Analyst skills in the analysed postings. They also show why demand and salary should be considered together: **Tableau**, for example, has a higher median salary than SQL in the final skill comparison, while SQL appears in a much larger share of postings.

The project strengthened my ability to move through a data-analysis workflow: preparing data, asking focused questions, analysing patterns, creating visualizations and communicating evidence-based findings.

## Dataset

- **Dataset:** [Luke Barousse — Data Jobs](https://huggingface.co/datasets/lukebarousse/data_jobs)
- **Geographic focus:** United Kingdom
- **Salary presentation:** GBP estimates converted from the dataset's USD annual salary field using a fixed multiplier of `0.7454`
