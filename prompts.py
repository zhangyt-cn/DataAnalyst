
data_visualize_prompt = """
Based on the user's requirements and the provided data, determine the most appropriate graph to present. Consider the data type, variables, and visualization goals (e.g., trends, comparisons, distributions). 
Currently, you can only decide from four types of graph, which are line graph, bar graph, scatter plot, and pie chart.

## user requirements
{query}

## data information
Here is partial data with column names and data types:
{data}

# response format
Your response should be formatted as a JSON object with the following keys:  
- graph_type: e.g., line chart, bar chart, pie chart, scatter plot
- x_axis_name: data column used as label for the horizontal axis (for pie chart, this means data values)
- y_axis_name: data column used as label for vertical axes (for pie chart, this means graph segment variable)
- split_name: data column used to split graph (e.g., multiple lines in line chart), otherwise set to None
- graph_title: descriptive title summarizing the visualization

Here is the final JSON format:
```json
{{
    "graph_type": <graph_type>,
    "x_axis_name": <x_axis_name>,
    "y_axis_name": <y_axis_name>,
    "split_name": <split_name>,
    "graph_title": <graph_title>
}}
```
"""


synthesize_query_prompt = """
You are an expert data analyst. Generate 50 diverse and insightful user questions for data analysis. Questions should focus on trends, patterns, anomalies, or business insights, NOT simple data retrieval. 

## Data Specification
Here is provided data in {domain}, data column names and types are as below:
{data_schema} 

## Constraints  
1. Prioritize time-based trends, comparisons, correlations, and predictive questions.  
2. Avoid yes/no questions.  
3. Phrase questions as a curious user/analyst would ask.  
4. Only contain questions in your response, NO other irrelevant content.

## Output Format  
<first_query>
<second_query>
"""

synthesize_response_prompt = """
You are a senior data analyst with expertise in pattern recognition, statistical analysis, and business intelligence. 
Perform an insightful analysis of the provided data to address the user's specific query. Your analysis should be high-level, consice, and readable. 

## User Query
{query}

## Dataset Context
The data is in {domain}, each record contains keys representing header and corresponding value. All records are shown as below:
{data}

## Response
You should ONLY give your final insightful analysis based on user's requirement and data content without your thought process.
Your response:
"""

inference_prompt = """
You are a senior data analyst with expertise in pattern recognition, statistical analysis, and business intelligence. 
Perform an insightful analysis of the provided data to address the user's specific query. Your analysis should be high-level, consice, and readable. 

## User Query
{query}

## Dataset Context
The original data is formatted as records, each record contains keys representing header and corresponding value. All records are shown as below:
{data}

## Response
You should ONLY give your final insightful analysis based on user's requirement and data content WITHOUT your thought process.
However, if you can not address user's concern due to data incompleteness or other reasons, you should honestly admit your limit and encourage another suitable question.
Your response:
"""

ensure_query_solvable_prompt = """
You are a senior analyst, excelled at data analyzing using your meticulous inspection and thoughtful reasoning.
Determine whether the provided question can be answered SOLELY using provided data sample.  
You should respond "yes" if The answer can be inferred unambiguously from the data, otherwise, "no" if the data is irrelevant, incomplete, or ambiguous for the question, without including extra explanation.

## Question
{query}

## Data Content
{data}

## Resonse Format
<yes_or_no> 
"""