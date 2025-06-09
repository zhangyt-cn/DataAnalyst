import configparser
import json
import re
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pandas import DataFrame
from plotly.graph_objs._figure import Figure
import warnings
warnings.filterwarnings('ignore')

from prompts import data_visualize_prompt, inference_prompt
from llm import OpenAILLM

# @dataclass
# class AnalysisRequest:
#     """Structure for user analysis requirements"""
#     analysis_type: str  # 'descriptive', 'correlation', 'clustering', 'trend', 'custom'
#     target_columns: List[str] = None
#     group_by: str = None
#     filters: Dict[str, Any] = None
#     chart_type: str = 'auto'  # 'bar', 'line', 'scatter', 'heatmap', 'histogram', 'auto'
#     custom_query: str = None

class DataAnalysisAgent:
    """
    Comprehensive Data Analysis Agent that processes CSV data,
    performs analysis, generates visualizations, and provides insights.
    """
    
    def __init__(self, llm_config):
        self.data = None
        self.analysis_results = ""
        self.visualizations = []
        self.llm = OpenAILLM(llm_config)
        self.support_graphs = ["line", "pie", "bar", "scatter"]

       
    def load_data(self, file_path: str) -> Union[DataFrame, None]:
        """Load data from file path"""
        try:
            if file_path.endswith(".csv"):
                self.data = pd.read_csv(file_path)
                self._clean_data()
                print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
                return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    
    def _clean_data(self):
        """Basic data cleaning operations"""
        if self.data is not None:
            # Handle missing values
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
            
            # Handle categorical missing values
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                self.data[col] = self.data[col].fillna(self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'Unknown')
    

    def analyze_data(self, query: str) -> Dict[str, Any]:
        """Make analysis based on user requirements using a LLM"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        data_records = self.data.to_dict(orient="records")
        try:
            pre_prompt = inference_prompt.format(query=query, data=data_records)
            messages = [
                {
                    "role": "system",
                    "content": pre_prompt
                }
            ]
            rsp = self.llm.generate_response(messages)
            self.analysis_results = rsp
            return rsp
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    
    def parse_response(self, response):
        pattern = re.compile(r'```json(?:[^\n]*\n)?(.*?)```', re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) == 0:
            pattern = re.compile(r'```(?:[^\n]*\n)?(.*?)```', re.DOTALL)
            matches = pattern.findall(response)
        match = matches[-1].strip()
        return match


    def generate_visualization(self, query: str) -> Union[Figure, None]: 
        """Generate visualizations based on analysis and requirements"""
        if self.data is None:
            return []
        
        # visualizations = []
        max_retries = 5
        while True:
            try:
                pre_prompt = data_visualize_prompt.format(data=self.data.dtypes, query=query)
                messages = [
                    {"role": "system", "content": pre_prompt}
                ]
                rsp = self.llm.generate_response(messages)
                rsp = rsp.replace("None", "null")
                visualizing_standards = json.loads(self.parse_response(rsp))
                if not any(t in visualizing_standards["graph_type"] for t in self.support_graphs):
                    raise ValueError(f"{visualizing_standards['graph_type']} is not supported")
               
                viz = self._create_specific_visualization(visualizing_standards)
                if viz:
                    # visualizations.append(viz)
                    return viz
                # self.visualizations = visualizations
                # return visualizations
                
            except Exception as e:
                print(f"Visualization error: {str(e)}, retrying...")
                max_retries -= 1
                if max_retries < 0:
                    return None
              

    def _create_specific_visualization(self, visual_info: dict) -> Union[Figure, None]:
        """Create specific visualization based on request"""
        
        chart_type = visual_info["graph_type"]
        split_name = visual_info['split_name']
        self.data = self.data.sort_values(by=visual_info['x_axis_name'])
       
        if 'scatter' in chart_type:
            if split_name:
                fig = px.scatter(self.data, 
                                x=visual_info["x_axis_name"], 
                                y=visual_info["y_axis_name"],
                                title=visual_info["graph_title"],
                                color=visual_info["split_name"])
                
            else:
                fig = px.scatter(self.data, 
                            x=visual_info["x_axis_name"], 
                            y=visual_info["y_axis_name"],
                            title=visual_info["graph_title"],
                            color_discrete_sequence=["lightblue"])
            return fig
            
        elif 'line' in chart_type:
            if split_name:
                fig = px.line(self.data, 
                            x=visual_info["x_axis_name"], 
                            y=visual_info["y_axis_name"],
                            color=visual_info["split_name"],
                            title=visual_info["graph_title"],
                            markers=True)
            else:
                fig = px.line(self.data, 
                        x=visual_info["x_axis_name"], 
                        y=visual_info["y_axis_name"],
                        color_discrete_sequence=["lightblue"],
                        title=visual_info["graph_title"],
                        markers=True)
            # return fig.to_html(include_plotlyjs='cdn')
            return fig
            
        elif 'bar' in chart_type:
            if split_name:
                fig = px.bar(self.data, x=visual_info['x_axis_name'], y=visual_info['y_axis_name'],
                        color=visual_info['split_name'],
                        title=visual_info['graph_title'],
                        height=400)
            else:
                fig = px.bar(self.data, x=visual_info['x_axis_name'], y=visual_info['y_axis_name'],
                        color_discrete_sequence=["lightblue"],
                        title=visual_info['graph_title'],
                        height=400)
            return fig
        
        elif 'pie' in chart_type:
            fig = px.pie(self.data, values=visual_info["x_axis_name"], names=visual_info["y_axis_name"],
                            title=visual_info["graph_title"])
            return fig
        # TODO: add more graphs
            
        else:
            print(f"{chart_type} graph not supported yet!")
            return None
    

    def save_visualizations(self, output_dir: str = "./outputs/") -> List[str]:
        """Save all generated visualizations to files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for i, viz_html in enumerate(self.visualizations):
            filename = f"visualization_{i+1}.html"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(viz_html)
            
            saved_files.append(filepath)
        
        return saved_files
    

# Example usage and demo
def demo_analysis_agent():
    """Demonstration of the Data Analysis Agent"""
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'sales': [10, 15, 12, 20, 18, 19],
        'marketing_spend': [1000, 900, 960, 800, 1200, 1100],
        'temperature': [24, 30, 35, 28, 26, 32],
        'region': ["north", "south", "west", "north", "east", "south"],
        'product_category': ["A", "B", "C", "B", "D", "A"],
        "time": [4, 5, 7, 1, 3, 9]
    })
    
    # Add some correlation
    # sample_data['sales'] += sample_data['marketing_spend'] * 0.5 + np.random.normal(0, 50, 100)
    
    config = configparser.ConfigParser()
    config.read('config.cfg')
    llm_config = dict(config["llm"])
    agent = DataAnalysisAgent(llm_config)
    
    # Load data
    agent.load_data_from_dataframe(sample_data)
    
    # Generate visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    # viz_request = AnalysisRequest(analysis_type='auto')
    user_query = "I want to know how's the sales in different regions"
    visualization = agent.generate_visualization(user_query)
    print(f"Successfully generated visualization!")
    
    # Get data summary
    summary = agent.analyze_data(user_query)
    print("=== DATA SUMMARY ===")
    print(summary)

    # Save results
    # saved_files = agent.save_visualizations()
    # print(f"\n=== RESULTS SAVED ===")
    # print(f"Visualizations saved: {saved_files}")
    
    return agent

if __name__ == "__main__":
    # Run demonstration
    demo_agent = demo_analysis_agent()
    
    print("\n=== DATA ANALYSIS AGENT READY ===")
    print("The agent is ready to analyze your CSV data!")
 