import configparser
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from prompts import synthesize_query_prompt, synthesize_response_prompt, ensure_query_solvable_prompt
from llm import OpenAILLM

class DataSynthesizer:
    """
    A comprehensive data synthesis system that analyzes datasets and generates
    insights based on user queries.
    """
    
    def __init__(self, llm_config: dict):

        self.domains = ["democratics", "finacial", "health", "marketing", "operation"] # 
        self.llm = OpenAILLM(llm_config)
        self.sentence_model = None
    

    def clustering_based_dedup(self, texts: List[str], eps: float = 0.1, 
                              min_samples: int = 1) -> List[str]:
        """Remove duplicates using clustering"""
        if not texts:
            return None
        
        try:
            if self.sentence_model is None:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self.sentence_model.encode(texts)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Keep one representative from each cluster!!
            unique_indices = []
            seen_clusters = set()
            
            for i, label in enumerate(cluster_labels):
                if label not in seen_clusters:
                    unique_indices.append(i)
                    seen_clusters.add(label)
            
            dedup_data = [texts[i] for i in unique_indices]
            return dedup_data 
            
        except Exception as e:
            print(f"Clustering-based deduplication failed: {e}")
            return None
        
    def filter_solvable_questions(self, queries: List[str], domain: str) -> List[str]:
        if not queries:
            return None
        
        data_df = pd.read_csv(f"{domain}.csv")
        data = data_df.to_dict(orient="records")
        solvable_queries = []
        try:
            for q in queries:
                pre_prompt = ensure_query_solvable_prompt.format(query=q, data=data)
                messages = [
                    {
                        "role": "system", "content": pre_prompt
                    }
                ]
                rsp = self.llm.generate_response(messages)
                if "yes" in rsp:
                    solvable_queries.append(q)
        except Exception as e:
            print(f"encounter error when filter solvable questions: {e}")
            return None
        
        return solvable_queries
        
    def generate_questions(self, num: int) -> pd.DataFrame:
        """Generate realistic multi-domain data samples and questions, 
           and preprocess queries to ensure data quality.
        """
        np.random.seed(42)
        
        # synthesize base data across multiple domains
        demo_data = {
            'id': range(1, num + 1),
            'age': np.random.normal(35, 12, num).astype(int),
            'gender': np.random.choice(['M', 'F', 'Other'], num, p=[0.48, 0.48, 0.04]),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num),
            'time': np.random.normal(2010, 2020, num).astype(int)
        }

        fin_data = {
            'revenue': np.random.lognormal(10, 1, num).astype(int),
            'profit_margin': np.random.normal(0.15, 0.08, num),
            'investment': np.random.exponential(50000, num).astype(int),
        }

        health_data = {
            'age': np.random.normal(35, 12, num).astype(int),
            'health_score': np.random.beta(2, 1, num) * 100,
            'treatment_cost': np.random.gamma(2, 1000, num).astype(int),
            'recovery_days': np.random.poisson(7, num),
        }

        mkt_data = {
            'conversion_rate': np.random.beta(2, 8, num),
            'ad_spend': np.random.exponential(5000, num).astype(int),
            'engagement_score': np.random.gamma(3, 20, num),
        }

        op_data = {
            'efficiency_score': np.random.beta(3, 2, num) * 100,
            'downtime_hours': np.random.exponential(2, num),
            'quality_rating': np.random.choice([1, 2, 3, 4, 5], num, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        }
        
        # Add some correlations to make data more realistic
        for i in range(num):
            # Age influences health and treatment cost
            if health_data['age'][i] > 50:
                health_data['health_score'][i] *= 0.8
                health_data['treatment_cost'][i] *= 1.3
            
            # Revenue influences investment capacity
            if fin_data['revenue'][i] > np.median(fin_data['revenue']):
                fin_data['investment'][i] *= 1.5
                
            # High engagement should correlate with conversion
            if mkt_data['engagement_score'][i] > np.percentile(mkt_data['engagement_score'], 75):
                mkt_data['conversion_rate'][i] *= 1.4
        
        data_dict = {
            "democratics": pd.DataFrame(demo_data),
            "finacial": pd.DataFrame(fin_data),
            "health": pd.DataFrame(health_data),
            "marketing": pd.DataFrame(mkt_data),
            "operation": pd.DataFrame(op_data),
        }


        # synthesize questions & preprocess data
        for category, domain_df in data_dict.items():
            for _ in range(4):
                pre_prompt = synthesize_query_prompt.format(domain=category, data_schema=domain_df.dtypes)
                messages = [
                    {"role": "system", "content": pre_prompt}
                ]
                rsp = self.llm.generate_response(messages)
                queries = rsp.split("\n")

                # data deduplication (cluster based)
                self.clustering_based_dedup(queries)

                # unsolvable problem removal
                self.filter_solvable_questions(queries, category)

                with open(f"{category}.txt", "a") as f_q:
                    f_q.write("\n".join(queries) + "\n")
            
            domain_df.to_csv(f"{category}.csv", index=False, mode="w")
    
    def generate_response(self):
        """
        Synthesize answers based on previously generated data samples and queries.
        """
        for category in self.domains:
            cate_data= []
            df = pd.read_csv(f"{category}.csv")
            data_lst = df.to_dict(orient="records")

            with open(f"{category}.txt", "r") as f:
                queries = f.read()
                queries = queries.split("\n")
            
            for q in queries:
                pre_prompt = synthesize_response_prompt.format(domain=category, data=data_lst, query=q)
                messages = [
                    {"role": "system", "content": pre_prompt}
                ]
                analysis = self.llm.generate_response(messages)
                cate_data.append(
                    {
                        "category": category,
                        "query": q,
                        "response": analysis
                    }
                )
                
            with open("full_data.json", "a") as f:
                json.dump(cate_data, f, indent=4)
           
            print(f"{category} data answer generated!")

            
def main():

    config = configparser.ConfigParser()
    config.read('config.cfg')
    llm_config = dict(config["llm"])

    synthesizer = DataSynthesizer(llm_config)
    
    # Generate sample dataset
    print("Generating sample dataset...")
    print("Data sample and questions successfully generated!")
    synthesizer.generate_sample_dataset(50)
    print("Answers successfully generated!")
    synthesizer.generate_response()


if __name__ == "__main__":
    main()