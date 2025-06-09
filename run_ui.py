import configparser
import gradio as gr
import pandas as pd
from data_analysis_agent import DataAnalysisAgent


config = configparser.ConfigParser()
config.read('config.cfg')
llm_config = config["llm"]
analyzer = DataAnalysisAgent(llm_config)

def clear_all():
    return None, "", ""

# Create the Gradio interface
with gr.Blocks(title="Data Analysis Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Data Analysis Chatbot")
    gr.Markdown("Upload a file and ask any questions about your data!")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            gr.Markdown("## 📁 Upload Your Data")
            file = gr.File(
                label="Upload File",
                file_types=[".csv"],
                type="filepath"
            )
            
            # Data info display
            gr.Markdown("## 📋 Dataset Information")

            data_content = gr.Dataframe(
                label="Dataset Overview",
                wrap=True,  # 自动换行
                row_count=(5, "fixed"),
                col_count=5,  
                max_height=200,
                # overflow_row_behaviour="paginate",  # 分页显示
                interactive=False,
                
            )
            # data_info = gr.Textbox(label="File Name")
            gr.Markdown("## 💬 Ask Questions About Your Data")
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g., 'Show me 5 insights about the data'",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Ask Question", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
    
                
        with gr.Column(scale=1):
            # Chat interface
            gr.Markdown("## 🤓 Visualizing & Analyzing")
            # chatbot = gr.Chatbot(
            #     label="Data Analysis Chat",
            #     height=400,
            #     show_copy_button=True
            # )
            graph = gr.Plot()
            
            analysis_report = gr.Markdown(
                label="Analysis Results",
                show_label=True,
                show_copy_button=True
            )

          
           
    # # Example questions
    # gr.Markdown("## 💡 Example Questions")
    # gr.Markdown("""
    # - What's the shape of the data?
    # - Show me the first 10 rows
    # - Describe the data
    # - What columns are available?
    # - Are there any missing values?
    # - Show correlation between columns
    # - What are the unique values in [column_name]?
    # - Show value counts for [column_name]
    # - What's the mean of numeric columns?
    # """)
    
    # Event handlers
    file.change(
        fn=analyzer.load_data,
        inputs=[file],
        outputs=[data_content]
    )
    
    submit_btn.click(
        fn=analyzer.generate_visualization,
        inputs=[question],
        outputs=[graph]
    ).then(
        fn=analyzer.analyze_data,
        inputs=[question],
        outputs=[analysis_report]
    )


    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[graph, question, analysis_report]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  
        server_name="0.0.0.0",  
        server_port=7860  
    )