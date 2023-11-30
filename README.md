# Chunks retrieval task

Hi there 👋,
Welcome on the chunk retrieval task readme file.
You will be able to explore the solution in different ways :
• EDA.ipynb file
• Python application
• Streamlit viewer


# EDA.ipynb

You will find here some data insights and the different attempted

# Python application

The classes where that will let you chose a strategy in order to complete the task.
Please refer to *config.yml* for configuration

## Installation

`pip install -r requirements.txt`

## Run

``python main.py --dialogues_path --summaries_path  --config_path``

if you to stick with default arguments, just place *dialogue.csv* and *reference.csv* in the **data/** folder

Please note that the scripts is saving intermediary files for speed execution purpose. If you decide to run it with another strategy **please make sure to  remove** *score_matrix.txt* file.

# Streamlit

View the results in a fancier way.
Place final under *result/final_df.csv*.

`streamlit run streamlit.py`