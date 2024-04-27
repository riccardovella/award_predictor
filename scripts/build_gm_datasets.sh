# Builds the five graph measures (embedding) datasets from the small graphs dataset
 
python src/graph_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_0.parquet ~/award_predictor/dataset/graph_measures_d2_0
python src/graph_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_1.parquet ~/award_predictor/dataset/graph_measures_d2_1
python src/graph_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_2.parquet ~/award_predictor/dataset/graph_measures_d2_2
python src/graph_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_3.parquet ~/award_predictor/dataset/graph_measures_d2_3
python src/graph_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_4.parquet ~/award_predictor/dataset/graph_measures_d2_4