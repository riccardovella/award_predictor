# Builds the five text features (embedding) datasets from the small graphs dataset

python src/text_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_0.parquet ~/award_predictor/dataset/text_features_0
python src/text_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_1.parquet ~/award_predictor/dataset/text_features_1
python src/text_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_2.parquet ~/award_predictor/dataset/text_features_2
python src/text_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_3.parquet ~/award_predictor/dataset/text_features_3
python src/text_embedding/main.py ~/award_predictor/dataset/small_graphs_dataset_d2_4.parquet ~/award_predictor/dataset/text_features_4