# Trains all models from all available datasets

python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_0.npz -hd 9 -l 0.0025 -e 2500 -s 0 -n gm00
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_1.npz -hd 9 -l 0.0025 -e 2500 -s 0 -n gm01
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_2.npz -hd 9 -l 0.0025 -e 2500 -s 0 -n gm02
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_3.npz -hd 9 -l 0.0025 -e 2500 -s 0 -n gm03
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_4.npz -hd 9 -l 0.0025 -e 2500 -s 0 -n gm04
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_0.npz -hd 9 -l 0.0025 -e 2500 -s 1 -n gm10
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_1.npz -hd 9 -l 0.0025 -e 2500 -s 1 -n gm11
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_2.npz -hd 9 -l 0.0025 -e 2500 -s 1 -n gm12
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_3.npz -hd 9 -l 0.0025 -e 2500 -s 1 -n gm13
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_4.npz -hd 9 -l 0.0025 -e 2500 -s 1 -n gm14

python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_0.npz -hd 128 -l 0.001 -e 1000 -s 0 -n tx00
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_1.npz -hd 128 -l 0.001 -e 1000 -s 0 -n tx01
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_2.npz -hd 128 -l 0.001 -e 1000 -s 0 -n tx02
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_3.npz -hd 128 -l 0.001 -e 1000 -s 0 -n tx03
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_4.npz -hd 128 -l 0.001 -e 1000 -s 0 -n tx04
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_0.npz -hd 128 -l 0.001 -e 1000 -s 1 -n tx10
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_1.npz -hd 128 -l 0.001 -e 1000 -s 1 -n tx11
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_2.npz -hd 128 -l 0.001 -e 1000 -s 1 -n tx12
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_3.npz -hd 128 -l 0.001 -e 1000 -s 1 -n tx13
python src/award_predictor/main.py ~/award_predictor/out --tx_data_path ~/award_predictor/dataset/text_features_4.npz -hd 128 -l 0.001 -e 1000 -s 1 -n tx14

python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_0.npz --tx_data_path ~/award_predictor/dataset/text_features_0.npz -hd 1 -l 0.05 -e 5 -s 0 -n mixed00 -m --gm_model_path ~/award_predictor/out/gm00/best_loss.pt --tx_model_path ~/award_predictor/out/tx00/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_1.npz --tx_data_path ~/award_predictor/dataset/text_features_1.npz -hd 1 -l 0.05 -e 5 -s 0 -n mixed01 -m --gm_model_path ~/award_predictor/out/gm01/best_loss.pt --tx_model_path ~/award_predictor/out/tx01/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_2.npz --tx_data_path ~/award_predictor/dataset/text_features_2.npz -hd 1 -l 0.05 -e 5 -s 0 -n mixed02 -m --gm_model_path ~/award_predictor/out/gm02/best_loss.pt --tx_model_path ~/award_predictor/out/tx02/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_3.npz --tx_data_path ~/award_predictor/dataset/text_features_3.npz -hd 1 -l 0.05 -e 5 -s 0 -n mixed03 -m --gm_model_path ~/award_predictor/out/gm03/best_loss.pt --tx_model_path ~/award_predictor/out/tx03/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_4.npz --tx_data_path ~/award_predictor/dataset/text_features_4.npz -hd 1 -l 0.05 -e 5 -s 0 -n mixed04 -m --gm_model_path ~/award_predictor/out/gm04/best_loss.pt --tx_model_path ~/award_predictor/out/tx04/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_0.npz --tx_data_path ~/award_predictor/dataset/text_features_0.npz -hd 1 -l 0.05 -e 5 -s 1 -n mixed10 -m --gm_model_path ~/award_predictor/out/gm10/best_loss.pt --tx_model_path ~/award_predictor/out/tx10/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_1.npz --tx_data_path ~/award_predictor/dataset/text_features_1.npz -hd 1 -l 0.05 -e 5 -s 1 -n mixed11 -m --gm_model_path ~/award_predictor/out/gm11/best_loss.pt --tx_model_path ~/award_predictor/out/tx11/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_2.npz --tx_data_path ~/award_predictor/dataset/text_features_2.npz -hd 1 -l 0.05 -e 5 -s 1 -n mixed12 -m --gm_model_path ~/award_predictor/out/gm12/best_loss.pt --tx_model_path ~/award_predictor/out/tx12/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_3.npz --tx_data_path ~/award_predictor/dataset/text_features_3.npz -hd 1 -l 0.05 -e 5 -s 1 -n mixed13 -m --gm_model_path ~/award_predictor/out/gm13/best_loss.pt --tx_model_path ~/award_predictor/out/tx13/best_f1.pt
python src/award_predictor/main.py ~/award_predictor/out --gm_data_path ~/award_predictor/dataset/graph_measures_d2_4.npz --tx_data_path ~/award_predictor/dataset/text_features_4.npz -hd 1 -l 0.05 -e 5 -s 1 -n mixed14 -m --gm_model_path ~/award_predictor/out/gm14/best_loss.pt --tx_model_path ~/award_predictor/out/tx14/best_f1.pt