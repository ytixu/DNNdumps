# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 50 -t 3 -l ../models/lstm_ae_losses.txt
# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 3 -l ../models/lstm_ae_losses.txt
# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 50 -t 5 -l ../models/lstm_ae_losses.txt
# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 5 -l ../models/lstm_ae_losses.txt
# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 50 -t 7 -l ../models/lstm_ae_losses.txt
# python LSTM_AE.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 7 -l ../models/lstm_ae_losses.txt

python OptionLSTM.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 5 -o 20 -l ../models/OptionLSTM_losses.txt
python OptionLSTM.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 5 -o 30 -l ../models/OptionLSTM_losses.txt
python OptionLSTM.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 5 -o 40 -l ../models/OptionLSTM_losses.txt
python OptionLSTM.py -id ../data/seq_human/ -od ../data/seq_robot/ -ld 100 -t 5 -o 50 -l ../models/OptionLSTM_losses.txt
