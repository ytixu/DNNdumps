#python HH_RNN-euler.py -t 25 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

# python H_RNN-euler.py -t 20 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

loss_func=( "mean_squared_error" "mean_absolute_error" );
optimizers=( "optimizers.Adam()" "optimizers.Nadam(lr=0.001)" );
for loss in "${loss_func[@]}"
do
	for opt in "${optimizers[@]}"
	do
		python H_RNN-euler.py -t 30 -ld 512 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64 -lf $loss -opt $opt;
	done
done;
