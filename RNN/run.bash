#python HH_RNN-euler.py -t 25 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

# python H_RNN-euler.py -t 20 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

loss_func=( "mean_squared_error" "mean_absolute_error" "binary_crossentropy" "kullback_leibler_divergence" "cosine_proximity");
optimizers=( "optimizers.Nadam()" "optimizers.Nadam(lr=0.001)" "optimizers.Nadam(lr=0.003)" "optimizers.SGD()" "optimizers.Adagrad()" "optimizers.Adadelta()" "optimizers.Adam()" "optimizers.Adamax()" "optimizers.TFOptimizer()");
for loss in "${loss_func[@]}"
do
	for opt in "${optimizers[@]}"
	do
		python H_RNN-euler.py -t 20 -ld 512 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64 -lf $loss -opt $opt;
	done
done;
