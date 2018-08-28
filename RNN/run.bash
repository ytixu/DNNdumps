#python HH_RNN-euler.py -t 25 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python HH_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 40 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;
# python H_RNN-euler.py -t 50 -ld 1024 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

# python H_RNN-euler.py -t 20 -ld 514 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64;

loss_func=( "mean_squared_error" "mean_absolute_error");
optimizers=( "optimizers.SGD(momentum=0.8,decay=0.001,nesterov=True)" "optimizers.SGD(momentum=0.8,decay=0.001,nesterov=False)" "optimizers.Adam(decay=0.0001)" "optimizers.Adam(lr=0.002)" "optimizers.Adam(lr=0.0005)" "optimizers.Nadam(schedule_decay=0.006)" "optimizers.Nadam(lr=0.0015)" );
for loss in "${loss_func[@]}"
do
	for opt in "${optimizers[@]}"
	do
		python H_RNN-euler.py -t 20 -ld 512 -id ../data/h3.6/full/train_euler/ -od ../data/h3.6/full/train_euler/ -bs 64 -lf $loss -opt $opt;
	done
done;
