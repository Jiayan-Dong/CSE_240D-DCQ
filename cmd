# Floating point
python3 dcq.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_float/"
# W32
python3 dcq.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_w32/" --compress=./quantization/svhn_dorefa_w32.yaml --resume=./logs/svhn_float/2023.12.16-125255/best.pth.tar
# W1
python3 dcq.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_w1/" --compress=./quantization/svhn_dorefa_w1.yaml --resume=./logs/svhn_w32/2023.12.16-125715/best.pth.tar

# W1_DCQ_Stage1
python3 svhn_partial.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_w1_dcq_s1/" --compress=./quantization/svhn_dorefa_w1.yaml --resume=./logs/svhn_w32/2023.12.16-125715/best.pth.tar --fpcompress=./quantization/svhn_dorefa_w32.yaml --fpresume=./logs/svhn_w32/2023.12.16-125715/best.pth.tar
# W1_DCQ_Stage2
python3 svhn_partial_2.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_w1_dcq_s2/" --compress=./quantization/svhn_dorefa_w1.yaml --resume=./logs/svhn_w1_dcq_s1/2023.12.16-130939/checkpoint.pth.tar --fpcompress=./quantization/svhn_dorefa_w32.yaml --fpresume=./logs/svhn_w32/2023.12.16-125715/best.pth.tar

# Fine Tun
python3 fine_tune.py -a svhn --lr 0.01 -p 50 -b 128 ./data/data.svhn/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/svhn_w1_dcq_fine_tune/" --compress=./quantization/svhn_dorefa_w1.yaml --resume=./logs/svhn_w1_dcq_s2/2023.12.16-152237/best.pth.tar --fpcompress=./quantization/svhn_dorefa_w32.yaml --fpresume=./logs/svhn_w32/2023.12.16-125715/best.pth.tar

# Floating point
python3 dcq.py -a lenet_mnist --lr 0.005 -p 50 -b 128 ./data/data.mnist/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/lenet_float/"

# W32
python3 dcq.py -a lenet_mnist --lr 0.01 -p 50 -b 128 ./data/data.mnist/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/lenet_w32/" --compress=./quantization/lenet_mnist_dorefa_w32.yaml --resume=./logs/lenet_float/best.pth.tar

# W1
python3 dcq.py -a lenet_mnist --lr 0.01 -p 50 -b 128 ./data/data.mnist/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/lenet_w1/" --compress=./quantization/lenet_mnist_dorefa_w1.yaml --resume=./logs/lenet_w32/best.pth.tar

# W1_DCQ_Stage1
python3 lenet_partial.py -a lenet_mnist --lr 0.01 -p 50 -b 128 ./data/data.mnist/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/lenet_w1_dcq_s1/" --compress=./quantization/lenet_mnist_dorefa_w1.yaml --resume=./logs/lenet_w32/best.pth.tar --fpcompress=./quantization/lenet_mnist_dorefa_w32.yaml --fpresume=./logs/lenet_w32/best.pth.tar

# W1_DCQ_Stage2
python3 lenet_partial_2.py -a lenet_mnist --lr 0.01 -p 50 -b 128 ./data/data.mnist/ -j 8 --epochs 20 --wd=0.0002 --vs=0 --gpus 0 --out-dir="logs/lenet_w1_dcq_s2/" --compress=./quantization/lenet_mnist_dorefa_w1.yaml --resume=./logs/lenet_w1_dcq_s1/checkpoint.pth.tar --fpcompress=./quantization/lenet_mnist_dorefa_w32.yaml --fpresume=./logs/lenet_w32/best.pth.tar

# Fine Tune