python train.py --depth 18 --width 1 --data ./ILSVRC2012/ --nthread 16 --epochs 100 \
                --train_batch 256 --test_batch 256 --lr 0.1 --schedule [30,60,90] --lr_decay_ratio 0.1 \
                --momentum 0.9 --weight_decay 1e-4 --gpu_id 0,1
