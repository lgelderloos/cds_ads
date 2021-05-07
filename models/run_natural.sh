for s in 123 234 345
do
    CUDA_VISIBLE_DEVICES=1 nohup python -u ads/run.py $s > logfiles/ads_$s.txt
    CUDA_VISIBLE_DEVICES=1 nohup python -u cds/run.py $s > logfiles/cds_$s.txt
done
