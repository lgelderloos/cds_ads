for s in 123 234 345
do
    CUDA_VISIBLE_DEVICES=0 nohup python -u ads_synth/run.py $s > logfiles/ads_synth_$s.txt
    CUDA_VISIBLE_DEVICES=0 nohup python -u cds_synth/run.py $s > logfiles/cds_synth_$s.txt
done
