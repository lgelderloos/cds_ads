import json
import numpy as np

def print_metrics(results_set, model, round_to=2):
    results = results_set[model]
    medr = np.mean([results[seed]['medr'] for seed in results])
    rec1 = np.mean([results[seed]['recall']['1'] for seed in results])
    rec5 = np.mean([results[seed]['recall']['5'] for seed in results])
    rec10 = np.mean([results[seed]['recall']['10'] for seed in results])
    print(round(medr, round_to), "\t",
          round(rec1, round_to), '\t',
          round(rec5, round_to), '\t',
          round(rec10, round_to))

# natural speech
print("Results for models trained on natural speech")
results_on_ADS = json.load(open("crosstest_results/ADS.json", "r"))
results_on_CDS = json.load(open("crosstest_results/CDS.json", "r"))
results_on_joined = json.load(open("joinedtest_results/naturalspeech.json", 'r'))

print('Model trained on CDS')
print('testset\tmedr\tr1\tr5\tr10')
print('CDS', end='\t')
print_metrics(results_on_CDS, 'CDS')
print('ADS', end='\t')
print_metrics(results_on_ADS, 'CDS')
print('joined', end='\t')
print_metrics(results_on_joined, 'CDS')

print('Model trained on ADS')
print('testset\tmedr\tr1\tr5\tr10')
print('CDS', end='\t')
print_metrics(results_on_CDS, 'ADS')
print('ADS', end='\t')
print_metrics(results_on_ADS, 'ADS')
print('joined', end='\t')
print_metrics(results_on_joined, 'ADS')

# synthetic speech
print("Results for models trained on synthetic speech")
results_on_ADS_synth = json.load(open("crosstest_results/ADS_synth.json", "r"))
results_on_CDS_synth = json.load(open("crosstest_results/CDS_synth.json", "r"))
results_on_joined_synth = json.load(open("joinedtest_results/syntheticspeech.json", 'r'))

print('Model trained on CDS')
print('testset\tmedr\tr1\tr5\tr10')
print('CDS', end='\t')
print_metrics(results_on_CDS_synth, 'CDS_synth')
print('ADS', end='\t')
print_metrics(results_on_ADS_synth, 'CDS_synth')
print('joined', end='\t')
print_metrics(results_on_joined_synth, 'CDS_synth')

print('Model trained on ADS')
print('testset\tmedr\tr1\tr5\tr10')
print('CDS', end='\t')
print_metrics(results_on_CDS_synth, 'ADS_synth')
print('ADS', end='\t')
print_metrics(results_on_ADS_synth, 'ADS_synth')
print('joined', end='\t')
print_metrics(results_on_joined_synth, 'ADS_synth')
