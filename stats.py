from revvie.revvie_classes import *
from revvie.revvie_helpers import *
from revvie.stats_helpers import *
import warnings


#dataset = 'BCM28382'
dataset = 'BCM27679_1'

dataset_path = '/Users/soitu/Desktop/datasets/' + dataset + '/revvie/'

config_file = dataset_path + 'revvie_config.toml'


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        args = ConfigReader(config_file)
        #args.matcher_name = create_matcher_profile()
        args.matcher_name = 'Cristi'
        args.dataset_path = dataset_path
        args.matcher_path = quick_dir(args.dataset_path, 'matcher_' + args.matcher_name)

        matcher1 = 'Erica'
        matcher2 = 'Jingyang'
        matcher1_path = quick_dir(args.dataset_path, 'matcher_' + matcher1)
        matcher2_path = quick_dir(args.dataset_path, 'matcher_' + matcher2)
        slices = ['Pos19', 'Pos20', 'Pos22', 'Pos23', 'Pos28']
        #slices = []
        #compute_prediction_accuracy(args, slices)
        compare_matching(args, matcher1_path, matcher2_path, slices)
        compare_automatch(args, [matcher1_path, matcher2_path], slices)
        
main()