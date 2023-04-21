import numpy as np
from revvie.revvie_helpers import *
import seaborn as sns
import matplotlib.pyplot as plt

def compute_prediction_accuracy(args, slices=[]):
    if slices == []:
        slices = args.positions

    stats_path = quick_dir(args.matcher_path, 'stats')
    initial_matches_total = []
    rejected_total = []
    accepted_total = []
    predicted_total = []
    percent_accepted = []
    percent_rejected = []
    for slice in slices:
        slice_path = args.matcher_path + 'slices/' + slice + '/'
        latest_state_path = slice_path + 'latest/'
        backup_folders = [f for f in os.listdir(slice_path) if os.path.isdir(slice_path + f)]
        backup_folders = [f for f in backup_folders if f != 'latest']
        backup_folders.sort(key=lambda x: x[0], reverse=True)
        backup_paths = [slice_path + f + '/' for f in backup_folders]
        previous_state_path = backup_paths[0]

        initial_matches = np.loadtxt(previous_state_path + 'matches.txt')
        accepted = np.loadtxt(latest_state_path + 'accepted.txt')
        rejected = np.loadtxt(latest_state_path + 'rejected.txt')
        
        initial_matches_total.append(len(initial_matches))
        rejected_total.append(len(rejected))
        accepted_total.append(len(accepted))
        predicted_total.append(len(accepted) + len(rejected))
        percent_accepted.append(len(accepted) / (len(accepted) + len(rejected)))
        percent_rejected.append(len(rejected) / (len(accepted) + len(rejected)))


    # make pandas dataframe [slice, initial_matches, accepted, rejected]
    
    matches_pd = pd.DataFrame({'slice': slices,
                                 'initial_matches': initial_matches_total,
                                    'predicted': predicted_total,
                                    'accepted': accepted_total,
                                    'rejected': rejected_total, 
                                    'percent_accepted': percent_accepted,
                                    'percent_rejected': percent_rejected})
    matches_pd.to_csv(stats_path + 'stats.csv', index=False)


def matching_rows(A, B):
    matches = [i for i in range(B.shape[0]) if np.any(np.all(A == B[i], axis=1))]
    if len(matches) == 0:
        return B[matches]
    return np.unique(B[matches], axis=0)

def compare_matching(args, matcher1_path, matcher2_path, slices=[]):
    stats_path = quick_dir(args.matcher_path, 'stats')
    if slices == []:
        slices = args.positions

    matcher1_matches_total = []
    matcher2_matches_total = []

    agreed_matches_total = []
    
    matcher1_correct_total = []
    matcher2_correct_total = []

    matcher1_correct_percent = []
    matcher2_correct_percent = []

    common_vitro_attempted = []
    common_vivo_attempted = []





    for slice in slices:

        slice_path = args.matcher_path + 'slices/' + slice + '/'
        latest_state_path = slice_path + 'latest/'
        backup_folders = [f for f in os.listdir(slice_path) if os.path.isdir(slice_path + f)]
        backup_folders = [f for f in backup_folders if f != 'latest']
        backup_folders.sort(key=lambda x: x[0], reverse=True)
        backup_paths = [slice_path + f + '/' for f in backup_folders]
        previous_state_path = backup_paths[0]

        initial_matches = np.loadtxt(previous_state_path + 'matches.txt')
        matches = np.loadtxt(latest_state_path + 'matches.txt')
        accepted = np.loadtxt(latest_state_path + 'accepted.txt')
        rejected = np.loadtxt(latest_state_path + 'rejected.txt')


        # load matches from matcher1
        matcher1_slice_path = matcher1_path + 'slices/' + slice + '/'
        matcher1_latest_state_path = matcher1_slice_path + 'latest/'
        matcher1_backup_folders = [f for f in os.listdir(matcher1_slice_path) if os.path.isdir(matcher1_slice_path + f)]
        matcher1_backup_folders = [f for f in matcher1_backup_folders if f != 'latest']
        matcher1_backup_folders.sort(key=lambda x: x[0], reverse=True)
        matcher1_backup_paths = [matcher1_slice_path + f + '/' for f in matcher1_backup_folders]
        matcher1_previous_state_path = matcher1_backup_paths[0]
        matcher1_initial_matches = np.loadtxt(matcher1_previous_state_path + 'matches.txt')

        # load matches from matcher2
        matcher2_slice_path = matcher2_path + 'slices/' + slice + '/'
        matcher2_latest_state_path = matcher2_slice_path + 'latest/'
        matcher2_backup_folders = [f for f in os.listdir(matcher2_slice_path) if os.path.isdir(matcher2_slice_path + f)]
        matcher2_backup_folders = [f for f in matcher2_backup_folders if f != 'latest']
        matcher2_backup_folders.sort(key=lambda x: x[0], reverse=True)
        matcher2_backup_paths = [matcher2_slice_path + f + '/' for f in matcher2_backup_folders]
        matcher2_previous_state_path = matcher2_backup_paths[0]
        matcher2_initial_matches = np.loadtxt(matcher2_previous_state_path + 'matches.txt')

        matcher1_matches_total.append(len(matcher1_initial_matches))
        matcher2_matches_total.append(len(matcher2_initial_matches))

        overlap = matching_rows(matcher1_initial_matches, matcher2_initial_matches)
        agreed_matches_total.append(len(overlap))

        matcher1_correct = matching_rows(matcher1_initial_matches, matches)
        matcher1_correct_total.append(len(matcher1_correct))

        matcher2_correct = matching_rows(matcher2_initial_matches, matches)
        matcher2_correct_total.append(len(matcher2_correct))

        matcher1_correct_percent.append(len(matcher1_correct) / len(matcher1_initial_matches))
        matcher2_correct_percent.append(len(matcher2_correct) / len(matcher2_initial_matches))


        common_vitro = 0
        common_vivo = 0
        for match in matcher1_initial_matches:
            if match[0] in matcher2_initial_matches[:,0]:
                common_vitro += 1
            if match[1] in matcher2_initial_matches[:,1]:
                common_vivo += 1

        common_vitro_attempted.append(common_vitro)
        common_vivo_attempted.append(common_vivo)

        

    # make dataframe [slice, matcher1_matches, matcher2_matches, agreed_matches, matcher1_correct, matcher2_correct]
    matches_pd = pd.DataFrame({'slice': slices,
                                'matcher1_matches': matcher1_matches_total,
                                'matcher2_matches': matcher2_matches_total,
                                'agreed_matches': agreed_matches_total,
                                'matcher1_correct': matcher1_correct_total,
                                'matcher2_correct': matcher2_correct_total, 
                                'matcher1_correct_percent': matcher1_correct_percent,
                                'matcher2_correct_percent': matcher2_correct_percent, 
                                'common_vitro_attempted': common_vitro_attempted,
                                'common_vivo_attempted': common_vivo_attempted})
    matches_pd.to_csv(stats_path + 'compare_matchers.csv', index=False)


def compare_automatch(args, matchers_path=[], slices=[]):
    stats_path = quick_dir(args.matcher_path, 'stats')
    if slices == []:
        slices = args.positions

    for matcher_id, matcher_path in enumerate(matchers_path):
        
        matcher_matches_total = []

        matcher_predictions_total = []

        matcher_correct_total = []

        matcher_correct_percent = []


        matcher_accepted_total = []
        matcher_rejected_total = []
        

        for slice in slices:

            slice_path = args.matcher_path + 'slices/' + slice + '/'
            latest_state_path = slice_path + 'latest/'
            backup_folders = [f for f in os.listdir(slice_path) if os.path.isdir(slice_path + f)]
            backup_folders = [f for f in backup_folders if f != 'latest']
            backup_folders.sort(key=lambda x: x[0], reverse=True)
            backup_paths = [slice_path + f + '/' for f in backup_folders]
            previous_state_path = backup_paths[0]

            initial_matches = np.loadtxt(previous_state_path + 'matches.txt')
            matches = np.loadtxt(latest_state_path + 'matches.txt')
            accepted = np.loadtxt(latest_state_path + 'accepted.txt')
            rejected = np.loadtxt(latest_state_path + 'rejected.txt')


            matcher_slice_path = matcher_path + 'slices/' + slice + '/'
            matcher_latest_state_path = matcher_slice_path + 'latest/'
            matcher_backup_folders = [f for f in os.listdir(matcher_slice_path) if os.path.isdir(matcher_slice_path + f)]
            matcher_backup_folders = [f for f in matcher_backup_folders if f != 'latest']
            matcher_backup_folders.sort(key=lambda x: x[0], reverse=True)
            matcher_backup_paths = [matcher_slice_path + f + '/' for f in matcher_backup_folders]
            matcher_previous_state_path = matcher_backup_paths[0]
            matcher_initial_matches = np.loadtxt(matcher_previous_state_path + 'matches.txt')
            matcher_accepted = np.loadtxt(matcher_latest_state_path + 'accepted.txt')
            matcher_rejected = np.loadtxt(matcher_latest_state_path + 'rejected.txt')
            matcher_matches = np.loadtxt(matcher_latest_state_path + 'matches.txt')

            matcher_matches_total.append(len(matcher_initial_matches))
            matcher_accepted_total.append(len(matcher_accepted))
            matcher_rejected_total.append(len(matcher_rejected))
            matcher_predictions_total.append(len(matcher_accepted) + len(matcher_rejected))

            matcher_correct = matching_rows(matcher_matches, matches)
            matcher_correct_total.append(len(matcher_correct))

            matcher_correct_percent.append(len(matcher_correct) / len(matcher_matches))

            
        matcher_percentage_accepted = [a / (a + r) for a, r in zip(matcher_accepted_total, matcher_rejected_total)]
        matcher_pecentage_rejected = [r / (a + r) for a, r in zip(matcher_accepted_total, matcher_rejected_total)]

        # make dataframe [slice, matcher1_matches, matcher2_matches, agreed_matches, matcher1_correct, matcher2_correct]
        matches_pd = pd.DataFrame({'slice': slices,
                                    'matcher_matches': matcher_matches_total,
                                    'matcher_predictions': matcher_predictions_total,
                                    'matcher_accepted_total': matcher_accepted_total,
                                    'matcher_rejected_total': matcher_rejected_total,
                                    'matcher_percentage_accepted': matcher_percentage_accepted,
                                    'matcher_pecentage_rejected': matcher_pecentage_rejected, 
                                    'matcher_correct_total': matcher_correct_total, 
                                    'matcher_correct_percent': matcher_correct_percent})
        matches_pd.to_csv(stats_path + 'matcher_' + str(matcher_id) + '_automatch' +'.csv', index=False)












