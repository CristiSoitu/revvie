import argparse
import numpy as np
from revvie.revvie_helpers import *
from revvie.delaunay_triangulation_nonrigid import *
import time


#def run_automatch_outside(args, slices=[]):
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run automatch on a list of slices')
    # I want to pass an entire list of argparse arguments to this function
    parser.add_argument('--matcher_path', type=str, help='path to matcher')
    parser.add_argument('--positions', nargs='+', help='positions to run automatch on')
    parser.add_argument('--slices', nargs='+', help='slices to run automatch on')
    args = parser.parse_args()


    if len(args.slices) == 0:
        slices = args.positions
    
    stack_path = args.matcher_path + 'stack/'
    vivo_func_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)
    func_ids = vivo_func_points[:, 3]

    for slice in slices:

        vivo_points = np.load(stack_path + 'struct_vivo_points.npy').astype(np.float32)
        #select only the first occurence of ids in column 3
        vivo_points = vivo_points[np.unique(vivo_points[:, 3], return_index=True)[1]]
        vivo_points[:, [0, 2]] = vivo_points[:, [2, 0]]


        slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
        create_backup(slice_path + '../')
        #wait 3 seconds to make sure the backup is created
        time.sleep(3)
        matches = np.loadtxt(slice_path + 'matches.txt')
        if len(matches) > 3:
            vitro_points = np.loadtxt(slice_path + 'true_points.txt')

            vitro_points = vitro_points[:, [0, 2, 1, 3]]

            output_path = quick_dir(slice_path, 'automatch')


            # Piecewise affine


            pc_return = piecewise_affine_cent(vitro_points, vivo_points, matches)

            for key in pc_return.keys():
                #check to see if the key is a dataframe
                if isinstance(pc_return[key], pd.DataFrame):
                    pc_return[key].to_csv(output_path + 'pc_return_' + key + '.csv')


            # Nonrigid
            near_invivo_cents = pc_return['invivo_units'][np.where(np.isin(pc_return['invivo_units'][:, 3], 
                                                            pc_return['matching_probs']['StackUnit']))[0], :]
            aff_invitro_coords = pc_return['invitro_units_trans'].loc[:, ['3D_X', '3D_Y', '3D_Z', 'ID']]

            test_nonrigid = nonrigid_demon(aff_invitro_cent = aff_invitro_coords,
                                    invivo_cent = pc_return['invivo_near_units'],
                                    match_df = pc_return['manual_matches'],
                                    invivo_l = pc_return['invivo_lookup'],
                                    threshes = pc_return['matchprob_filts'],
                                    pot_match_tab = pc_return['matching_probs'])

            #save all fields of test_nonrigid to file in output_path 
            for key in test_nonrigid.keys():
                if isinstance(test_nonrigid[key], pd.DataFrame):
                    test_nonrigid[key].to_csv(output_path + 'test_nonrigid_' + key + '.csv', index=False)

            # Automatic mathces
            automatic_matches = automatch(test_nonrigid, vivo_points, vitro_points)

            automatic_matches.to_csv(output_path + 'automatic_matches.csv', index=False)

            #transformed_vitro_points = test_nonrigid['invitro_units']
            #extract columns 'SliceUnit' and 'StackUnit' from automatic_matches
            predicted_matches_id = automatic_matches.loc[:, ['SliceUnit', 'StackUnit']]
            #turn this into a numpy array
            predicted_matches = predicted_matches_id.to_numpy().astype(np.int32)
            #save only those where StackUnit is in func_ids
            print('Predicted ' + str(len(predicted_matches)) + ' struct matches.')
            predicted_matches = predicted_matches[np.where(np.isin(predicted_matches[:, 1], func_ids))[0], :]
            print('Predicted ' + str(len(predicted_matches)) + ' func matches.')
            
            rejected = np.loadtxt(slice_path + 'rejected.txt')
            #save only those rows not in rejected
            overlap = matching_rows(predicted_matches, rejected)
            overlap_indices = []
            #find overlap indices in predicted_matches
            for i in range(len(overlap)):
                for j in range(len(predicted_matches)):
                    if overlap[i, 0] == predicted_matches[j, 0] and overlap[i, 1] == predicted_matches[j, 1]:
                        overlap_indices.append(j)
            predicted_matches = np.delete(predicted_matches, overlap_indices, axis=0)
            rejected = np.empty((0, 2), dtype=np.int32)
            np.savetxt(slice_path + 'rejected.txt', rejected, fmt='%i')
            print('Predicted ' + str(len(predicted_matches)) + ' matches after removing rejected.')

            matches = np.loadtxt(slice_path + 'matches.txt')
            overlap = matching_rows(predicted_matches, matches)
            overlap_indices = []
            #find overlap indices in predicted_matches
            for i in range(len(overlap)):
                for j in range(len(predicted_matches)):
                    if overlap[i, 0] == predicted_matches[j, 0] and overlap[i, 1] == predicted_matches[j, 1]:
                        overlap_indices.append(j)
            predicted_matches = np.delete(predicted_matches, overlap_indices, axis=0)
            print('Predicted ' + str(len(predicted_matches)) + ' matches after removing matches.')

            accepted = np.empty((0, 2), dtype=np.int32)
            np.savetxt(slice_path + 'accepted.txt', rejected, fmt='%i')

            np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')
            return predicted_matches

