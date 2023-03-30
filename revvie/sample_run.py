
f = open('delaunay_triangulation_nonrigid.py')
source = f.read()
exec(source)


#######################
#######################
#######################


# Test case

# with open('/D/Dropbox/matched datasets/BCM27679/matching/tables/all_geneseq_cells.csv', newline='') as csvfile:
#     full_stack_match = csv.reader(csvfile, delimiter=" ")

# Raw data is in y-x-ID
with open("/D/Dropbox/matched datasets/BCM27679/matching/centroids/geneseq_slices_centroids.txt", newline='') as csvfile:
    all_slice_cent = np.loadtxt(csvfile, delimiter=" ")
    
padded_struct = np.load("/D/Dropbox/matched datasets/BCM27679/matching/centroids/padded_struct_centroids.npy")

rot_struct = np.array([[padded_struct[padded_struct[:, 3] == i, 0].mean(),
                            padded_struct[padded_struct[:, 3] == i, 1].mean(),
                            padded_struct[padded_struct[:, 3] == i, 2].mean(), i]
                           for i in pd.unique(padded_struct[:, 3])])

matches = pd.read_csv("/D/Dropbox/matched datasets/BCM27679/matching/tables/all_matched_geneseq_cells.csv")
slice_match =  matches.iloc[:, [2, 3, 4, 5, 1]]
stack_match_units = matches.iloc[:, [10]]

# Raw data is in z-y-x-ID
stack_match = np.array([[rot_struct[rot_struct[:, 3] == i, 2][0],
                            rot_struct[rot_struct[:, 3] == i, 1][0],
                            rot_struct[rot_struct[:, 3] == i, 0][0], i]
                           for i in matches['sunit_ID']])
random_seed = None  # Seed for torch random number generator, if desired.


#######################
#######################
#######################

# Convert data to x-y-z-ID
all_slice_cent = all_slice_cent[:, [0, 2, 1, 3]]

# Convert data to x-y-ID
rot_struct = rot_struct[:, [2, 1, 0, 3]]

# Unit IDs of manual matches.
match_arr = np.array(matches)[:, [1, 10]]
match_arr[:, 1] = match_arr[:, 1].astype(int)

#######################
#######################
#######################

torch.set_printoptions(precision=10)

# Piecewise affine
pc_return = piecewise_affine_cent(all_slice_cent, rot_struct, match_arr)
# torch.save(pc_return, '/D/ImageMatch/superaffine.pt')

# Nonrigid
near_invivo_cents = pc_return['invivo_units'].iloc[np.where(np.isin(pc_return['invivo_units']['ID'], 
                                                      pc_return['matching_probs']['StackUnit']))[0], :][['X', 'Y', 'Z', 'ID']]
aff_invitro_coords = pc_return['invitro_units_trans'].loc[:, ['3D_X', '3D_Y', '3D_Z', 'ID']]
test_nonrigid = nonrigid_demon(aff_invitro_cent = aff_invitro_coords,
                                invivo_cent = near_invivo_cents,
                                match_df = pc_return['manual_matches'],
                                threshes = pc_return['matchprob_filts'])
# torch.save(test_nonrigid, '/D/ImageMatch/supernonrigid.pt')

# Automatic mathces
automatic_matches = automatch(test_nonrigid, rot_struct, all_slice_cent)