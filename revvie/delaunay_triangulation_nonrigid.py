import numpy as np
import scipy
import statistics as stat
from scipy import spatial
from scipy import stats
import math
import csv
import torch
import pandas as pd
from torch import optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import random

#######################
#######################
#######################

# Test if two line segments intersect
# p1, p2: 2 by 2 array of coordinate endpoints for line segments.
def line_seg_intersect(p1, p2):
    
    # Directionality of 3 points
    def p_orient(a, b, c):
        return((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))
    
    return(p_orient(p1[0, :], p2[0, :], p2[1, :]) != p_orient(p1[1, :], p2[0, :], p2[1, :]) and \
           p_orient(p1[0, :], p1[1, :], p2[0, :]) != p_orient(p1[0, :], p1[1, :], p2[1, :]))

#######################
#######################
#######################

 # Check if quadrilateral is proper.
 ## Intersection between 1-4 and 2-3 or 1-2 and 3-4.
 ## If intersection, do a swap.
def proper_quad_test(coords):
    if not line_seg_intersect(coords[[0, 3], :], coords[[1, 2], :]) and \
        not line_seg_intersect(coords[[0, 1], :], coords[[2, 3], :]):
            return(coords)
    for i in range(0, 3):
        coords_temp = coords.copy()
        coords_temp[[i + 1, i], :] = coords_temp[[i, i + 1], :]
        if not line_seg_intersect(coords_temp[[0, 3], :], coords_temp[[1, 2], :]) and \
            not line_seg_intersect(coords_temp[[0, 1], :], coords_temp[[2, 3], :]):
                return(coords_temp)
    return(coords)

#######################
#######################
#######################

# Test if a point is within a quadrilateral
# coords: 2 by 4 array of coordinate points for the polygon.
# point: 2 by 1 array of test point.
def quad_intersect_test(coords, point, xmin):
    
    # Left intersection test.
    left_seg = np.array([[xmin, point[1]], [point[0], point[1]]])
    int_sum = 0
    for c in range(0, len(coords)):
        if line_seg_intersect(coords[[c, ((c + 1) % len(coords))], :], left_seg):
            int_sum += 1
    # If number of intersection is odd, point is in polygon.
    return(int_sum % 2 == 1)

#######################
#######################
#######################

# Apply 2-D to 3-D affine transformation.     
# X: n by 2 set of coordinates
# A: 3 by 2 affine transform tensor
# b: length 3 tensor, translation
def affine_product(X, A, b):
    return(torch.einsum('ij,kj->ki', (A.double(), X.double())) + b)

#######################
#######################
#######################

# Define possible matches.
# Use GMM metric to calculate probabilities.
# matched_cent:  x-y-z coordinates, ID of transformed (slice) center.
# true_cent: x-y-z coordinates, ID of true (stack) center.
# manual: if there are manual matches, unit IDs of matched (col 0) and true (col 1) manual matches.
def gmm_prob(matched_cent, true_cent, thresh_x, thresh_y, thresh_z, 
             manual = None, tensor = False, delta_cov = None):
    # Initialize match matrix
    total_df = pd.DataFrame(columns=['TransUnit', 'TrueUnit',
                                         'trans_x', 'trans_y', 'trans_z',
                                         'true_x', 'true_y', 'true_z',
                                         'dx', 'dy', 'dz'])
    
    # Filter through all in vitro and in vivo neurons for plausible matches
    for m in range(0, len(matched_cent)): 
        #If in manual matches, skip.
        if tensor:
            if np.all(manual != None) and torch.isin(matched_cent[m, 3], torch.tensor(manual[:, 0])):
                p = np.where(torch.isin(torch.tensor(manual[:, 0]), matched_cent[m, 3]).numpy())[0]
                q = np.where(torch.isin(true_cent[:, 3], torch.tensor(manual[p, 1])).numpy())[0]
                if q.size > 0:
                    dx = true_cent[q, 0] - matched_cent[m, 0]
                    dy = true_cent[q, 1] - matched_cent[m, 1]
                    dz = true_cent[q, 2] - matched_cent[m, 2]
                    total_df.loc[len(total_df.index)] = [matched_cent[m, 3].item(), true_cent[q, 3].item(),
                                                     matched_cent[m, 0].item(), matched_cent[m, 1].item(),
                                                     matched_cent[m, 2].item(), true_cent[q, 0].item(),
                                                     true_cent[q, 1].item(), true_cent[q, 2].item(),
                                                     dx.item(), dy.item(), dz.item()]
                continue
        else: 
            if np.all(manual != None) and np.isin(matched_cent[m, 3], manual[:, 0]):
                p = np.where(np.isin(manual[:, 0], matched_cent[m, 3]))[0]
                q = np.where(np.isin(true_cent[:, 3], manual[p, 1]))[0]
                if q.size > 0:
                    dx = true_cent[q, 0] - matched_cent[m, 0]
                    dy = true_cent[q, 1] - matched_cent[m, 1]
                    dz = true_cent[q, 2] - matched_cent[m, 2]
                    total_df.loc[len(total_df.index)] = [matched_cent[m, 3], true_cent[q, 3][0],
                                                     matched_cent[m, 0], matched_cent[m, 1],
                                                     matched_cent[m, 2], true_cent[q, 0][0],
                                                     true_cent[q, 1][0], true_cent[q, 2][0],
                                                     dx[0], dy[0], dz[0]]
                continue
        for r in range(0, len(true_cent)):
            #If in manual matches, skip.
            if tensor:
                if np.all(manual != None) and torch.isin(true_cent[r, 3], torch.tensor(manual[:, 1])):
                    continue
            else: 
                if np.all(manual != None) and np.isin(true_cent[r, 3], manual[:, 1]):
                    continue
            dx = true_cent[r, 0] - matched_cent[m, 0]
            dy = true_cent[r, 1] - matched_cent[m, 1]
            dz = true_cent[r, 2] - matched_cent[m, 2]
            if abs(dx) < thresh_x and abs(dy) < thresh_y and abs(dz) < thresh_z:
                if tensor:
                    total_df.loc[len(total_df.index)] = [matched_cent[m, 3].item(), true_cent[r, 3].item(),
                                                     matched_cent[m, 0].item(), matched_cent[m, 1].item(),
                                                     matched_cent[m, 2].item(), true_cent[r, 0].item(),
                                                     true_cent[r, 1].item(), true_cent[r, 2].item(),
                                                     dx.item(), dy.item(), dz.item()]
                else:
                    total_df.loc[len(total_df.index)] = [matched_cent[m, 3], true_cent[r, 3],
                                                     matched_cent[m, 0], matched_cent[m, 1],
                                                     matched_cent[m, 2], true_cent[r, 0],
                                                     true_cent[r, 1], true_cent[r, 2],
                                                     dx, dy, dz]
    
    # Covariance matrix of all possible shifts.
    delta_dat = total_df.loc[:, ['dx', 'dy', 'dz']]
    if len(delta_dat) == 0:
        print("No potential matches found.")
        return(None)
    if np.any(delta_cov == None):
        delta_cov = np.matmul(delta_dat.transpose(), delta_dat) / len(delta_dat)
    total_df.index = pd.RangeIndex(len(total_df))  
    
    # Space for probaiblity calculations      
    total_df['dens'] = 0
    total_df['prob'] = 0
    
    for u in range(0, len(matched_cent)):
        if tensor:
            scan_match_rows = np.where(total_df['TransUnit'] == matched_cent[u, 3].item())
        else:
            scan_match_rows = np.where(total_df['TransUnit'] == matched_cent[u, 3])
        if len(scan_match_rows[0]) == 0:
            continue
        elif len(scan_match_rows[0]) == 1:
            total_df.loc[scan_match_rows[0][0], 'prob'] = 1
            total_df.loc[scan_match_rows[0][0], 'dens'] = \
                scipy.stats.multivariate_normal.pdf(total_df.loc[scan_match_rows[0][0], ['dx', 'dy', 'dz']], 
                                                    [0, 0, 0], cov = delta_cov)
        else:
            temp_prob = np.zeros(len(scan_match_rows[0]))
            ## Use multivariate Gaussian for GMM calculation.
            for j in range(0, len(scan_match_rows[0])):
                temp_prob[j] = scipy.stats.multivariate_normal.pdf(total_df.loc[scan_match_rows[0][j], ['dx', 'dy', 'dz']], 
                                                                   [0, 0, 0], cov = delta_cov)
            total_df.loc[scan_match_rows[0], 'dens'] = temp_prob 
            total_df.loc[scan_match_rows[0], 'prob'] = temp_prob / sum(temp_prob)

    total_df = total_df.sort_values('TransUnit')
    return(total_df)

#######################
#######################
#######################

# Match filtering
# match1, match2: x-y-z-ID of centroids.
# gr: scaling for threshold filter.   
def match_filter_dict_3d(match1, match2, gx, gy, gz, same_units = False):
    match_dict = {}
    for a in range(0, len(match1)):
        pot_matches = []
        pot_dist = []
        for b in range(0, len(match2)):
            if(match1[a, 3] == match2[b, 3] and same_units):
                continue
            dx = match1[a, 0] - match2[b, 0]
            dy = match1[a, 1] - match2[b, 1]
            dz = match1[a, 2] - match2[b, 2]
            if abs(dx) < gx and abs(dy) < gy and abs(dz) < gz:
                pot_matches.append(match2[b, 3])
                pot_dist.append(dx ** 2 + dy ** 2 + dz ** 2)
        match_dict[str(match1[a, 3])] = {'matchIDs': pot_matches, 'dists': pot_dist}
    return(match_dict)

#######################
#######################
#######################

# Piecewise affine transformation
# Requires at least 4 matched points from all slices.
# Not all in vitro units are transformed - currently must be inside of triangulation.
# invitro_cent: slice-x-y-ID of in vitro (slice) centroids.
# invivo_cent: x-y-z-ID of in vivo (stack) centroids.
## invitro_cent and invivo_cent must contain coordinate of matched centroids.
# match_df: unit IDs of in vitro (col 0) and in vivo (col 1) manual matches.
# conf_alpha: p-value tolerance for confidence interval.
# z_range_scale: extra range on z-axis for finding potential matches
def piecewise_affine_cent(invitro_cent, invivo_cent, match_df, conf_alpha=0.05,
                          z_range_scale = 1.75):
    ## Initial parameterization
    lr_linear = 0.001  # learning rate / step size for the linear part of the affine matrix
    lr_translation = 1  # learning rate / step size for the translation vector
    min_affine_iters = 3000
    max_affine_iters = 30000
    tr = 10  # volume thrshold constant for GMM

    invivo_avg_cent = np.array([[invivo_cent[invivo_cent[:, 3] == i, 0].mean(),
                                invivo_cent[invivo_cent[:, 3] == i, 1].mean(),
                                invivo_cent[invivo_cent[:, 3] == i, 2].mean(), i]
                               for i in pd.unique(invivo_cent[:, 3])])
    # Preserve order
    invitro_matches = invitro_cent[[np.where(np.isin(invitro_cent[:, 3], x))[
        0][0] for x in match_df[:, 0]]]
    invivo_matches = invivo_avg_cent[[np.where(np.isin(invivo_cent[:, 3], x))[
        0][0] for x in match_df[:, 1]]]
    sliceids = np.unique(invitro_matches[:, 0])

    # Create transform information saving.
    affine_res_matrices = []
    invitro_trans_df = pd.DataFrame(invitro_cent[:, [0, 1, 2, 3]],
                                    columns=['Slice', '2D_X', '2D_Y', 'ID', ])

    invitro_trans_df[['3D_X', '3D_Y', '3D_Z']] = 0
    invitro_trans_df[['Del_Tri']] = 0

    # Create Delaunay triangle ID saving.
    invivo_del_df = pd.DataFrame(invivo_avg_cent[:, :],
                                 columns=['X', 'Y', 'Z', 'ID'])
    invivo_del_df[['Slice_Num', "Del_Tri"]] = -1
    stack_del_dict = {}
    # For each slice
    for s in range(0, len(sliceids)):
        slicenum = sliceids[s]
        print(slicenum)

        ## Find all corresponding centroids in the slice.
        slicenum_cents = invitro_cent[np.where(invitro_cent[:, 0] == slicenum)]
        min_x = min(slicenum_cents[:, 1]) - \
            (max(slicenum_cents[:, 1]) - min(slicenum_cents[:, 1])) * 0.01

        ## Find matches corresponding to slice.
        stack_core = invivo_matches[np.where(
            invitro_matches[:, 0] == slicenum)]
        slice_core = invitro_matches[np.where(
            invitro_matches[:, 0] == slicenum)]

        if(len(slice_core) < 4):
            print("Slice " + str(int(slicenum)) +
                  " does not contain at least 4 manual matches.")
            continue

        # Calculate Delaunay triangulation.
        stack_del = scipy.spatial.Delaunay(
            stack_core[:, 0:3], qhull_options="QJ")
        stack_del_dict["slice" + str(slicenum)] = stack_del
        stack_tri_cents = stack_del.find_simplex(invivo_avg_cent[:, 0:3])
        # print(stack_tri_cents[np.where(stack_tri_cents != -1)[0]])

        # Attached Delaunay triangulation info back to in vivo table.
        invivo_del_df.loc[np.where(stack_tri_cents != -1)
                          [0], 'Slice_Num'] = slicenum
        invivo_del_df.loc[np.where(
            stack_tri_cents != -1)[0], 'Del_Tri'] = stack_tri_cents[np.where(stack_tri_cents != -1)[0]]
        # For each triangle
        for t in range(0, len(stack_del.simplices)):
            ## Take 4 guideposts from 3D
            ## Find corresponding 2D guideposts.
            stack_guide = torch.as_tensor(
                stack_core[stack_del.simplices[t], 0:3])
            slice_guide = torch.as_tensor(
                slice_core[stack_del.simplices[t], 1:3])
            ## Parameter initialization
            rig_z = 0.0
            rig_y = 0.0
            rig_x = 0.0
            # first two columns of rotation matrix
            linear = torch.nn.Parameter(torch.eye(3)[:, :2])
            translation = torch.nn.Parameter(torch.tensor(
                [rig_x, rig_y, rig_z]))  # translation vector
            affine_optimizer = optim.Adam([{'params': linear, 'lr': lr_linear},
                                           {'params': translation, 'lr': lr_translation}])

            dist_loss_old = math.inf
            # Optimize
            for i in range(max_affine_iters):
                # Zero gradients
                affine_optimizer.zero_grad()

                # Compute gradients
                pred_slice_guides = affine_product(
                    slice_guide, linear, translation)
                ## Euclidean distance as optimization metric.
                dist_loss = (((stack_guide - pred_slice_guides)**2)).sum()

                if (dist_loss_old - dist_loss) / dist_loss < 10 ** -9 and i > min_affine_iters:
                    break
                dist_loss_var = torch.std(
                    torch.sum(((stack_guide - pred_slice_guides)**2), dim=1))
                # print('Dist at iteration {}: {:5.4f}'.format(i, dist_loss))

                # Update
                dist_loss.backward()
                affine_optimizer.step()
                dist_loss_old = dist_loss

            # 2nd moment information
            def distance_metric_hess(lin, trans):
                psg = affine_product(slice_guide, lin, trans)
                return((stack_guide - psg)**2).sum()

            affine_linear = linear.detach().clone()
            affine_translation = translation.detach().clone()
            aff = torch.autograd.functional.hessian(distance_metric_hess,
                                                    (affine_linear, affine_translation))
            linear_g2 = torch.tensor([[aff[0][0][0, 0, 0, 0], aff[0][0][0, 1, 0, 1]],
                                      [aff[0][0][1, 0, 1, 0],
                                          aff[0][0][1, 1, 1, 1]],
                                      [aff[0][0][2, 0, 2, 0], aff[0][0][2, 1, 2, 1]]])
            translation_g2 = torch.tensor(
                [aff[1][1][0, 0], aff[1][1][1, 1], aff[1][1][2, 2]])

            # Calculate confidence intervals.
            conf_constant = dist_loss_var * 4 * scipy.stats.norm.ppf(1 - conf_alpha / 2) / \
                scipy.stats.chi2.ppf(
                    (1 - conf_alpha) ** 2, len(affine_linear) + len(affine_translation))
            upperconf_affine_linear = conf_constant * linear_g2 ** -1 + affine_linear
            upperconf_affine_translation = conf_constant * \
                translation_g2 ** -1 + affine_translation
            lowerconf_affine_linear = -conf_constant * linear_g2 ** -1 + affine_linear
            lowerconf_affine_translation = -conf_constant * \
                translation_g2 ** -1 + affine_translation
            sd_affine_linear = dist_loss_var * linear_g2
            sd_affine_translation = dist_loss_var * translation_g2

            ## Save parameters as a list of dicts.
            affine_res_matrices.append({'slice': slicenum, "triangle": t, "alpha": conf_alpha, 
                                        'dist': dist_loss,
                                        "stack_guides": stack_core[stack_del.simplices[t], :],
                                        "slice_guides": slice_core[stack_del.simplices[t], :],
                                        "affine_linear": affine_linear,
                                        "affine_translation": affine_translation,
                                        "sd_linear": sd_affine_linear,
                                        "sd_translation": sd_affine_translation,
                                        "upper_linear": upperconf_affine_linear,
                                        "lower_linear": lowerconf_affine_linear,
                                        "upper_translation": upperconf_affine_translation,
                                        "lower_translation": lowerconf_affine_translation})

            # Apply affine transform from 2D to 3D.
            ## For each slice neuron, find if it belongs to quad and apply transform.
            slice_guide_array = proper_quad_test(
                slice_core[stack_del.simplices[t], 1:3])

            for l in range(0, len(slicenum_cents)):
                
                ## Use intersection test.
                if np.isin(slicenum_cents[l, 3], slice_core[stack_del.simplices[t], 3]) or\
                    quad_intersect_test(slice_guide_array, slicenum_cents[l, 1:3], min_x):
                    if (np.isin(slicenum_cents[l, 3], match_df[:, 0]) and not np.isin(slicenum_cents[l, 3], slice_core[stack_del.simplices[t], 3])):
                        continue
                    ### Apply affine transform.
                    trans_slice_coords = affine_product(torch.tensor(slicenum_cents[l, 1:3]).unsqueeze(dim=0),
                                                        affine_linear, affine_translation)

                    ### Attach back to original matrix of in vitro
                    full_slice_ind = np.where(
                        invitro_cent[:, 3] == slicenum_cents[l, 3])
                    invitro_trans_df.loc[full_slice_ind[0], '3D_X'] = \
                        (trans_slice_coords[0][0].item() + \
                            invitro_trans_df.loc[full_slice_ind[0], '3D_X'] * \
                                invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri']) /\
                            (invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri'] + 1) 
                    invitro_trans_df.loc[full_slice_ind[0], '3D_Y'] = \
                        (trans_slice_coords[0][1].item() + \
                            invitro_trans_df.loc[full_slice_ind[0], '3D_Y'] * \
                                invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri']) /\
                            (invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri'] + 1) 
                    invitro_trans_df.loc[full_slice_ind[0], '3D_Z'] = \
                        (trans_slice_coords[0][2].item() + \
                            invitro_trans_df.loc[full_slice_ind[0], '3D_Z'] * \
                                invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri']) /\
                            (invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri'] + 1) 
                    invitro_trans_df.loc[full_slice_ind[0], 'Del_Tri'] += 1

    # Calculate probabilities of matches overall in the whole table.
    # Smaller set to bigger.
    tx = tr
    ty = tr
    tz = tr * z_range_scale
    invitro_trans_df_rel = invitro_trans_df.iloc[np.where(
        np.array(invitro_trans_df[['Del_Tri']] != 0))[0], :]
        
    matchprob_df = gmm_prob(np.array(invitro_trans_df_rel[['3D_X', '3D_Y', '3D_Z', 'ID']]), 
                            invivo_avg_cent,
                            tx, ty, tz)
    matchprob_df = matchprob_df.rename(columns={'TransUnit': 'SliceUnit', 'TrueUnit': 'StackUnit',
                                                'trans_x': 'slice_x', 'trans_y': 'slice_y', 'trans_z': 'slice_z',
                                                'true_x': 'stack_x', 'true_y': 'stack_y', 'true_z': 'stack_z'})

    ## Return: affine transform parameters, estimated transform in the in vitro, GMM prob table.
    return_dict = {'manual_matches': match_df,
                   'affine_parameters': affine_res_matrices,
                   'invivo_units': invivo_del_df,
                   'invitro_units_trans': invitro_trans_df_rel,
                   'invitro_units_full': invitro_trans_df,
                   'matching_probs': matchprob_df,
                   'matchprob_filts': [tx, ty, tz],
                   'stack_del': stack_del_dict}
    return(return_dict)

#######################
#######################
#######################

# Nonrigid registration on overall data.
# Transforming invitro to invivo (2d to 3d)
## Optimization as GMM metric for demons. 

# aff_invitro_cent: x-y-z-ID of affine-transformed in vitro (slice) centroids.
# invivo_cent: x-y-z-ID of in vivo (stack) centroids.
# random_seed: if desired, set a seed for nonrigid initialization.
# match_df: unit IDs of in vitro (col 0) and in vivo (col 1) manual matches.
# conf_alpha: p-value tolerance for confidence interval.
# threshes: manual distance thresholds for being considered a possible match.


def nonrigid_demon(aff_invitro_cent, invivo_cent, match_df, conf_alpha=0.05,
                   random_seed=None, threshes=None,
                   z_range_scale=1.75):

    if type(random_seed) == int:
        torch.manual_seed(random_seed)

    lr_deformations = 0.5  # learning rate / step size for defpc_return = piecewise_affine_cent(all_slice_cent, rot_struct, match_arr)ormation values
    wd_deformations = 1e-4  # weight decay for deformations; controls their size
    smoothness_factor = 0.01  # factor to keep the deformation field smooth
    dens_factor = 50
    true_factor = 1
    td_factor = 50
    min_demon_iters = 200
    max_demon_iters = 2000
    grid_coef = 0.2
    invivo_coef = 0.2
    ex = 5
    ey = 5
    ez = 2
    tx = 12
    ty = 10
    tz = 10

    # Match scores between invitro points
    invivo_scores = match_filter_dict_3d(np.array(invivo_cent)[:, 0:4],
                                         np.array(invivo_cent)[:, 0:4],
                                         ex, ey, ez, True)
    total_invivo_scores = 0
    for s in invivo_scores:
        invivo_scores[s]['scores'] = [math.exp(-((x * invivo_coef / ((ex ** 2 + ey ** 2 + ez ** 2) ** 0.5)) ** 2))
                                      for x in invivo_scores[s]['dists']]
        total_invivo_scores += sum(invivo_scores[s]['scores'])

    # Match scores between invitro + invitro points
    cross_scores = match_filter_dict_3d(np.array(aff_invitro_cent),
                                        np.array(invivo_cent)[:, 0:4],
                                        ex, ey, ez, False)
    for s in cross_scores:
        cross_scores[s]['scores'] = [math.exp(-((x * grid_coef / ((ex ** 2 + ey ** 2 + ez ** 2) ** 0.5)) ** 2))
                                     for x in cross_scores[s]['dists']]

    deformations = torch.nn.Parameter(torch.randn(
        (len(invivo_cent), 3)) / 10)  # N(0, 0.1)
    deformations_v_old = torch.nn.Parameter(torch.zeros((len(invivo_cent), 3)))
    nonrigid_optimizer = optim.Adam([{'params': deformations, 'lr': lr_deformations,
                                      'weight_decay': wd_deformations}])

    deformations_old = deformations.clone()
    if threshes == None:
        gx = tx
        gy = ty
        gz = tz * z_range_scale
    else:
        gx = threshes[0]
        gy = threshes[1]
        gz = threshes[2]
    loss_old = math.inf

    pot_match_tab = np.zeros([0, 5])
    matched_cent = torch.tensor(np.array(aff_invitro_cent))
    true_cent = torch.tensor(np.array(invivo_cent))
    manual = match_df
    for m in range(0, len(matched_cent)):
        print(str(m) + "/" + str(len(matched_cent)))
        for r in range(0, len(true_cent)):
            dx = true_cent[r, 0] - matched_cent[m, 0]
            dy = true_cent[r, 1] - matched_cent[m, 1]
            dz = true_cent[r, 2] - matched_cent[m, 2]
            if abs(dx) < gx and abs(dy) < gy and abs(dz) < gz:
                pot_match_tab = np.append(pot_match_tab, [[matched_cent[m, 3].item(), true_cent[r, 3].item(),
                                                           dx.item(), dy.item(), dz.item()]], axis=0)
    delta_cov = torch.cov(torch.tensor(
        pot_match_tab[:, 2:5].transpose().astype(float))) / 2
    delta_cov[2, 2] = delta_cov[2, 2] * z_range_scale

    for i in range(0, max_demon_iters):
        # Zero gradients
        nonrigid_optimizer.zero_grad()

        # Calculate warping field
        warping_field = torch.tensor(np.zeros([len(cross_scores), 3]))
        for j in range(0, len(cross_scores)):
            cent_id = str(aff_invitro_cent.iloc[j, 3])
            nonzero_elements = np.array(cross_scores[cent_id]['matchIDs'])
            for k in range(0, 3):
                warping_field[j, k] = torch.matmul(torch.tensor(cross_scores[cent_id]['scores']).unsqueeze(dim=0),
                                                   deformations[[np.where(np.isin(np.array(invivo_cent)[:, 3], z))[0]
                                                                 for z in nonzero_elements], k])

        ## Predicted
        pred_invitro = torch.tensor(
            np.array(aff_invitro_cent.iloc[:, 0:3])) + warping_field
        pred_invitro = torch.cat((pred_invitro, torch.tensor(
            np.array(aff_invitro_cent.iloc[:, 3])).unsqueeze(dim=1)), 1)

        ## GMM loss
        matched_cent = pred_invitro
        true_cent = torch.tensor(np.array(invivo_cent))
        manual = match_df[:, [0, 1]]

        for n in range(0, len(pot_match_tab)):
            m = np.where(matched_cent[:, 3] == pot_match_tab[n, 0])[0][0]
            r = np.where(true_cent[:, 3] == pot_match_tab[n, 1])[0][0]
            dx = true_cent[r, 0] - matched_cent[m, 0]
            dy = true_cent[r, 1] - matched_cent[m, 1]
            dz = true_cent[r, 2] - matched_cent[m, 2]
            if 'gmm_tensor' not in locals():
                gmm_tensor = torch.cat((matched_cent[m, 3:4], true_cent[r, 3:4], matched_cent[m, 0:3],
                                        true_cent[r, 0:3], dx.unsqueeze(0), dy.unsqueeze(0), dz.unsqueeze(0)), dim=0).unsqueeze(0)
            else:
                gmm_tensor = torch.cat((gmm_tensor, torch.cat((matched_cent[m, 3:4], true_cent[r, 3:4], matched_cent[m, 0:3],
                                        true_cent[r, 0:3], dx.unsqueeze(0), dy.unsqueeze(0), dz.unsqueeze(0)), dim=0).unsqueeze(0)), dim=0)
        gmm_tensor = torch.cat((gmm_tensor, torch.zeros(
            len(gmm_tensor), 1), torch.zeros(len(gmm_tensor), 1)), dim=1)

        normalpdf = MultivariateNormal(
            loc=torch.zeros(3), covariance_matrix=delta_cov)
        large_prob = torch.zeros(1)
        large_prob_norm = torch.exp(
            normalpdf.log_prob(torch.tensor([0, 0, 0])))
        for u in range(0, len(matched_cent)):
            scan_match_rows = torch.where(
                gmm_tensor[:, 0] == matched_cent[u, 3].item())
            if len(scan_match_rows[0]) == 0:
                continue
            elif len(scan_match_rows[0]) == 1:
                gmm_tensor[scan_match_rows[0][0], 12] = 1
                gmm_tensor[scan_match_rows[0][0], 11] = torch.exp(
                    normalpdf.log_prob(gmm_tensor[scan_match_rows[0][0], [8, 9, 10]]))
            else:
                temp_prob = torch.zeros(
                    len(scan_match_rows[0]), dtype=torch.float64)
                ## Use multivariate Gaussian for GMM calculation.
                for j in range(0, len(scan_match_rows[0])):
                    temp_prob[j] = torch.exp(normalpdf.log_prob(
                        gmm_tensor[scan_match_rows[0][j], [8, 9, 10]]))
                gmm_tensor[scan_match_rows[0], 11] = temp_prob
                gmm_tensor[scan_match_rows[0], 12] = temp_prob / sum(temp_prob)
                large_prob += torch.max(temp_prob)

        true_prob = torch.zeros([0])
        true_dens = torch.zeros([0])
        for u in range(0, len(true_cent)):
            scan_match_rows = torch.where(
                gmm_tensor[:, 1] == true_cent[u, 3].item())
            if len(scan_match_rows[0]) == 0:
                continue
            else:
                true_dens = torch.cat((true_dens, torch.max(
                    gmm_tensor[scan_match_rows[0], 11]).unsqueeze(0)))
                true_prob = torch.cat((true_prob, torch.max(
                    gmm_tensor[scan_match_rows[0], 12]).unsqueeze(0)))
                
        gmm_loss = (gmm_tensor[:, 12] * (1-(gmm_tensor[:, 12]))).sum()
        gmm_loss_var = torch.std(gmm_tensor[:, 12] * (1-(gmm_tensor[:, 12])))

        # Smoothing regularization
        norm_deformations = deformations / torch.norm(deformations + 10 ** -20, dim=-1,
                                                      keepdim=True)
        #Sum the deformations that are relevant.
        reg_term = 0
        for j in range(0, len(invivo_scores)):
            cent_id = str(np.array(invivo_cent)[j, 3])
            nonzero_elements = np.array(invivo_scores[cent_id]['matchIDs'])
            if len(nonzero_elements) == 0:
                continue
            elif len(nonzero_elements) == 1:
                reg_term += -(torch.tensor(invivo_scores[cent_id]['scores']) *
                              torch.mm(norm_deformations[[np.where(np.isin(np.array(invivo_cent)[:, 3], z))[0][0]
                                                          for z in nonzero_elements], :],
                                       norm_deformations[j, :].unsqueeze(dim=0).t()).squeeze())
            else:
                reg_term += -torch.matmul(torch.tensor(invivo_scores[cent_id]['scores']).unsqueeze(dim=0)[0],
                                          torch.mm(norm_deformations[[np.where(np.isin(np.array(invivo_cent)[:, 3], z))[0][0]
                                                                      for z in nonzero_elements], :],
                                          norm_deformations[j, :].unsqueeze(dim=0).t()))

        reg_term = torch.tensor(reg_term / (total_invivo_scores + 10 ** -9))
        dens_oto = (torch.sort(true_dens, descending=True)
                    [0][0:len(matched_cent)]).sum()
        prob_oto = (1 - torch.sort(true_prob, descending=True)
                    [0][0:len(matched_cent)]).sum()

        loss = (gmm_loss / len(matched_cent) +
                smoothness_factor * reg_term -
                dens_factor * large_prob / large_prob_norm / len(matched_cent) -
                td_factor * dens_oto / large_prob_norm / len(matched_cent) +
                true_factor * prob_oto / len(matched_cent))
        print('Loss at iteration {}: {:5.4f}'.format(i, loss.item()))

        # Update
        loss.backward()
        nonrigid_optimizer.step()

        # Save 2nd moment information.
        deformations_g2 = (nonrigid_optimizer.state_dict()['state'].get(0)['exp_avg_sq'] -
                           deformations_v_old * nonrigid_optimizer.state_dict()['param_groups'][0]['betas'][1]) / \
            (1 - nonrigid_optimizer.state_dict()
             ['param_groups'][0]['betas'][1])
        deformations_v_old = nonrigid_optimizer.state_dict()['state'].get(0)[
            'exp_avg_sq'].clone()
        # print(nonrigid_optimizer.state_dict())
        # print(sum(abs(deformations_old - deformations)))
        deformations_old = deformations.clone()

        if (loss_old - loss) / loss < 10 ** -9 and i > min_demon_iters:
            break
        else:
            loss_old = loss.clone()
            del gmm_tensor

    # Save final results
    nonrigid_deformations = deformations.detach().clone()
    conf_constant = gmm_loss_var * 4 * scipy.stats.norm.ppf(1 - conf_alpha / 2) / \
        scipy.stats.chi2.ppf((1 - conf_alpha) ** 2,
                             len(nonrigid_deformations) * 3)
    upperconf_nonrigid_deformations = conf_constant * \
        deformations_g2 ** -1 + nonrigid_deformations
    lowerconf_nonrigid_deformations = -conf_constant * \
        deformations_g2 ** -1 + nonrigid_deformations
    sd_nonrigid_deformations = gmm_loss_var * deformations_g2

    nonrigid_res_matrices = {"alpha": conf_alpha,
                             "nonrigid_deformations": nonrigid_deformations,
                             "sd_deformations": sd_nonrigid_deformations,
                             "upper_deformations": upperconf_nonrigid_deformations,
                             "lower_deformations": lowerconf_nonrigid_deformations}

    
    matchprob_df = pd.DataFrame(gmm_tensor.detach().numpy(),
                                columns=['SliceUnit', 'StackUnit',
                                         'slice_x', 'slice_y', 'slice_z',
                                         'stack_x', 'stack_y', 'stack_z',
                                         'dx', 'dy', 'dz', 'dens', 'prob'])

    ## Return: nonrigid transform parameters, estimated transform in the in vitro, GMM prob table.
    nic = pd.DataFrame(pred_invitro[:, 0:4].clone().detach().numpy(),
                       columns=['3D_X', '3D_Y', '3D_Z', 'ID'])

    return_dict = {'manual_matches': match_df,
                   'nonrigid_parameters': nonrigid_res_matrices,
                   'invivo_units': invivo_cent,
                   'invitro_units': pd.DataFrame(pred_invitro.detach().numpy(), columns=['X', 'Y', 'Z', 'ID']),
                   'matching_probs': matchprob_df,
                   'aff_invitro_cent': aff_invitro_cent,
                   'nonrigid_invitro_cent': nic}

    return(return_dict)

#######################
#######################
#######################

# Find one-to-one automatic matches
# nonrigid_torch: return dictionary from running nonrigid_demon function.
# struct_units: x-y-z-ID of in vivo (stack) centroids.
# slice_units: slicenum-x-y-ID of in vivo (stack) centroids.

def automatch(nonrigid_torch, struct_units, slice_units):
    predprob_df = pd.DataFrame(columns=['SliceNum', 'SliceUnit', 'StackUnit', 'PredProb', 
                                        'NRMatchDx', 'NRMatchDy', 'NRMatchDz'])
    oto_predprob_df = pd.DataFrame(columns=['SliceNum', 'SliceUnit', 'StackUnit', 'PredProb', 
                                        'NRMatchDx', 'NRMatchDy', 'NRMatchDz', 'MatchDist'])
    
        
    for kk in range(0, len(np.unique(nonrigid_torch['matching_probs']['SliceUnit']))):
        print(str(kk) + "/" + str(len(np.unique(nonrigid_torch['matching_probs']['SliceUnit']))))
        slicenum = slice_units[np.where(slice_units[:, 3] == nonrigid_torch['matching_probs']['SliceUnit'][kk])[0], 0][0]
        nonrigid_match_probs = nonrigid_torch['matching_probs'].iloc[np.where(nonrigid_torch['matching_probs']['SliceUnit'] == np.unique(nonrigid_torch['matching_probs']['SliceUnit'])[kk])]
        nr_match_choice = np.argmax(nonrigid_match_probs['prob'])
        predprob_df.loc[len(predprob_df.index)] = [slicenum, np.unique(nonrigid_torch['matching_probs']['SliceUnit'])[kk],
                                                   nonrigid_match_probs['StackUnit'].iloc[nr_match_choice],
                                               nonrigid_match_probs['prob'].iloc[nr_match_choice],
                                               nonrigid_match_probs['dx'].iloc[nr_match_choice],
                                               nonrigid_match_probs['dy'].iloc[nr_match_choice],
                                               nonrigid_match_probs['dz'].iloc[nr_match_choice]]    
    predprob_df['MatchDist'] = np.sqrt(predprob_df['NRMatchDx'] ** 2 + predprob_df['NRMatchDy'] ** 2 + predprob_df['NRMatchDz'] ** 2)
    
    for jj in range(0, len(np.unique(predprob_df['StackUnit']))):
        predprob_df_probs = predprob_df.iloc[np.where(predprob_df['StackUnit'] == np.unique(predprob_df['StackUnit'])[jj])]
        oto_match_choice = np.argmin(predprob_df_probs['MatchDist'])
        oto_predprob_df.loc[len(oto_predprob_df.index)] = predprob_df_probs.iloc[oto_match_choice]
        
    high_conf = (oto_predprob_df['MatchDist'] < 7.5) & (oto_predprob_df['PredProb'] > 0.5)
    mid_conf = (oto_predprob_df['MatchDist'] < 15) & (oto_predprob_df['MatchDist'] > 7.5) & (oto_predprob_df['PredProb'] > 0.5)
    low_conf = (oto_predprob_df['MatchDist'] < 15) & (oto_predprob_df['PredProb'] < 0.5)
    no_conf = (oto_predprob_df['MatchDist'] > 15) 
    
    conf_vec = np.zeros(len(oto_predprob_df)).astype(str)
    conf_vec[high_conf] = "high"
    conf_vec[mid_conf] = "medium"
    conf_vec[low_conf] = "low"
    conf_vec[no_conf] = "no"
    
    oto_predprob_df['ConfidenceLevel'] = conf_vec
    
    return(oto_predprob_df)

#######################
#######################
#######################