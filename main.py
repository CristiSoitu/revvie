
from revvie.revvie_classes import *
from revvie.revvie_helpers import *


## Program that implements revvie, a pipleine that matches in vivo and in vitro neurons.


dataset_path = '/Users/soitu/Desktop/datasets/BCM28382/revvie/'
config_file = dataset_path + 'revvie_config.toml'


def main():
    args = ConfigReader(config_file)

    vitro_images_list = ['blood_vessels.tif']
    vitro_images_list = [args.dataset_path + '/revvie/images/' + image for image in vitro_images_list]
    vivo_images_list = ['rotated_blood_vessels_stack.tif']
    vivo_images_list = [args.dataset_path + 'revvie/images/' + image for image in vivo_images_list]
    
    
    vitro_points_list = [args.dataset_path + '/revvie/centroids/' + 'geneseq_slices_centroids.txt']
    vivo_points_list = [args.dataset_path + '/revvie/centroids/' + 'padded_struct_centroids.txt']
    

    vitro_points, vivo_points = load_latest_state(args)
    vitro_cloud_points = CloudPoints(xyz=vitro_points[:, 0:3], ids=vitro_points[:, 3], colors=vitro_points[:, 4:7], edge_colors=vitro_points[:, 7:10],
                                    size=args.vitro_point_size, opacity=args.opacity, edge_width=args.vitro_edge_size, symbol=args.symbol, ndim=3, name='points',
                                    matched_color=args.matched_color, unmatched_color=args.unmatched_color, accepted_edge_color=args.accepted_edge_color, rejected_edge_color=args.rejected_edge_color)
    vivo_cloud_points = CloudPoints(xyz=vivo_points[:, 0:3], ids=vivo_points[:, 3], colors=vivo_points[:, 4:7], edge_colors=vivo_points[:, 7:10],
                                    size=args.vivo_point_size, opacity=args.opacity, edge_width=args.vivo_edge_size, symbol=args.symbol, ndim=3, name='points',
                                    matched_color=args.matched_color, unmatched_color=args.unmatched_color, accepted_edge_color=args.accepted_edge_color, rejected_edge_color=args.rejected_edge_color)            
    # create revvie viewer
    revViewer = RevViewer(vitro_images_list, vivo_images_list, vitro_cloud_points, vivo_cloud_points)

    revViewer.run()



main()
