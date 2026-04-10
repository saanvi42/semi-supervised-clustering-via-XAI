import sys

from SSE_hierarchical import PartitionTree_SSE


from queue import Queue

import scipy
from graph_construction import knn_affinity, knn_cosine_sim, generate_constraints_pairwise, generate_constraints_label, knn_k_estimating
from SSE_partitioning import FlatSSE
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from utils import PartitionTree
from SSE_hierarchical import cal_dendrogram_purity
import argparse
import os
import json
import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, choices=['SSE_partitioning_pairwise', 'SSE_partitioning_label',
                                                         'SSE_partitioning_bio_pairwise', 'SSE_partitioning_bio_label', 'SSE_hierarchical'])
parser.add_argument('--dataset', required=True)
parser.add_argument('--constraint_ratio', type=float, required=True)
parser.add_argument('--constraint_weight', default=2.0, type=float)
parser.add_argument('--sigmasq', default=100, type=float, help='square of Gaussian kernel band width, i.e., sigma^2')
parser.add_argument('--exp_repeats', default=10, type=int)
parser.add_argument('--knn_constant', default=20, type=float)

# for hierarchical clustering
parser.add_argument('--hie_knn_k', default=5)
parser.add_argument('--save_artifacts', action='store_true', help='Save explainability artifacts (A, A_dense, constraints, preds, tree, merge history) when set')
parser.add_argument('--save_dir', default=None, help='Directory to write artifacts to when --save_artifacts is set')
parser.add_argument('--result_path', default=None, help='Append run-level metrics to this file')
args = parser.parse_args()



def SSE_pairwise_clustering(path):
    data_f = scipy.io.loadmat(path)
    X = np.array(data_f['fea']).astype(float)
    X = MinMaxScaler().fit_transform(X)
    y = np.array(data_f['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    knn_k = knn_k_estimating(n_cluster, n_instance, args.knn_constant)
    A, A_dense = knn_affinity(X, args.sigmasq, knn_k)
    ARIs = []
    NMIs = []
    DPs = []
    for run_idx in range(args.exp_repeats):
        A_constraints = generate_constraints_pairwise(y, int(A.shape[0] * args.constraint_ratio), int(A.shape[0] * args.constraint_ratio), A_dense)
        flatSSE = FlatSSE(A, args.constraint_weight * A_constraints, len(np.unique(y)))
        y_pred_flatSSE = flatSSE.build_tree()
        ARI = adjusted_rand_score(y, y_pred_flatSSE)
        NMI = normalized_mutual_info_score(y, y_pred_flatSSE)
        # Compute DP using hierarchical tree from the same data
        partitiontree_SSE = PartitionTree_SSE(A, args.constraint_weight * A_constraints)
        root_id, hierarchical_tree_node = partitiontree_SSE.build_coding_tree(2, mode='v1')
        DP = cal_dendrogram_purity(root_id, hierarchical_tree_node, n_instance, y)
        ARIs.append(ARI)
        NMIs.append(NMI)
        DPs.append(DP)
        print(path, "DP:", DP, "ARI:", ARI, "NMI:", NMI)
        
        # Save explainability artifacts if requested
        if args.save_artifacts and args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            # Save NPZ with all data
            np.savez_compressed(
                os.path.join(args.save_dir, f"SSE_partitioning_pairwise_{args.dataset}_run{run_idx}.npz"),
                A=A, A_dense=A_dense, A_constraints=A_constraints, X=X,
                y_pred=y_pred_flatSSE, y=y
            )
            # Save merge history from FlatSSE
            try:
                with open(os.path.join(args.save_dir, f"SSE_partitioning_pairwise_{args.dataset}_run{run_idx}_merge_history.json"), 'w') as mhf:
                    json.dump(flatSSE.merge_history, mhf)
            except Exception:
                pass
            # Save communities structure
            try:
                communities_data = {}
                for comm_id, comm_info in flatSSE.communities.items():
                    communities_data[str(comm_id)] = {
                        'members': list(comm_info[0]),
                        'volume': float(comm_info[1]),
                        'cut': float(comm_info[2]),
                        'cut_con': float(comm_info[3]),
                        'SSE': float(comm_info[4])
                    }
                with open(os.path.join(args.save_dir, f"SSE_partitioning_pairwise_{args.dataset}_run{run_idx}_communities.json"), 'w') as cf:
                    json.dump(communities_data, cf)
            except Exception:
                pass
            # Save tree structure from hierarchical representation
            try:
                serial_tree = partitiontree_SSE.serialize_tree(hierarchical_tree_node)
                with open(os.path.join(args.save_dir, f"SSE_partitioning_pairwise_{args.dataset}_run{run_idx}_tree.json"), 'w') as jf:
                    json.dump(serial_tree, jf)
            except Exception:
                pass
    print("average: {}\tDP: {:.4f}\tARI: {:.4f}\tNMI: {:.4f}\n".format(args.dataset, np.mean(DPs), np.mean(ARIs), np.mean(NMIs)))
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    x = range(1, len(ARIs) + 1)
    plt.plot(x, DPs, 'o-', label=f'DP (avg: {np.mean(DPs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, ARIs, 's-', label=f'ARI (avg: {np.mean(ARIs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, NMIs, '^-', label=f'NMI (avg: {np.mean(NMIs):.4f})', linewidth=2, markersize=8)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'SSE Partitioning Pairwise - {args.dataset}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'SSE_partitioning_pairwise_{args.dataset}_metrics.png', dpi=300)
    print(f"Plot saved as: SSE_partitioning_pairwise_{args.dataset}_metrics.png")
    plt.show()

def SSE_label_clustering(path):
    data_f = scipy.io.loadmat(path)
    X = np.array(data_f['fea']).astype(float)
    X = MinMaxScaler().fit_transform(X)
    y = np.array(data_f['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    knn_k = knn_k_estimating(n_cluster, n_instance, args.knn_constant)
    A, A_dense = knn_affinity(X, args.sigmasq, knn_k)
    ARIs = []
    NMIs = []
    DPs = []
    for run_idx in range(args.exp_repeats):
        A_constraints = generate_constraints_label(y, int(A.shape[0] * args.constraint_ratio), int(A.shape[0] * args.constraint_ratio), A_dense)
        flatSSE = FlatSSE(A, args.constraint_weight * A_constraints, len(np.unique(y)))
        y_pred_flatSSE = flatSSE.build_tree()
        ARI = adjusted_rand_score(y, y_pred_flatSSE)
        NMI = normalized_mutual_info_score(y, y_pred_flatSSE)
        # Compute DP using hierarchical tree from the same data
        partitiontree_SSE = PartitionTree_SSE(A, args.constraint_weight * A_constraints)
        root_id, hierarchical_tree_node = partitiontree_SSE.build_coding_tree(2, mode='v1')
        DP = cal_dendrogram_purity(root_id, hierarchical_tree_node, n_instance, y)
        ARIs.append(ARI)
        NMIs.append(NMI)
        DPs.append(DP)
        print(path, "DP:", DP, "ARI:", ARI, "NMI:", NMI)
        
        # Save explainability artifacts if requested
        if args.save_artifacts and args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            # Save NPZ with all data
            np.savez_compressed(
                os.path.join(args.save_dir, f"SSE_partitioning_label_{args.dataset}_run{run_idx}.npz"),
                A=A, A_dense=A_dense, A_constraints=A_constraints, X=X,
                y_pred=y_pred_flatSSE, y=y
            )
            # Save merge history from FlatSSE
            try:
                with open(os.path.join(args.save_dir, f"SSE_partitioning_label_{args.dataset}_run{run_idx}_merge_history.json"), 'w') as mhf:
                    json.dump(flatSSE.merge_history, mhf)
            except Exception:
                pass
            # Save communities structure
            try:
                communities_data = {}
                for comm_id, comm_info in flatSSE.communities.items():
                    communities_data[str(comm_id)] = {
                        'members': list(comm_info[0]),
                        'volume': float(comm_info[1]),
                        'cut': float(comm_info[2]),
                        'cut_con': float(comm_info[3]),
                        'SSE': float(comm_info[4])
                    }
                with open(os.path.join(args.save_dir, f"SSE_partitioning_label_{args.dataset}_run{run_idx}_communities.json"), 'w') as cf:
                    json.dump(communities_data, cf)
            except Exception:
                pass
            # Save tree structure from hierarchical representation
            try:
                serial_tree = partitiontree_SSE.serialize_tree(hierarchical_tree_node)
                with open(os.path.join(args.save_dir, f"SSE_partitioning_label_{args.dataset}_run{run_idx}_tree.json"), 'w') as jf:
                    json.dump(serial_tree, jf)
            except Exception:
                pass
    print("average: {}\tDP: {:.4f}\tARI: {:.4f}\tNMI: {:.4f}\n".format(args.dataset, np.mean(DPs), np.mean(ARIs), np.mean(NMIs)))
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    x = range(1, len(ARIs) + 1)
    plt.plot(x, DPs, 'o-', label=f'DP (avg: {np.mean(DPs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, ARIs, 's-', label=f'ARI (avg: {np.mean(ARIs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, NMIs, '^-', label=f'NMI (avg: {np.mean(NMIs):.4f})', linewidth=2, markersize=8)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'SSE Partitioning Label - {args.dataset}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'SSE_partitioning_label_{args.dataset}_metrics.png', dpi=300)
    print(f"Plot saved as: SSE_partitioning_label_{args.dataset}_metrics.png")
    plt.show()

def get_partition_from_2d(partition_tree, n_instances):
    root_id = partition_tree.root_id
    y_pred = np.zeros(n_instances)
    children = partition_tree.tree_node[root_id].children
    for index, child in enumerate(children):
        child = partition_tree.tree_node[child]
        partition = child.partition
        for vertex in partition:
            y_pred[vertex] = index
    return y_pred

def SSE_hierar_clustering(path, result_path=None):
    data_f = scipy.io.loadmat(path)
    X = np.array(data_f['fea']).astype(float)
    X = StandardScaler().fit_transform(X)
    y = np.array(data_f['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    A, A_dense = knn_cosine_sim(X, args.hie_knn_k)
    DPs = []
    ARIs = []
    NMIs = []
    for _ in range(args.exp_repeats):
        A_constraints = generate_constraints_pairwise(y, int(A.shape[0] * args.constraint_ratio), int(A.shape[0] * args.constraint_ratio), A_dense)
        partitiontree_SSE = PartitionTree_SSE(A, args.constraint_weight*A_constraints)
        root_id, hierarchical_tree_node = partitiontree_SSE.build_coding_tree(2, mode='v1')
        y_pred = get_partition_from_2d(partitiontree_SSE, n_instance)
        DP = cal_dendrogram_purity(root_id, hierarchical_tree_node, n_instance, y)
        ARI = adjusted_rand_score(y, y_pred)
        NMI = normalized_mutual_info_score(y, y_pred)
        DPs.append(DP)
        ARIs.append(ARI)
        NMIs.append(NMI)
        print(path, DP, ARI, NMI)
        # optional artifact saving for explainability
        if args.save_artifacts and args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            run_idx = len(DPs)-1
            np.savez_compressed(os.path.join(args.save_dir, f"{args.method}_{args.dataset}_run{run_idx}.npz"), A=A, A_dense=A_dense, A_constraints=A_constraints, X=X, y_pred=y_pred, y=y)
            # save serialized tree and merge history
            try:
                serial_tree = partitiontree_SSE.serialize_tree(hierarchical_tree_node)
                with open(os.path.join(args.save_dir, f"{args.method}_{args.dataset}_run{run_idx}_tree.json"), 'w') as jf:
                    json.dump(serial_tree, jf)
            except Exception:
                pass
            try:
                with open(os.path.join(args.save_dir, f"{args.method}_{args.dataset}_run{run_idx}_merge_history.json"), 'w') as mhf:
                    json.dump(partitiontree_SSE.merge_history, mhf)
            except Exception:
                pass
    print("average: {}\tDP: {:.4f}\tARI: {:.4f}\tNMI: {:.4f}\n".format(args.dataset, np.mean(DPs), np.mean(ARIs), np.mean(NMIs)))
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    x = range(1, len(DPs) + 1)
    plt.plot(x, DPs, 'o-', label=f'DP (avg: {np.mean(DPs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, ARIs, 's-', label=f'ARI (avg: {np.mean(ARIs):.4f})', linewidth=2, markersize=8)
    plt.plot(x, NMIs, '^-', label=f'NMI (avg: {np.mean(NMIs):.4f})', linewidth=2, markersize=8)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'SSE Hierarchical - {args.dataset}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'SSE_hierarchical_{args.dataset}_metrics.png', dpi=300)
    print(f"Plot saved as: SSE_hierarchical_{args.dataset}_metrics.png")
    plt.show()
    
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write("DP:\t")
            for DP in DPs:
                f.write("{}\t".format(DP))
            f.write("average:\t{}\nARI:\t".format(np.mean(DPs)))
            for ARI in ARIs:
                f.write("{}\t".format(ARI))
            f.write("average:\t{}\nNMI:\t".format(np.mean(ARIs)))
            for NMI in NMIs:
                f.write("{}\t".format(NMI))
            f.write("average:\t{}\n".format(np.mean(NMIs)))

def SSE_pairwise_clustering_bio(path):
    data_f = scipy.io.loadmat(path)
    X = np.array(data_f['fea']).astype(float)
    X = MinMaxScaler().fit_transform(X)
    y = np.array(data_f['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    knn_k = knn_k_estimating(n_cluster, n_instance, args.knn_constant)
    A, A_dense = knn_cosine_sim(X, knn_k)
    ARIs = []
    NMIs = []
    for _ in range(args.exp_repeats):
        A_constraints = generate_constraints_pairwise(y, int(A.shape[0] * args.constraint_ratio),
                                                      int(A.shape[0] * args.constraint_ratio), A_dense)
        flatSSE = FlatSSE(A, args.constraint_weight * A_constraints, len(np.unique(y)), mustlink_first=True)
        y_pred_flatSSE = flatSSE.build_tree()
        ARI = adjusted_rand_score(y, y_pred_flatSSE)
        NMI = normalized_mutual_info_score(y, y_pred_flatSSE)
        ARIs.append(ARI)
        NMIs.append(NMI)
        print(path, ARI, NMI)
    print(np.mean(ARIs), np.mean(NMIs))

def SSE_label_clustering_bio(path):
    data_f = scipy.io.loadmat(path)
    X = np.array(data_f['fea']).astype(float)
    X = MinMaxScaler().fit_transform(X)
    y = np.array(data_f['gnd']).astype(float).squeeze()
    n_instance = y.shape[0]
    n_cluster = np.unique(y).shape[0]
    knn_k = knn_k_estimating(n_cluster, n_instance, args.knn_constant)
    A, A_dense = knn_cosine_sim(X, knn_k)
    ARIs = []
    NMIs = []
    for _ in range(args.exp_repeats):
        A_constraints = generate_constraints_label(y, int(A.shape[0] * args.constraint_ratio),
                                                      int(A.shape[0] * args.constraint_ratio), A_dense)
        flatSSE = FlatSSE(A, args.constraint_weight * A_constraints, len(np.unique(y)), mustlink_first=True)
        y_pred_flatSSE = flatSSE.build_tree()
        ARI = adjusted_rand_score(y, y_pred_flatSSE)
        NMI = normalized_mutual_info_score(y, y_pred_flatSSE)
        ARIs.append(ARI)
        NMIs.append(NMI)
        print(path, ARI, NMI)
    print(np.mean(ARIs), np.mean(NMIs))

if __name__=='__main__':
    if args.method == "SSE_partitioning_pairwise":
        path = "./datasets/clustering/{}.mat".format(args.dataset)
        SSE_pairwise_clustering(path)
    elif args.method == "SSE_partitioning_label":
        path = "./datasets/clustering/{}.mat".format(args.dataset)
        SSE_label_clustering(path)
    elif args.method == "SSE_partitioning_bio_pairwise":
        path = "./datasets/RNA-seq/{}.mat".format(args.dataset)
        SSE_pairwise_clustering_bio(path)
    elif args.method == "SSE_partitioning_bio_label":
        path = "./datasets/RNA-seq/{}.mat".format(args.dataset)
        SSE_label_clustering_bio(path)
    elif args.method == "SSE_hierarchical":
        path = "./datasets/hierarchical/{}.mat".format(args.dataset)
        SSE_hierar_clustering(path)
