from phy import IPlugin, connect
import logging
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
import umap

logger = logging.getLogger('phy')


class ReclusterUMAP(IPlugin):
    """
    Modern spike sorting plugin with optimized performance and intelligent merging
    """

    def __init__(self):
        super(ReclusterUMAP, self).__init__()
        self._shortcuts_created = False
        self._umap_reducer = None
        self._last_n_spikes = None

    def attach_to_controller(self, controller):
        def prepareFeatures(spikeIds):
            data = controller.model._load_features().data[spikeIds]
            features = np.reshape(data, (data.shape[0], -1))
            return features

        def compute_template_correlation(features, labels):
            """Compute correlation between cluster templates"""
            unique_labels = np.unique(labels[labels > 0])
            n_clusters = len(unique_labels)
            correlations = np.zeros((n_clusters, n_clusters))
            
            # Cache templates for efficiency
            templates = {}
            for i, label in enumerate(unique_labels):
                templates[i] = np.mean(features[labels == label], axis=0)

            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Use cached templates
                    template_i = templates[i]
                    template_j = templates[j]
                    # Correlation between templates
                    corr = np.corrcoef(template_i, template_j)[0, 1]
                    correlations[i, j] = correlations[j, i] = corr

            return correlations, unique_labels

        def check_spatial_consistency(original_features, labels, threshold=0.6):
            """Check if clusters are spatially consistent using original features"""
            unique_labels = np.unique(labels[labels > 0])
            n_clusters = len(unique_labels)
            spatial_consistent = np.zeros((n_clusters, n_clusters), dtype=bool)

            # Cache channel variance for each cluster
            cluster_channels = {}
            for i, label in enumerate(unique_labels):
                spikes_i = original_features[labels == label]
                cluster_channels[i] = np.var(spikes_i, axis=0).argsort()[-4:]  # Top 4 channels

            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Check channel overlap
                    common_channels = len(set(cluster_channels[i]) & set(cluster_channels[j]))
                    threshold_channels = len(cluster_channels[i]) * threshold
                    spatial_consistent[i, j] = spatial_consistent[j, i] = (
                            common_channels >= threshold_channels
                    )

            return spatial_consistent

        def merge_similar_clusters(features, original_features, labels, template_threshold=0.9, spatial_threshold=0.6):
            """Merge clusters based on template similarity and spatial consistency"""
            while True:
                unique_labels = np.unique(labels[labels > 0])
                n_clusters = len(unique_labels)
                if n_clusters <= 2:  # Don't merge if only 2 clusters remain
                    break

                # Compute similarity matrices
                correlations, label_indices = compute_template_correlation(features, labels)
                spatial_consistent = check_spatial_consistency(original_features, labels, spatial_threshold)

                # Find most similar pair that's spatially consistent
                max_corr = template_threshold
                merge_pair = None

                for i in range(n_clusters):
                    for j in range(i + 1, n_clusters):
                        if (correlations[i, j] > max_corr and spatial_consistent[i, j]):
                            max_corr = correlations[i, j]
                            merge_pair = (i, j)

                if merge_pair is None:
                    break

                # Perform merge - create a label mapping for continuous labeling
                i, j = merge_pair
                label_i = label_indices[i]
                label_j = label_indices[j]
                
                # Create new labels by merging j into i
                new_labels = labels.copy()
                new_labels[labels == label_j] = label_i
                
                # Relabel to ensure continuous numbering
                unique_new_labels = np.unique(new_labels[new_labels > 0])
                label_map = {old: new+1 for new, old in enumerate(unique_new_labels)}
                
                result_labels = np.zeros_like(labels)
                for old_label, new_label in label_map.items():
                    result_labels[new_labels == old_label] = new_label
                
                labels = result_labels
                logger.info(f"Merged clusters (correlation: {max_corr:.3f})")

            return labels

        def fastClustering(embedding, original_features, target_clusters=4):
            """Fast clustering with intelligent merging"""
            # Initial over-clustering
            initial_clusters = min(target_clusters * 3, len(embedding) // 50)
            initial_clusters = max(initial_clusters, target_clusters)  # Ensure we have at least the target number

            # Use GMM for clustering to match function name
            gmm = GaussianMixture(
                n_components=initial_clusters,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            initial_labels = gmm.fit_predict(embedding) + 1  # Make labels 1-based

            # Merge similar clusters
            final_labels = merge_similar_clusters(
                embedding,
                original_features,  # Pass original features for spatial consistency check
                initial_labels,
                template_threshold=0.9,
                spatial_threshold=0.6
            )

            return final_labels

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+k', prompt=True, prompt_default=lambda: 4)
            def umapGmmClustering(target_clusters):
                """Fast UMAP-GMM Clustering with intelligent merging (Alt+K)"""
                try:
                    target_clusters = int(target_clusters)
                    if target_clusters < 2:
                        logger.warn("Need at least 2 clusters, using 2")
                        target_clusters = 2

                    clusterIds = controller.supervisor.selected
                    if not clusterIds:
                        logger.warn("No clusters selected!")
                        return

                    bunchs = controller._amplitude_getter(clusterIds, name='template', load_all=True)
                    spikeIds = bunchs[0].spike_ids
                    n_spikes = len(spikeIds)
                    
                    if n_spikes < target_clusters * 5:
                        logger.warn(f"Too few spikes ({n_spikes}) for {target_clusters} clusters")
                        return
                        
                    logger.info(f"Processing {n_spikes} spikes with target {target_clusters} clusters")

                    # Feature preparation
                    features = prepareFeatures(spikeIds)
                    original_features = features.copy()  # Keep original features for spatial consistency
                    
                    scaler = StandardScaler()
                    featuresScaled = scaler.fit_transform(features)

                    # Dimensionality reduction
                    pca = PCA(n_components=min(30, featuresScaled.shape[1]))
                    featuresPca = pca.fit_transform(featuresScaled)

                    # UMAP reduction with more robust n_neighbors setting
                    if (self._umap_reducer is None or
                            self._last_n_spikes is None or
                            abs(self._last_n_spikes - n_spikes) > n_spikes * 0.2):
                        n_neighbors = max(15, min(50, n_spikes // 100))
                        self._umap_reducer = umap.UMAP(
                            n_neighbors=n_neighbors,
                            min_dist=0.2,
                            n_components=2,
                            random_state=42,
                            n_jobs=-1,
                            metric='euclidean',
                            low_memory=True
                        )
                        self._last_n_spikes = n_spikes

                    embedding = self._umap_reducer.fit_transform(featuresPca)

                    # Clustering with merging, passing original_features for spatial consistency
                    labels = fastClustering(embedding, original_features, target_clusters)
                    n_clusters = len(np.unique(labels))

                    logger.info(f"Created {n_clusters} clusters after merging")
                    controller.supervisor.actions.split(spikeIds, labels)

                except Exception as e:
                    logger.error(f"Error in umapGmmClustering: {str(e)}")

            # Keep the existing templateBasedSplit function unchanged
            @controller.supervisor.actions.add(shortcut='alt+t', prompt=True, prompt_default=lambda: 0.85)
            def templateBasedSplit(similarityThreshold):
                """Template-based spike sorting (Alt+T)"""
                # ... [rest of the existing templateBasedSplit code remains unchanged]