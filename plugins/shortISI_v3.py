from phy import IPlugin, connect
import numpy as np
import logging
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Optional, Any, Union

logger = logging.getLogger('phy')


class ImprovedISIAnalysis(IPlugin):
    """
    More reliable spike analysis using combined metrics.
    
    This plugin analyzes spike patterns to identify suspicious spikes using:
    - Inter-spike interval (ISI) violations 
    - Amplitude anomalies
    - Waveform distortions
    """

    def __init__(self):
        super(ImprovedISIAnalysis, self).__init__()
        self._shortcuts_created = False
        # Add configurable parameters
        self.config = {
            'isi_threshold': 0.0015,      # seconds
            'amp_variation_threshold': 1.5,  
            'wave_variation_threshold': 0.1,
            'min_suspicious_count': 10,
            'max_suspicious_fraction': 0.5,
        }

    def attach_to_controller(self, controller):
        def get_waveform_features(spike_ids: np.ndarray) -> np.ndarray:
            """
            Extract key waveform features from spike data.
            
            Parameters:
            -----------
            spike_ids : np.ndarray
                IDs of spikes to extract features for
                
            Returns:
            --------
            np.ndarray
                Reshaped waveform features for analysis
            """
            # Get waveforms
            data = controller.model._load_features().data[spike_ids]
            return np.reshape(data, (data.shape[0], -1))

        def analyze_suspicious_spikes(
            spike_times: np.ndarray, 
            spike_amps: np.ndarray, 
            waveforms: np.ndarray, 
            isi_threshold: float = None
        ) -> np.ndarray:
            """
            Analyze spikes to identify suspicious ones using multiple criteria.
            
            Parameters:
            -----------
            spike_times : np.ndarray
                Array of spike timestamps in seconds
            spike_amps : np.ndarray
                Array of spike amplitudes
            waveforms : np.ndarray
                Array of waveform features for each spike
            isi_threshold : float, optional
                Minimum time difference between spikes to trigger analysis
                
            Returns:
            --------
            np.ndarray
                Boolean mask identifying suspicious spikes
            """
            if isi_threshold is None:
                isi_threshold = self.config['isi_threshold']
                
            n_spikes = len(spike_times)
            suspicious = np.zeros(n_spikes, dtype=bool)
            
            if n_spikes < 3:  # Need at least 3 spikes for meaningful analysis
                logger.warning("Too few spikes for reliable analysis")
                return suspicious

            # Find ISI violations more efficiently
            isi_prev = np.diff(spike_times, prepend=spike_times[0] - 1)
            isi_next = np.diff(spike_times, append=spike_times[-1] + 1)
            
            # Identify spikes with short ISIs
            short_isi_mask = (isi_prev < isi_threshold) | (isi_next < isi_threshold)
            candidate_indices = np.where(short_isi_mask)[0]
            
            # Skip if no candidates
            if len(candidate_indices) == 0:
                return suspicious
            
            # Global amplitude variation baseline
            amp_std_global = np.std(spike_amps)
            if amp_std_global == 0:  # Avoid division by zero
                amp_std_global = 1e-10
                
            # Analyze each candidate spike
            for i in candidate_indices:
                # Window indices with boundary checking
                window_start = max(0, i - 1)
                window_end = min(n_spikes, i + 2)
                
                # 1. Amplitude check
                amp_window = spike_amps[window_start:window_end]
                amp_variation = np.std(amp_window)
                
                # 2. Waveform check
                waves = waveforms[window_start:window_end]
                wave_distances = cdist(waves, waves, metric='correlation')
                wave_variation = np.mean(wave_distances)
                
                # Mark suspicious if criteria met
                if (amp_variation > amp_std_global * self.config['amp_variation_threshold'] or
                        wave_variation > self.config['wave_variation_threshold']):
                    suspicious[i] = True
            
            return suspicious

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+i')
            def analyze_spike_patterns():
                """
                Analyze spike patterns using multiple metrics:
                - ISI violations
                - Amplitude changes
                - Waveform changes
                Only splits when multiple criteria suggest different units.
                """
                try:
                    # Get selected clusters
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids:
                        logger.warning("No clusters selected!")
                        return

                    for cluster_id in cluster_ids:
                        logger.info(f"Analyzing cluster {cluster_id}...")
                        
                        # Get spike data
                        try:
                            bunchs = controller._amplitude_getter([cluster_id], name='template', load_all=True)
                            spike_ids = bunchs[0].spike_ids
                            spike_times = controller.model.spike_times[spike_ids]
                            spike_amps = bunchs[0].amplitudes
                        except IndexError:
                            logger.error(f"Failed to get spike data for cluster {cluster_id}")
                            continue
                            
                        if len(spike_times) < 10:
                            logger.info(f"Cluster {cluster_id} has too few spikes ({len(spike_times)}) for analysis")
                            continue

                        # Get waveform features
                        try:
                            waveforms = get_waveform_features(spike_ids)
                        except Exception as e:
                            logger.error(f"Failed to extract waveform features: {str(e)}")
                            continue

                        # Analyze spikes
                        suspicious = analyze_suspicious_spikes(
                            spike_times,
                            spike_amps,
                            waveforms,
                            isi_threshold=self.config['isi_threshold']
                        )

                        # Prepare labels
                        labels = np.ones(len(spike_ids), dtype=int)
                        labels[suspicious] = 2

                        # Count suspicious spikes
                        n_suspicious = np.sum(suspicious)
                        min_required = self.config['min_suspicious_count']
                        max_fraction = self.config['max_suspicious_fraction']

                        if n_suspicious > 0:
                            # Log analysis results
                            logger.info(f"Found {n_suspicious} suspicious spikes "
                                      f"({n_suspicious / len(spike_ids) * 100:.1f}%) "
                                      f"with notable physical changes")

                            # Only split if we found enough suspicious spikes
                            if n_suspicious >= min_required and n_suspicious <= len(spike_ids) * max_fraction:
                                controller.supervisor.actions.split(spike_ids, labels)
                                logger.info("Split suspicious spikes for manual review")
                            else:
                                logger.info("Too few or too many suspicious spikes for reliable splitting")
                        else:
                            logger.info("No suspicious spikes found")

                except ValueError as ve:
                    logger.error(f"Value error in analyze_spike_patterns: {str(ve)}")
                    logger.debug("Stack trace:", exc_info=True)
                except IndexError as ie:
                    logger.error(f"Index error in analyze_spike_patterns: {str(ie)}")
                    logger.debug("Stack trace:", exc_info=True)
                except Exception as e:
                    logger.error(f"Unexpected error in analyze_spike_patterns: {str(e)}")
                    logger.debug("Stack trace:", exc_info=True)