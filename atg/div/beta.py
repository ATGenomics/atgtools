#!/usr/bin/env python

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List


class BetaDiversity:
    """
    Measures compositional differences between samples using various distance metrics.
    Each metric captures different aspects of community dissimilarity:
    - Euclidean: Overall abundance differences
    - Bray-Curtis: Relative abundance differences
    - Jensen-Shannon: Information-theoretic divergence
    - Jaccard: Presence/absence differences
    - UniFrac: Phylogenetic differences
    """

    def __init__(self, species_profile_df: pd.DataFrame, species_identifier_col: str = "taxonomy"):
        """
        Initializes beta diversity calculator with species abundance data.
        
        Parameters
        ----------
        species_profile_df : pd.DataFrame
            Species abundance matrix with samples as columns
        species_identifier_col : str, optional
            Column name containing species identifiers
        """
        if not isinstance(species_profile_df, pd.DataFrame):
            raise ValueError("species_profile_df must be a pandas DataFrame")
            
        if species_identifier_col not in species_profile_df.columns:
            raise ValueError(f"Species identifier column '{species_identifier_col}' not found")
            
        metadata_cols = {species_identifier_col}
        sample_cols = []
        for col in species_profile_df.columns:
            if col not in metadata_cols:
                try:
                    pd.to_numeric(species_profile_df[col])
                    sample_cols.append(col)
                except (ValueError, TypeError):
                    continue
        if not sample_cols:
            raise ValueError("must be numeric")

        if not species_profile_df[species_identifier_col].is_unique:
            species_data = species_profile_df.groupby(species_identifier_col)[sample_cols].sum()
        else:
            species_data = species_profile_df.set_index(species_identifier_col)[sample_cols]
            
        self.processed_df = species_data.T
        self.sample_names = self.processed_df.index.tolist()
        self.species_names = species_data.index.tolist()
        
        self._validate_processed_data()
        self.normalized_abundance_matrix = self._normalize_data()
        
        self._metric_methods_map = {
            "euclidean": self.euclidean,
            "braycurtis": self.braycurtis,
            "jensenshannon": self.jensenshannon,
            "jaccard": self.jaccard,
            "unifrac": self.unifrac,
            "unweighted_unifrac": self.unweighted_unifrac
        }

    def _validate_processed_data(self) -> None:
        """Ensures data integrity for diversity calculations."""
        for col_name in self.processed_df.columns:
            if not pd.api.types.is_numeric_dtype(self.processed_df[col_name]):
                raise ValueError(f"Column '{col_name}' in processed data must be numeric")
        if (self.processed_df < 0).any().any():
            raise ValueError("Processed DataFrame contains negative values")

    def _normalize_data(self) -> np.ndarray:
        """Converts abundances to relative proportions for meaningful comparisons."""
        data_array = np.array(self.processed_df, dtype=float)
        row_sums = data_array.sum(axis=1)
        
        normalized_matrix = np.zeros_like(data_array)
        valid_rows_mask = row_sums > 0
        if np.any(valid_rows_mask):
            normalized_matrix[valid_rows_mask] = \
                data_array[valid_rows_mask] / row_sums[valid_rows_mask][:, np.newaxis]
        return normalized_matrix

    @staticmethod
    def _validate_vectors(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validates and cleans input vectors for distance calculations.
        
        Parameters
        ----------
        x : np.ndarray
            First abundance vector
        y : np.ndarray
            Second abundance vector
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cleaned vectors with NaN values removed
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.shape != y.shape:
            raise ValueError("Input vectors must have the same shape")
            
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        return x[valid_mask], y[valid_mask]

    @staticmethod
    def euclidean(x: np.ndarray, y: np.ndarray) -> float:
        """Measures absolute differences in species abundances."""
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        return np.sqrt(np.sum((valid_x - valid_y) ** 2))

    @staticmethod
    def braycurtis(x: np.ndarray, y: np.ndarray) -> float:
        """Measures relative differences in species abundances."""
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        numerator = np.sum(np.abs(valid_x - valid_y))
        denominator = np.sum(valid_x + valid_y)
        return numerator / denominator if denominator > 0 else 0

    @staticmethod
    def jensenshannon(x: np.ndarray, y: np.ndarray) -> float:
        """Measures information-theoretic divergence between communities.
        
        The Jensen-Shannon divergence is a symmetric and bounded measure of the 
        similarity between two probability distributions. It is based on the 
        Kullback-Leibler divergence but is symmetric and bounded between 0 and 1.
        
        Edge cases:
        - If either distribution has zero sum, return 1.0 (maximum divergence)
        - If distributions are identical, return 0.0 (minimum divergence)
        - Handle zero probabilities in KL divergence calculation
        """
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        
        sum_x = np.sum(valid_x)
        sum_y = np.sum(valid_y)
        
        # Handle zero sum cases
        if sum_x == 0 or sum_y == 0:
            return 1.0
        
        # Normalize to probability distributions
        x_norm = valid_x / sum_x
        y_norm = valid_y / sum_y
        
        # Check if distributions are identical
        if np.allclose(x_norm, y_norm):
            return 0.0
        
        # Calculate mean distribution
        m = 0.5 * (x_norm + y_norm)
        
        # Calculate KL divergences safely
        kl_xm = 0.0
        kl_ym = 0.0
        
        # Only consider non-zero probabilities
        x_mask = x_norm > 0
        y_mask = y_norm > 0
        
        if np.any(x_mask):
            kl_xm = np.sum(x_norm[x_mask] * np.log2(x_norm[x_mask] / m[x_mask]))
        if np.any(y_mask):
            kl_ym = np.sum(y_norm[y_mask] * np.log2(y_norm[y_mask] / m[y_mask]))
        
        return 0.5 * (kl_xm + kl_ym)

    @staticmethod
    def jaccard(x: np.ndarray, y: np.ndarray) -> float:
        """Measures presence/absence differences between communities."""
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        
        x_pres = valid_x > 0
        y_pres = valid_y > 0
        
        intersection = np.sum(x_pres & y_pres)
        union = np.sum(x_pres | y_pres)
        
        return 1 - (intersection / union) if union > 0 else 1

    @staticmethod
    def unifrac(x: np.ndarray, y: np.ndarray, tree: Optional[np.ndarray] = None) -> float:
        """Measures phylogenetic differences weighted by abundances.
        
        Follows scikit-bio's implementation of weighted UniFrac:
        1. Normalizes abundances to relative proportions
        2. For each branch in the tree:
           - Calculates the proportion of descendants in each sample
           - Weights by branch length
           - Takes absolute difference between samples
        3. Sums weighted differences and normalizes by total branch length
        """
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        
        sum_x = np.sum(valid_x)
        sum_y = np.sum(valid_y)
        
        if sum_x == 0 or sum_y == 0:
            return 1.0
            
        x_norm = valid_x / sum_x
        y_norm = valid_y / sum_y
        
        if tree is None:
            return np.sum(np.abs(x_norm - y_norm)) / 2
            
        # Get branch lengths and descendant counts
        branch_lengths = tree['length']
        descendants = tree['descendants']
        
        # Calculate weighted differences for each branch
        weighted_diffs = []
        total_length = 0
        
        for i, length in enumerate(branch_lengths):
            if length > 0:  # Skip zero-length branches
                # Get proportions of descendants in each sample
                x_prop = np.sum(x_norm[descendants[i]])
                y_prop = np.sum(y_norm[descendants[i]])
                
                # Weight by branch length
                weighted_diffs.append(length * abs(x_prop - y_prop))
                total_length += length
        
        if total_length == 0:
            return 1.0
            
        return np.sum(weighted_diffs) / total_length

    @staticmethod
    def unweighted_unifrac(x: np.ndarray, y: np.ndarray, tree: Optional[np.ndarray] = None) -> float:
        """Measures phylogenetic differences ignoring abundances.
        
        Follows scikit-bio's implementation of unweighted UniFrac:
        1. Converts abundances to presence/absence
        2. For each branch in the tree:
           - Checks if branch has descendants in each sample
           - Weights by branch length
           - Takes absolute difference between samples
        3. Sums weighted differences and normalizes by total branch length
        """
        valid_x, valid_y = BetaDiversity._validate_vectors(x, y)
        
        x_pres = valid_x > 0
        y_pres = valid_y > 0
        
        if tree is None:
            return BetaDiversity.jaccard(valid_x, valid_y)
            
        # Get branch lengths and descendant counts
        branch_lengths = tree['length']
        descendants = tree['descendants']
        
        # Calculate weighted differences for each branch
        weighted_diffs = []
        total_length = 0
        
        for i, length in enumerate(branch_lengths):
            if length > 0:  # Skip zero-length branches
                # Check if branch has descendants in each sample
                x_has_desc = np.any(x_pres[descendants[i]])
                y_has_desc = np.any(y_pres[descendants[i]])
                
                # Weight by branch length
                weighted_diffs.append(length * abs(int(x_has_desc) - int(y_has_desc)))
                total_length += length
        
        if total_length == 0:
            return 1.0
            
        return np.sum(weighted_diffs) / total_length

    def calculate_pairwise_distances(self, metric: str) -> pd.DataFrame:
        """
        Calculates pairwise distances between all samples.
        
        Parameters
        ----------
        metric : str
            Distance metric to use for comparison
            
        Returns
        -------
        pd.DataFrame
            Symmetric matrix of pairwise distances
        """
        if metric not in self._metric_methods_map:
            raise ValueError(f"Invalid metric: {metric}. Choose from {list(self._metric_methods_map.keys())}")
        
        n_samples = len(self.sample_names)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self._metric_methods_map[metric](
                    self.normalized_abundance_matrix[i],
                    self.normalized_abundance_matrix[j]
                )
                distances[i, j] = distances[j, i] = dist
        
        return pd.DataFrame(distances, index=self.sample_names, columns=self.sample_names)

    def calculate_all_distances(self) -> dict:
        """
        Calculates all distance metrics for comprehensive community comparison.
        
        Returns
        -------
        dict
            Dictionary of distance matrices for each metric
        """
        return {metric: self.calculate_pairwise_distances(metric) 
                for metric in self._metric_methods_map.keys()}
